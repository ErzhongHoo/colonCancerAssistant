"""User authentication and account management.

Uses SQLite for persistent user storage and JWT for stateless authentication.
Passwords are hashed with SHA-256 + per-user salt (no extra dependency needed).
Each user gets a persistent data directory for their vector store and timeline.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jwt


# ---------------------------------------------------------------------------
#  JWT Configuration
# ---------------------------------------------------------------------------
JWT_ALGORITHM = "HS256"
JWT_DEFAULT_EXPIRE_SECONDS = 7 * 24 * 3600  # 7 days


def _get_jwt_secret() -> str:
    """Read JWT_SECRET from env; auto-generate and persist if missing."""
    secret = os.getenv("JWT_SECRET", "").strip()
    if secret:
        return secret
    # Auto-generate a strong secret – printed once so the user can copy it
    secret = secrets.token_hex(32)
    print(f"[auth] JWT_SECRET not found in environment. Auto-generated: {secret}")
    print("[auth] Please add  JWT_SECRET=<value>  to your .env file for persistence across restarts.")
    os.environ["JWT_SECRET"] = secret
    return secret


# ---------------------------------------------------------------------------
#  Data Model
# ---------------------------------------------------------------------------
@dataclass
class UserRecord:
    user_id: str
    username: str
    password_hash: str
    salt: str
    created_at: float
    last_login_at: float = 0.0
    display_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_password(password: str, salt: str) -> str:
    """Hash password with SHA-256 + salt."""
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
#  SQLite helpers
# ---------------------------------------------------------------------------
_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_login_at REAL DEFAULT 0,
    display_name  TEXT DEFAULT ''
);
"""


class UserManager:
    """Manages user accounts with SQLite persistence and JWT tokens."""

    def __init__(self, db_path: Path, user_data_root: Path) -> None:
        self.db_path = db_path
        self.user_data_root = user_data_root
        self.jwt_secret = _get_jwt_secret()
        self.jwt_expire_seconds = int(
            os.getenv("JWT_EXPIRE_SECONDS", str(JWT_DEFAULT_EXPIRE_SECONDS))
        )

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_data_root.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Migrate from legacy JSON file if it exists alongside the new DB
        self._migrate_from_json()

    # ── Database Initialisation ──────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new connection (SQLite connections are cheap)."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute(_CREATE_USERS_TABLE)
            conn.commit()
        finally:
            conn.close()

    def _migrate_from_json(self) -> None:
        """One-time migration from the old users.json file to SQLite."""
        json_path = self.db_path.parent / "users.json"
        if not json_path.exists():
            return

        # Guard: some old deployments accidentally left a SQLite file named users.json.
        try:
            header = json_path.read_bytes()[:16]
            if header.startswith(b"SQLite format 3"):
                backup = json_path.with_suffix(".json.sqlite_legacy")
                index = 1
                while backup.exists():
                    backup = json_path.with_suffix(f".json.sqlite_legacy.{index}")
                    index += 1
                json_path.rename(backup)
                print(f"[auth] Found SQLite users.json, moved to {backup.name}")
                return
        except Exception:
            pass

        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            users = raw.get("users", [])
            if not users:
                # Empty JSON – just rename it out of the way
                json_path.rename(json_path.with_suffix(".json.migrated"))
                return

            conn = self._get_conn()
            try:
                migrated = 0
                for item in users:
                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO users
                               (user_id, username, password_hash, salt, created_at, last_login_at, display_name)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                item["user_id"],
                                item["username"],
                                item["password_hash"],
                                item["salt"],
                                item["created_at"],
                                item.get("last_login_at", 0),
                                item.get("display_name", ""),
                            ),
                        )
                        migrated += 1
                    except Exception as exc:
                        print(f"[auth] Skipping user during migration: {exc}")
                conn.commit()
                print(f"[auth] Migrated {migrated} users from users.json → SQLite")
            finally:
                conn.close()

            # Rename old file so migration doesn't run again
            json_path.rename(json_path.with_suffix(".json.migrated"))
        except Exception as exc:
            print(f"[auth] JSON migration failed (non-fatal): {exc}")

    # ── User CRUD ────────────────────────────────────────────────────────

    def _row_to_user(self, row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            user_id=row["user_id"],
            username=row["username"],
            password_hash=row["password_hash"],
            salt=row["salt"],
            created_at=row["created_at"],
            last_login_at=row["last_login_at"] or 0.0,
            display_name=row["display_name"] or "",
        )

    def _get_user_by_username(self, username: str) -> UserRecord | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()

    def _get_user_by_id(self, user_id: str) -> UserRecord | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()

    def _normalize_username(self, username: str) -> str:
        return (username or "").strip().lower()

    def _validate_username(self, username: str) -> str | None:
        if not username or len(username) < 2:
            return "用户名至少需要2个字符"
        if len(username) > 30:
            return "用户名不能超过30个字符"
        return None

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, username: str, password: str, display_name: str = "") -> tuple[bool, str]:
        """Register a new user. Returns (success, message_or_token)."""
        username = self._normalize_username(username)
        username_error = self._validate_username(username)
        if username_error:
            return False, username_error
        if not password or len(password) < 4:
            return False, "密码至少需要4个字符"

        if self._get_user_by_username(username):
            return False, "该用户名已被注册"

        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        user_id = f"u-{secrets.token_hex(8)}"
        now = time.time()

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO users
                   (user_id, username, password_hash, salt, created_at, last_login_at, display_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, username, password_hash, salt, now, now, display_name or username),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return False, "该用户名已被注册"
        finally:
            conn.close()

        # Create user data directory
        user_dir = self.user_data_root / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Auto-login after registration
        token = self._generate_token(user_id)
        return True, token

    # ── Login ────────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> tuple[bool, str]:
        """Login a user. Returns (success, message_or_token)."""
        username = self._normalize_username(username)
        user = self._get_user_by_username(username)
        if not user:
            return False, "用户名或密码错误"

        password_hash = _hash_password(password, user.salt)
        if password_hash != user.password_hash:
            return False, "用户名或密码错误"

        # Update last_login_at
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE user_id = ?",
                (time.time(), user.user_id),
            )
            conn.commit()
        finally:
            conn.close()

        token = self._generate_token(user.user_id)
        return True, token

    def change_username(self, user_id: str, new_username: str) -> tuple[bool, str]:
        """Change username for the given user_id."""
        user = self._get_user_by_id(user_id)
        if not user:
            return False, "用户不存在"

        new_username = self._normalize_username(new_username)
        username_error = self._validate_username(new_username)
        if username_error:
            return False, username_error
        if new_username == user.username:
            return False, "新用户名不能与当前用户名相同"

        existing = self._get_user_by_username(new_username)
        if existing and existing.user_id != user_id:
            return False, "该用户名已被注册"

        # Keep customized display_name unchanged; only sync when it tracked username.
        old_username = user.username
        should_sync_display_name = (
            not user.display_name
            or self._normalize_username(user.display_name) == old_username
        )
        next_display_name = new_username if should_sync_display_name else user.display_name

        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE users SET username = ?, display_name = ? WHERE user_id = ?",
                (new_username, next_display_name, user_id),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return False, "该用户名已被注册"
        finally:
            conn.close()

        return True, "ok"

    # ── Logout (JWT is stateless – client just discards the token) ────────

    def logout(self, token: str) -> bool:
        """JWT logout is a no-op on the server side.

        The client simply removes the token from localStorage.
        For true revocation you'd need a blocklist (left as future work).
        """
        return True

    # ── Token generation & verification ──────────────────────────────────

    def _generate_token(self, user_id: str) -> str:
        """Generate a JWT token for the given user_id."""
        now = time.time()
        payload = {
            "sub": user_id,
            "iat": int(now),
            "exp": int(now) + self.jwt_expire_seconds,
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=JWT_ALGORITHM)

    def _verify_token(self, token: str) -> str | None:
        """Verify a JWT token and return the user_id, or None if invalid."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[JWT_ALGORITHM])
            return payload.get("sub")
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_user_by_token(self, token: str) -> UserRecord | None:
        """Get user record from a valid JWT token."""
        user_id = self._verify_token(token)
        if not user_id:
            return None
        return self._get_user_by_id(user_id)

    # ── User data directory ──────────────────────────────────────────────

    def get_user_data_dir(self, user_id: str) -> Path:
        """Get the persistent data directory for a user."""
        user_dir = self.user_data_root / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def delete_user_data(self, user_id: str) -> dict[str, Any]:
        """Delete all uploaded data for a user (vector store + timeline)."""
        user_dir = self.user_data_root / user_id
        deleted_files = 0
        if user_dir.exists():
            for f in user_dir.iterdir():
                if f.is_file():
                    f.unlink()
                    deleted_files += 1
        return {"deleted_files": deleted_files, "user_id": user_id}

    def delete_account(self, username: str) -> bool:
        """Delete a user account and all associated data."""
        user = self._get_user_by_username(username)
        if not user:
            return False

        # Remove data directory
        user_dir = self.user_data_root / user.user_id
        if user_dir.exists():
            import shutil
            shutil.rmtree(user_dir, ignore_errors=True)

        # Remove user from database
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user.user_id,))
            conn.commit()
        finally:
            conn.close()

        return True

    def get_user_info(self, user: UserRecord) -> dict[str, Any]:
        """Get safe user info dict (no password/salt)."""
        return {
            "user_id": user.user_id,
            "username": user.username,
            "display_name": user.display_name,
            "created_at": user.created_at,
            "last_login_at": user.last_login_at,
        }
