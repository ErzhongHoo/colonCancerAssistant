# 结直肠病例助手（RAG + 时序认知 + 证据约束）

目标：先跑通“内部指南维护 + 用户病例临时上传 -> 检索增强 -> 对话回答（科普+治疗路径建议）”。

## 1. 安装（uv 管理虚拟环境与依赖）

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

如需启用本地 OCR（PaddleOCR），还需要安装 `paddlepaddle`（CPU 版示例）：

```bash
uv pip install --python .venv/bin/python paddlepaddle
```

## 2. 配置

1. 将 `.env.example` 复制为 `.env`
2. 填入你的阿里云百炼 API Key（已兼容 `DASHSCOPE_API_KEY` / `BAILIAN_API_KEY`）

## 3. 启动

```bash
uv run --python .venv/bin/python uvicorn app.main:app --reload --port 8000
```

打开 `http://127.0.0.1:8000`

## 4. 系统能力

- 内部 RAG（持久化）：指南/共识文档入库后长期保存
- 外部 RAG（临时会话）：用户上传 PDF / 图片 / txt 报告，仅当前页面会话有效
- 对话检索链路已切换为 **OpenViking-only**：回答前必须先 `search/find`，并仅基于 OpenViking 证据作答
- 外部上传自动脱敏（手机号/身份证号/邮箱/银行卡/姓名字段/地址字段）
- 自动抽取文本（图片走多模态模型 OCR 提取；扫描版 PDF 会自动 OCR 兜底）
- 文本切片 + Embedding + 向量检索（向量库抽象层，当前默认 `local_json`）
- ChatGPT 风格单页对话
- 回答采用“关键主张引用”策略：仅在关键结论/关键数字处标注少量证据（如 `[证据#1]`），详细来源在证据卡片查看
- 回答结构按问题复杂度自适应（简单问题直答，复杂问题再结构化），不固定“三段式”模板
- 患者病程时间线抽取与线性状态编码（`risk_level/risk_score`）
- 证据不足时会明确说明“证据不足”，并在必要时给出补充检索关键词
- 隐私审计日志（上传处理、聊天检索、TTL 过期销毁证明）

## 5. 挑战杯增强点（可演示实物）

- 医学时序认知引擎：
  - 上传病例后自动抽取病程事件（检验、分期、治疗事件）
  - 通过线性状态编码聚合风险，驱动检索路由提示
- 因果/证据约束推理（工程版）：
  - 关键结论应绑定证据编号（如 `[证据#1]`），并可映射到下方可展开证据片段
  - 服务端返回 `evidence_guard` 结果（`coverage`、`verified_claims`、`unsupported_claims`）
- 隐私沙箱可验证闭环：
  - 会话隔离、TTL 到期自动清理
  - `audit_log.jsonl` 记录过期销毁事件，可通过接口查询“TTL 证明”

## 6. 内部指南维护

1. 将指南文件放到 `data/guidelines/`（支持 PDF / 图片 / txt）
2. 触发导入：

```bash
curl -X POST "http://127.0.0.1:8000/api/internal/import-guidelines" \
  -H "Content-Type: application/json" \
  -d '{"reset": true}'
```

- `reset: true` 表示重建内部指南库（推荐）
- 如果设置了 `INTERNAL_RAG_TOKEN`，需额外加请求头：`X-Internal-Token: <token>`
- 也可以直接在首页左侧“内部维护”卡片里点击“导入内部指南”（支持填写令牌和选择是否 reset）

## 7. 新增 API（增强能力）

- `POST /api/literature/refresh?force=true&max_results=120`
  - 执行“阶段1-4”本地论文Agent流程：增量抓取 PubMed -> 医学肿瘤过滤 -> 结构化 -> 写入本地文献向量库
- `GET /api/literature/search?q=<query>&top_k=5`
  - 优先检索本地文献库（后台Agent每日可增量更新），无结果时可按配置回退联网
  - 返回文献类型（clinical_trial/meta_or_systematic_review/guideline/...）、年份、venue、doi、url、relevance

- `GET /api/session/timeline`
  - 返回当前会话病程事件、时序编码状态和摘要
- `GET /api/audit/recent?limit=50`
  - 返回最近审计日志
- `GET /api/audit/ttl-proof?session_id=<id>`
  - 返回 TTL 过期清理证明事件
- `POST /api/chat`
  - 额外返回 `timeline_state`、`timeline_summary`、`retrieval_hint`、`evidence_guard`
  - 当证据不足时会明确返回“证据不足”并给出下一步检索关键词

## 8. 关键配置

- `VECTOR_BACKEND=local_json`：用于上传切片缓存（对话检索主链路已改为 OpenViking）
- `ENABLE_UPLOAD_DEID=true`：是否对外部用户上传文本先脱敏再入库
- `OCR_PROVIDER=auto|paddle|vision`：OCR 引擎选择（默认 `auto`，先 PaddleOCR 再回退视觉模型）
- `PADDLE_OCR_LANG=ch`：PaddleOCR 语言包
- 内部指南导入会自动使用“逐字转写”OCR策略，避免被“检验项模板”误抽取
- `PDF_OCR_MAX_PAGES=500`：扫描版 PDF 的 OCR 最大页数
- `SESSION_TTL_SECONDS=1800`：用户会话临时库过期时间（秒，手动清除模式下仅保留配置）
- `MANUAL_SESSION_CLEAR_ONLY=true`：仅手动清除会话数据（默认开启）
- `INTERNAL_RAG_TOKEN=`：内部导入接口令牌（可选）
- `TIMELINE_ENCODER=linear|ssm|mamba`：病程状态编码器（默认 `mamba`，支持 `linear`、`ssm` 回退）
- `MAMBA_MODEL_PATH=models/mamba_timeline_v1.npz`：Mamba 编码器权重路径（不存在时会自动初始化 bootstrap 权重）
- `ENABLE_LOCAL_LITERATURE_AGENT=true`：启用后台本地论文Agent（推荐）
- `LITERATURE_AGENT_TOPIC_QUERY=...`：后台Agent跟踪主题（建议限定到结直肠肿瘤）
- `LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS=120`：单次增量更新抓取上限
- `LITERATURE_AGENT_REFRESH_HOURS=24`：本地文献库自动刷新间隔（小时）
- `ENABLE_WEB_LITERATURE=false`：是否启用联网检索兜底
- `LITERATURE_PROVIDER=pubmed`：学术来源（默认仅医学数据库 PubMed）
- `LITERATURE_TOP_K=8`：每轮检索返回目标条数（动态返回，不再强制5条）
- `LITERATURE_TIMEOUT_SECONDS=8`：联网检索超时时间（秒）
- `LITERATURE_MEDICAL_ONCOLOGY_ONLY=true`：仅保留医学肿瘤文献
- `LITERATURE_MIN_RELEVANCE=0.18`：最低相关性阈值（低于阈值直接丢弃）
- `AUTO_EVIDENCE_REWRITE=true`：旧链路参数（OpenViking-only 对话路径下不生效）
- `EVIDENCE_REWRITE_MIN_COVERAGE=0.75`：旧链路参数（OpenViking-only 对话路径下不生效）
- `EVIDENCE_REWRITE_MAX_UNSUPPORTED=1`：旧链路参数（OpenViking-only 对话路径下不生效）
- `OPENVIKING_NATIVE_ENABLED=true`：启用官方 OpenViking SDK 作为分层检索主通道（异常时自动回退本地实现）
- `OPENVIKING_NATIVE_STORAGE_PATH=./data/openviking_native`：官方 OpenViking 本地存储目录
- `OPENVIKING_NATIVE_AGFS_PORT=1833`：官方 OpenViking 内嵌 AGFS 端口（多实例部署时需错开）
- `OPENVIKING_LEGACY_DUAL_WRITE=false`：是否同时写入 `openviking_layers.json`（默认关闭以减少重复存储；native 失败时仍会自动回退写入）
- `OPENVIKING_RAG_L1_BUDGET=6`：默认最多读取 L1 概览数量
- `OPENVIKING_RAG_L2_BUDGET=2`：默认最多读取 L2 原文数量
- `OPENVIKING_RAG_DEEP_L1_BUDGET=10`：用户要求深入时的 L1 预算
- `OPENVIKING_RAG_DEEP_L2_BUDGET=4`：用户要求深入时的 L2 预算
- `JWT_SECRET=<your_secret>`：JWT 签名密钥（详见下方认证章节）
- `JWT_EXPIRE_SECONDS=604800`：JWT 令牌过期时间（默认 7 天）

## 8.1 用户认证与数据库

### 技术方案

系统采用 **SQLite + JWT** 实现用户认证：

| 组件 | 技术 | 说明 |
|---|---|---|
| 用户数据存储 | **SQLite** | 嵌入式数据库，无需单独安装或启动服务 |
| 认证令牌 | **JWT (JSON Web Token)** | 无状态令牌，服务器重启后登录状态不丢失 |
| 密码安全 | SHA-256 + per-user salt | 每个用户独立的随机盐值 |
| 用户数据 | 文件系统 | 每个用户独立目录 `data/user_data/{user_id}/` |

### 数据库说明

SQLite 是**嵌入式数据库**，不需要单独安装或启动数据库服务器：

- 数据库文件位于 `data/users.db`，应用启动时自动创建
- Python 标准库自带 `sqlite3` 模块，**零额外依赖**
- 使用 WAL 模式支持并发读写

### JWT 配置

在 `.env` 中配置 JWT 相关参数：

```env
# JWT 签名密钥（必须设置，否则每次重启会自动生成新密钥，导致已有令牌失效）
JWT_SECRET=your_strong_random_secret_here

# JWT 令牌过期时间，单位秒（默认 604800 = 7天）
JWT_EXPIRE_SECONDS=604800
```

> **注意**：生产环境中请务必在 `.env` 中设置一个强随机密钥作为 `JWT_SECRET`。如果未设置，系统会在启动时自动生成一个临时密钥，但每次重启后所有用户都需要重新登录。
>
> 可以用以下命令生成一个安全的密钥：
> ```bash
> python3 -c "import secrets; print(secrets.token_hex(32))"
> ```

### 从旧版本升级

如果你之前使用了旧版（JSON 文件存储 + 内存 Token），升级时系统会**自动完成数据迁移**：

1. 首次启动时，检测到 `data/users.json` 文件存在
2. 自动将所有用户数据迁移到 `data/users.db`（SQLite）
3. 旧文件重命名为 `data/users.json.migrated`（作为备份）
4. 迁移完成后，后续启动不再触发迁移

> **注意**：旧版的内存 Token 无法迁移（它们本来就不持久化），升级后所有用户需要重新登录。

### 认证 API

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/auth/register` | POST | 注册新用户，返回 JWT 令牌 |
| `/api/auth/login` | POST | 用户登录，返回 JWT 令牌 |
| `/api/auth/logout` | POST | 登出（客户端清除令牌即可） |
| `/api/auth/me` | GET | 获取当前登录用户信息和数据统计 |
| `/api/user/change-username` | POST | 修改当前登录用户的用户名 |
| `/api/user/delete-data` | POST | 清除当前用户的所有上传数据 |
| `/api/user/delete-account` | POST | 注销账户（永久删除） |
| `/api/user/delete-upload` | POST | 删除指定上传文件的数据 |

所有需要认证的接口通过 `Authorization: Bearer <token>` 请求头传递 JWT 令牌。

## 9. 当前已完成

- 已接入阿里云百炼兼容 OpenAI API（对话、向量、视觉）
- `.env` 支持 `DASHSCOPE_API_KEY` / `BAILIAN_API_KEY` / `APIKEY` / `apikey`
- 修复 `favicon.ico` 404 噪声日志
- 上传接口支持单文件容错，失败会返回具体错误原因
- 兼容部分浏览器图片上传的 `application/octet-stream` MIME
- 内外双库架构：内部持久化 + 用户临时会话（刷新页面后用户资料自动失效）
- 外部上传自动脱敏并返回脱敏命中统计
- 已完成 Git 初始化与首个提交（`20347ca`）

## 10. 演示脚本（挑战杯答辩）

1. 上传 1 份随访报告 + 1 份病理报告，展示：
   - 上传脱敏命中统计
   - 自动提取病程时间线和风险状态
2. 提问“目前分期风险如何，下一步治疗路径建议？”
   - 展示回答中的 `[证据#n]` 标注
   - 展示 `evidence_guard.coverage` 与失败 claim
3. 等待会话过期或手动触发过期场景后，查询：
   - `GET /api/audit/ttl-proof` 显示会话销毁审计证据

## 12. 一键跑完阶段1-4

```bash
uv run --python .venv/bin/python -m app.literature_agent \
  --topic "(colorectal cancer OR colon cancer OR rectal cancer) AND (clinical OR guideline OR trial OR treatment)" \
  --max-results 120
```

执行后会产出：
- `data/literature_store.jsonl`（结构化文献库）
- `data/literature_vector_store.json`（文献向量索引）
- `data/literature_agent_state.json`（增量状态）

## 11. 后续可扩展

- 切到正式向量数据库（Milvus / pgvector / Elasticsearch）
- ~~增加用户身份与隐私隔离~~ 已完成（SQLite + JWT 认证）
- JWT Token 黑名单机制（支持主动吊销令牌）
- 升级到 PostgreSQL 等生产级数据库（当前 SQLite 适用于中小规模）
- 增加多轮病程“时间线”结构化抽取
- 后续再加 LoRA 或私有微调模型
