# 🏥 长智久安

> **RAG 检索增强 × 医学时序认知 × 证据约束推理**

基于大语言模型的结直肠癌智能诊疗辅助系统，融合 RAG（检索增强生成）、病程时序认知引擎和证据约束推理三大核心技术，为医患提供可靠、可追溯、隐私安全的诊疗知识问答服务。

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.129-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-orange)

---

## 📑 目录

- [系统架构](#-系统架构)
- [核心功能](#-核心功能)
- [环境要求](#-环境要求)
- [安装部署](#-安装部署)
- [配置说明](#-配置说明)
- [启动运行](#-启动运行)
- [内部指南维护](#-内部指南维护)
- [API 接口参考](#-api-接口参考)
- [项目结构](#-项目结构)
- [演示脚本](#-演示脚本挑战杯答辩)
- [后续规划](#-后续规划)

---

## 🏗 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                     前端 (Web UI)                         │
│  ChatGPT 风格单页对话 · 证据卡片 · 时间线可视化 · 会话管理  │
└──────────────┬──────────────────────────────┬─────────────┘
               │  HTTP / REST API             │
┌──────────────▼──────────────────────────────▼─────────────┐
│                  FastAPI 后端服务                          │
│                                                           │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────────┐ │
│  │  对话引擎    │ │ 上传 & OCR   │ │  用户认证 (JWT)    │ │
│  └──────┬──────┘ └──────┬───────┘ └────────────────────┘ │
│         │               │                                 │
│  ┌──────▼───────────────▼─────────────────────────────┐  │
│  │            OpenViking 分层检索引擎                    │  │
│  │   L0 (概览索引) → L1 (结构摘要) → L2 (原文段落)      │  │
│  └──────┬─────────────────────────────────┬───────────┘  │
│         │                                 │               │
│  ┌──────▼──────┐  ┌──────────────┐  ┌────▼───────────┐  │
│  │ 内部指南库   │  │ 用户上传临时库 │  │ 本地文献库     │  │
│  │ (持久化)    │  │ (会话/用户级) │  │ (PubMed Agent) │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│                                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │
│  │ 时序认知引擎    │  │ 证据约束推理    │  │ 隐私脱敏    │  │
│  │ Mamba/SSM 编码 │  │ Evidence Guard │  │ & 审计日志  │  │
│  └────────────────┘  └────────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## ✨ 核心功能

### 🔍 双库 RAG 检索增强
- **内部指南库**（持久化）：导入 CSCO/NCCN 等权威指南，长期保存
- **用户上传库**（会话/用户级）：上传 PDF / 图片 / txt 报告，支持登录用户持久化或游客会话临时存储
- 对话检索链路基于 **OpenViking 分层检索引擎**，回答前必须先检索证据

### 🧠 医学时序认知引擎
- 上传病例后自动抽取病程事件（检验、分期、治疗事件）
- 通过 **Mamba 状态空间模型**（可回退 SSM/Linear）聚合风险评分
- 时序编码驱动检索路由提示，提升检索精准度

### 🛡 证据约束推理
- 关键结论绑定证据编号（如 `[证据#1]`），映射到可展开证据片段
- 服务端返回 `evidence_guard` 校验结果（`coverage`、`verified_claims`、`unsupported_claims`）
- 证据不足时明确提示，并给出补充检索关键词

### 🔒 隐私安全与审计
- 上传文本自动脱敏（手机号 / 身份证号 / 邮箱 / 银行卡 / 姓名 / 地址）
- 会话隔离 + TTL 到期自动清理
- `audit_log.jsonl` 完整审计链路，支持 TTL 过期销毁证明查询

### 📚 本地文献 Agent
- 增量抓取 PubMed 论文 → 医学肿瘤过滤 → 结构化 → 向量入库
- 支持后台定时刷新（默认 24h）或手动触发
- 检索返回文献类型、年份、venue、DOI 等结构化信息

### 👤 用户认证系统
- SQLite + JWT 无状态认证，服务器重启不丢失登录态
- 支持注册、登录、修改用户名、数据管理、账户注销
- 每用户独立数据目录隔离

---

## 📋 环境要求

| 组件 | 要求 | 说明 |
|------|------|------|
| **Python** | ≥ 3.11 | 推荐使用 3.11 |
| **包管理器** | [uv](https://docs.astral.sh/uv/) | 推荐；也可使用 `pip` |
| **操作系统** | Linux / macOS / Windows | Linux 推荐 |
| **API Key** | 阿里云百炼 API Key | 用于 LLM、Embedding、OCR 等 |
| **磁盘空间** | ≥ 500 MB | 含向量索引和文献库 |

### 可选依赖

| 组件 | 用途 | 安装条件 |
|------|------|----------|
| **PaddleOCR** | 本地 OCR 引擎 | 需要本地 OCR 时安装 |
| **PaddlePaddle** | PaddleOCR 后端 | 配合 PaddleOCR 使用 |

---

## 🚀 安装部署

### 方式一：使用 uv（推荐）

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd colon_cancer_research_v3_colon

# 2. 创建虚拟环境
uv venv --python 3.11 .venv

# 3. 安装依赖
uv pip install --python .venv/bin/python -r requirements.txt
```

### 方式二：使用 pip

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd colon_cancer_research_v3_colon

# 2. 创建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 安装本地 OCR（可选）

如需启用 PaddleOCR 本地识别：

```bash
# Debian/Ubuntu 服务器建议先安装系统依赖
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1

# 使用 uv
uv pip install --python .venv/bin/python paddlepaddle

# 或使用 pip
pip install paddlepaddle
```

`requirements.txt` 只会安装 `paddleocr` 包；`paddlepaddle` 需要手动安装，否则本地 OCR 和图片脱敏打码预览可能无法正常工作。

安装完成后，建议用下面的命令验证 PaddleOCR 是否可正常初始化：

```bash
python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ch', use_textline_orientation=True); print('ok')"
```

> **提示**：如不安装 PaddleOCR，系统会自动回退到阿里云在线 OCR 或视觉模型 OCR。

---

## ⚙ 配置说明

### 基本配置

1. 将 `.env.example` 复制为 `.env`：

```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入必要配置。

### 环境变量详解

#### 🔑 必需配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | — | 阿里云百炼 API Key（**必填**，也支持 `BAILIAN_API_KEY` / `APIKEY` / `apikey`） |
| `JWT_SECRET` | 自动生成 | JWT 签名密钥（**生产环境必填**，否则每次重启后用户需重新登录） |

#### 🤖 模型配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 基础 URL |
| `CHAT_MODEL` | `qwen-plus` | 对话模型 |
| `VISION_MODEL` | `qwen-vl-max` | 多模态视觉模型 |
| `EMBEDDING_PROVIDER` | `remote` | 向量嵌入来源：`remote` / `local` |
| `EMBEDDING_MODEL` | `text-embedding-v3` | 向量嵌入模型 |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地 embedding 模型；可填 Hugging Face 仓库名或本地目录，离线环境建议直接填本地目录 |
| `LOCAL_EMBEDDING_DEVICE` | `cpu` | 本地 embedding 设备，如 `cpu` / `cuda` |
| `LOCAL_EMBEDDING_NORMALIZE` | `true` | 本地向量是否归一化 |
| `LOCAL_EMBEDDING_CACHE_DIR` | 空 | 本地 embedding 模型缓存目录（可选） |
| `LOCAL_EMBEDDING_LOCAL_ONLY` | `false` | 仅从本地磁盘加载模型，不访问 Hugging Face |
| `LOCAL_EMBEDDING_TRUST_REMOTE_CODE` | `false` | 是否允许加载模型仓库中的自定义代码；只有模型明确要求时再开启 |

#### 📷 OCR 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_PROVIDER` | `auto` | OCR 引擎：`auto`（自动选择）/ `paddle` / `aliyun` / `vision` |
| `PADDLE_OCR_LANG` | `ch` | PaddleOCR 语言包 |
| `ALIYUN_OCR_MODEL` | `qwen-vl-ocr-latest` | 百炼在线 OCR 模型 |
| `ALIYUN_OCR_MIN_PIXELS` | `3072` | 在线 OCR 最小像素约束（0 表示不设置） |
| `ALIYUN_OCR_MAX_PIXELS` | `8388608` | 在线 OCR 最大像素约束（0 表示不设置） |
| `PDF_OCR_MAX_PAGES` | `500` | 扫描版 PDF 的 OCR 最大页数 |

> **OCR 引擎选择优先级**（`auto` 模式）：PaddleOCR → 百炼在线 OCR → 视觉模型 OCR

#### 🔐 安全与隐私配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_UPLOAD_DEID` | `true` | 是否对上传文本自动脱敏 |
| `SAVE_UPLOAD_ORIGINALS` | `false` | 是否保存上传原件（用于抽取对照与审计） |
| `SESSION_TTL_SECONDS` | `1800` | 游客会话过期时间（秒） |
| `MANUAL_SESSION_CLEAR_ONLY` | `true` | 仅手动清除会话数据（默认开启） |
| `INTERNAL_RAG_TOKEN` | 空 | 内部指南导入接口令牌（可选） |
| `JWT_EXPIRE_SECONDS` | `604800` | JWT 令牌过期时间（默认 7 天） |

#### 🧬 时序认知引擎配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TIMELINE_ENCODER` | `mamba` | 病程状态编码器：`mamba` / `ssm` / `linear` |
| `MAMBA_MODEL_PATH` | `models/mamba_timeline_v1.npz` | Mamba 编码器权重路径（不存在时自动初始化） |
| `ENABLE_LLM_TIMELINE` | `true` | 是否启用 LLM 辅助时间线抽取 |
| `ENABLE_IMAGE_DATE_VLM` | `true` | 是否用视觉模型提取图片中的日期 |

#### 🗂 OpenViking 检索配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENVIKING_NATIVE_ENABLED` | `true` | 启用官方 OpenViking SDK 作为主检索通道 |
| `OPENVIKING_NATIVE_STORAGE_PATH` | `./data/openviking_native` | OpenViking 本地存储目录 |
| `OPENVIKING_NATIVE_AGFS_PORT` | `1833` | 内嵌 AGFS 端口（多实例需错开） |
| `OPENVIKING_NATIVE_CALL_TIMEOUT_SECONDS` | `12` | 单次原生检索/读取调用超时；设为 `0` 或 `none` 可禁用 |
| `OPENVIKING_NATIVE_COOLDOWN_SECONDS` | `30` | 原生调用超时后的熔断冷却时间 |
| `OPENVIKING_RAG_L1_BUDGET` | `6` | 默认 L1 概览读取数量 |
| `OPENVIKING_RAG_L2_BUDGET` | `2` | 默认 L2 原文读取数量 |
| `OPENVIKING_RAG_DEEP_L1_BUDGET` | `10` | 深入模式 L1 预算 |
| `OPENVIKING_RAG_DEEP_L2_BUDGET` | `4` | 深入模式 L2 预算 |
| `OPENVIKING_INTERNAL_COMPLEX_SEARCH` | `false` | 内部库复杂问题是否启用 `search` |
| `OPENVIKING_LEGACY_DUAL_WRITE` | `false` | 是否同时写入旧版存储 |

> **本地 embedding 注意**：当 `EMBEDDING_PROVIDER=local` 时，系统会自动停用 OpenViking 原生索引，回退到项目内置的本地分层检索路径，因此不再依赖阿里云 embedding 配额。

#### 📚 文献 Agent 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_LOCAL_LITERATURE_AGENT` | `true` | 启用后台本地论文 Agent |
| `LITERATURE_AGENT_TOPIC_QUERY` | 结直肠癌相关关键词 | Agent 跟踪检索主题 |
| `LITERATURE_AGENT_BOOTSTRAP_MAX_RESULTS` | `120` | 单次增量更新抓取上限 |
| `LITERATURE_AGENT_REFRESH_HOURS` | `24` | 自动刷新间隔（小时） |
| `ENABLE_WEB_LITERATURE` | `false` | 是否启用联网检索兜底 |
| `LITERATURE_PROVIDER` | `pubmed` | 学术来源 |
| `LITERATURE_TOP_K` | `8` | 每轮检索返回条数 |
| `LITERATURE_TIMEOUT_SECONDS` | `8` | 联网检索超时（秒） |
| `LITERATURE_MEDICAL_ONCOLOGY_ONLY` | `true` | 仅保留医学肿瘤文献 |
| `LITERATURE_MIN_RELEVANCE` | `0.18` | 最低相关性阈值 |

#### 🔧 其他配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VECTOR_BACKEND` | `local_json` | 向量存储后端（当前默认本地 JSON） |
| `CHAT_COMPLETION_TIMEOUT_SECONDS` | `120` | 对话生成超时（设为 `0` 或 `none` 禁用） |
| `EMBED_CACHE_MAX` | `2048` | 全局 Embedding 缓存条目上限 |

### `.env` 示例

```env
# ==================== 必填 ====================
DASHSCOPE_API_KEY=your_api_key_here
JWT_SECRET=your_strong_random_secret_here

# ==================== 模型 ====================
CHAT_MODEL=qwen-plus
VISION_MODEL=qwen-vl-max
EMBEDDING_PROVIDER=remote
EMBEDDING_MODEL=text-embedding-v3
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
LOCAL_EMBEDDING_DEVICE=cpu
LOCAL_EMBEDDING_NORMALIZE=true
LOCAL_EMBEDDING_CACHE_DIR=
LOCAL_EMBEDDING_LOCAL_ONLY=false
LOCAL_EMBEDDING_TRUST_REMOTE_CODE=false

# ==================== OCR ====================
OCR_PROVIDER=auto
PADDLE_OCR_LANG=ch

# ==================== 安全 ====================
ENABLE_UPLOAD_DEID=true
SAVE_UPLOAD_ORIGINALS=false
INTERNAL_RAG_TOKEN=

# ==================== 时序引擎 ====================
TIMELINE_ENCODER=mamba

# ==================== 文献 Agent ====================
ENABLE_LOCAL_LITERATURE_AGENT=true
LITERATURE_PROVIDER=pubmed
LITERATURE_MEDICAL_ONCOLOGY_ONLY=true

# ==================== 认证 ====================
JWT_EXPIRE_SECONDS=604800
```

### 切换到本地 Embedding

1. 安装本地 embedding 依赖：

```bash
uv pip install --python .venv/bin/python sentence-transformers
```

或：

```bash
pip install sentence-transformers
```

2. 准备本地模型目录。模型不要求放在项目目录里，只要进程可读即可；推荐放在独立目录，例如 `/opt/models/...`、`/data/models/...` 或你的家目录下。

如果你用 `hf` 下载并希望放在固定目录，推荐这样做：

```bash
hf download BAAI/bge-small-zh-v1.5 --local-dir /opt/models/bge-small-zh-v1.5
```

如果你已经用 `hf` 下载到了默认缓存，也可以直接复用 Hugging Face 缓存目录；`LOCAL_EMBEDDING_MODEL` 需要指向 `snapshots/<hash>` 这一层，而不是 `models--...` 根目录，例如：

```text
~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/<snapshot-id>
```

3. 在 `.env` 中设置。离线环境建议直接填本地目录，并开启 `LOCAL_EMBEDDING_LOCAL_ONLY=true`：

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=/opt/models/bge-small-zh-v1.5
LOCAL_EMBEDDING_DEVICE=cpu
LOCAL_EMBEDDING_NORMALIZE=true
LOCAL_EMBEDDING_LOCAL_ONLY=true
LOCAL_EMBEDDING_TRUST_REMOTE_CODE=false
```

如果你直接复用 HF 缓存，写法类似：

```env
LOCAL_EMBEDDING_MODEL=/home/your-user/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/<snapshot-id>
```

4. 重启服务。

5. 重新导入内部指南，并重新上传需要检索的资料。

这是必要步骤，因为旧索引里的向量仍然是按原来的 embedding 生成的。`/admin` 页面上传文件只会把原件保存到内部指南目录；真正重建向量索引需要再点一次“立即同步知识库”。

6. 验证是否切换成功。

常见验证方式：

- 启动日志里不再出现对阿里云 embedding 接口的请求。
- `LOCAL_EMBEDDING_LOCAL_ONLY=true` 时，不再尝试访问 Hugging Face。
- `/admin` 同步完成后，日志里会出现 `文件名: N chunks (OK)`，说明内部指南已经重新切片并入库。

> 切到本地 embedding 后，问答和 OCR 仍然可以继续使用阿里云模型；变化的只是向量生成这一步。

> 如果你看到 `Network is unreachable` 且日志里还在访问 `https://huggingface.co/...`，通常说明 `LOCAL_EMBEDDING_MODEL` 填的是仓库名而不是本地目录，或者目录指到了 `models--...` 根目录而不是 `snapshots/<hash>`。

> 💡 **生成安全的 JWT 密钥**：
> ```bash
> python3 -c "import secrets; print(secrets.token_hex(32))"
> ```

---

## 🟢 启动运行

### 开发模式（推荐）

```bash
# 使用 uv
uv run --python .venv/bin/python uvicorn app.main:app --reload --port 8000

# 或者激活虚拟环境后
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### 生产模式

```bash
# 多 Worker 模式（根据 CPU 核心数调整）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 访问应用

启动后打开浏览器访问：

```
http://127.0.0.1:8000
```

即可进入 ChatGPT 风格的对话界面。

### 首次启动检查清单

1. ✅ `.env` 中填入了有效的 `DASHSCOPE_API_KEY`
2. ✅ `.env` 中设置了 `JWT_SECRET`（生产环境）
3. ✅ 依赖已安装（`requirements.txt` 中的所有包）
4. ✅ 如需本地 PaddleOCR，已手动安装 `paddlepaddle`，并在 Debian/Ubuntu 上执行过 `sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1`
5. ✅ `data/` 目录可写（应用会在此目录自动创建数据库和索引文件）

---

## 📖 内部指南维护

### 1. 放置指南文件

将指南文件放到 `data/guidelines/` 目录，支持以下格式：
- PDF 文档
- 图片文件（JPG / PNG 等）
- TXT 文本

### 2. 触发导入

#### 方式一：API 触发

```bash
curl -X POST "http://127.0.0.1:8000/api/internal/import-guidelines" \
  -H "Content-Type: application/json" \
  -d '{"reset": true}'
```

- `reset: true`：重建内部指南库（推荐首次使用）
- 如设置了 `INTERNAL_RAG_TOKEN`，需额外加请求头：`X-Internal-Token: <token>`

#### 方式二：Web UI

在首页左侧"内部维护"面板中点击 **"导入内部指南"**，支持填写令牌和选择是否 reset。

---

## 📡 API 接口参考

### 对话 & 聊天

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 主对话接口，返回回答、证据、时间线状态、evidence_guard 等 |
| `/api/chat/progress` | GET | 查询对话进度状态 |

### 文件上传

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传文件（PDF / 图片 / txt），自动 OCR + 切片 + 入库 |
| `/api/upload/debug` | GET | 查询上传源的原件/抽取文本/时间线对照信息 |

### 文献检索

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/literature/refresh` | POST | 执行本地论文 Agent 增量更新（参数：`force`, `max_results`） |
| `/api/literature/search` | GET | 检索本地文献库（参数：`q`, `top_k`） |

### 时间线

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/session/timeline` | GET | 返回当前会话病程事件、时序编码和摘要 |

### 审计

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/audit/recent` | GET | 返回最近审计日志（参数：`limit`） |
| `/api/audit/ttl-proof` | GET | 返回 TTL 过期清理证明（参数：`session_id`） |

### 用户认证

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/auth/register` | POST | 注册新用户 |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/logout` | POST | 登出 |
| `/api/auth/me` | GET | 获取当前用户信息 |

### 用户管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/user/change-username` | POST | 修改用户名 |
| `/api/user/delete-data` | POST | 清除用户上传数据 |
| `/api/user/delete-upload` | POST | 删除指定上传文件 |
| `/api/user/delete-account` | POST | 注销账户（永久） |

### 会话管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/sessions` | GET | 获取用户历史会话列表 |
| `/api/sessions/new` | POST | 创建新会话 |
| `/api/session/clear` | POST | 清除当前会话数据 |

### 内部维护

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/internal/import-guidelines` | POST | 导入内部指南（参数：`reset`） |

> **认证说明**：需要登录的接口通过 `Authorization: Bearer <token>` 请求头传递 JWT 令牌。

---

## 📂 项目结构

```
colon_cancer_research_v3_colon/
├── app/                          # 后端核心代码
│   ├── main.py                   # FastAPI 主入口 & 路由定义
│   ├── auth.py                   # 用户认证（SQLite + JWT）
│   ├── audit.py                  # 审计日志模块
│   ├── evidence_guard.py         # 证据约束校验
│   ├── literature_agent.py       # 本地文献 Agent（PubMed 增量抓取）
│   ├── literature_search.py      # 文献检索逻辑
│   ├── llm.py                    # LLM / Embedding / OCR 调用封装
│   ├── mamba_encoder.py          # Mamba 状态空间模型编码器
│   ├── ssm_encoder.py            # SSM 编码器（回退方案）
│   ├── ocr.py                    # PaddleOCR 本地 OCR 封装
│   ├── openviking.py             # OpenViking 分层检索引擎
│   ├── privacy.py                # 隐私脱敏处理
│   ├── rag.py                    # 文本切片 & 检索链路
│   ├── timeline.py               # 病程时间线抽取 & 编码
│   └── vector_store.py           # 向量存储抽象层
├── web/                          # 前端页面
│   ├── index.html                # 主界面（ChatGPT 风格对话 UI）
│   └── admin.html                # 管理页面
├── data/                         # 数据目录（运行时生成）
│   ├── guidelines/               # 内部指南文件存放
│   ├── openviking_native/        # OpenViking 本地索引
│   ├── user_data/                # 用户数据目录（按用户隔离）
│   ├── users.db                  # 用户数据库（SQLite，自动创建）
│   ├── audit_log.jsonl           # 审计日志
│   ├── literature_store.jsonl    # 结构化文献库
│   └── literature_vector_store.json  # 文献向量索引
├── models/                       # 模型权重
│   └── mamba_timeline_v1.npz     # Mamba 时序编码器权重
├── docs/                         # 文档资料
│   └── architecture/             # 系统架构图（Mermaid / PNG / PDF）
├── .env.example                  # 环境变量模板
├── .env                          # 环境变量（不纳入版本控制）
├── requirements.txt              # Python 依赖清单
└── README.md                     # 项目说明
```

---

## 📚 一键构建本地文献库

运行以下命令执行"增量抓取 PubMed → 医学肿瘤过滤 → 结构化 → 向量入库"全流程：

```bash
uv run --python .venv/bin/python -m app.literature_agent \
  --topic "(colorectal cancer OR colon cancer OR rectal cancer) AND (clinical OR guideline OR trial OR treatment)" \
  --max-results 120
```

执行后会产出：
- `data/literature_store.jsonl` — 结构化文献库
- `data/literature_vector_store.json` — 文献向量索引
- `data/literature_agent_state.json` — 增量状态记录

---

## 🎤 演示脚本（挑战杯答辩）

### 步骤 1：上传病例

上传 1 份随访报告 + 1 份病理报告，展示：
- ✅ 上传自动脱敏命中统计
- ✅ 自动提取病程时间线和风险状态

### 步骤 2：智能问答

提问 *"目前分期风险如何，下一步治疗路径建议？"*

展示：
- ✅ 回答中的 `[证据#n]` 引用标注
- ✅ `evidence_guard.coverage` 与失败 claim 分析

### 步骤 3：隐私审计

等待会话过期或手动清除后，查询：

```bash
GET /api/audit/ttl-proof?session_id=<id>
```

展示：
- ✅ 会话销毁审计证据（TTL 过期证明）

---

## 🗃 用户认证与数据库

### 技术方案

| 组件 | 技术 | 说明 |
|------|------|------|
| 用户数据存储 | **SQLite** | 嵌入式数据库，零额外依赖 |
| 认证令牌 | **JWT** | 无状态令牌，重启后不丢失 |
| 密码安全 | SHA-256 + per-user salt | 每用户独立随机盐值 |
| 用户数据 | 文件系统 | 每用户独立目录 `data/user_data/{user_id}/` |

### 数据库说明

- 数据库文件 `data/users.db`，应用启动时**自动创建**
- Python 标准库自带 `sqlite3` 模块，**无需额外安装**
- 使用 WAL 模式支持并发读写

### 从旧版本升级

如之前使用 JSON 文件存储，系统会**自动迁移**：
1. 首次启动检测到 `data/users.json`
2. 自动迁移到 `data/users.db`
3. 旧文件重命名为 `data/users.json.migrated`

> 旧版内存 Token 无法迁移，升级后需重新登录。

---

## 🗺 后续规划

- [ ] 切换到生产级向量数据库（Milvus / pgvector / Elasticsearch）
- [x] ~~用户身份与隐私隔离~~ ✅ 已完成（SQLite + JWT）
- [ ] JWT Token 黑名单机制（支持主动吊销令牌）
- [ ] 升级到 PostgreSQL 等生产级数据库
- [ ] 增强多轮病程"时间线"结构化抽取
- [ ] LoRA / 私有微调模型适配

---

## 📄 许可证

本项目为学术研究用途，仅供内部使用和学习交流。

---

<p align="center">
  <sub>Built with ❤️ for medical AI research</sub>
</p>
