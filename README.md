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
- 外部上传自动脱敏（手机号/身份证号/邮箱/银行卡/姓名字段/地址字段）
- 自动抽取文本（图片走多模态模型 OCR 提取；扫描版 PDF 会自动 OCR 兜底）
- 文本切片 + Embedding + 向量检索（向量库抽象层，当前默认 `local_json`）
- ChatGPT 风格单页对话
- 回答时附带引用证据（来源、rank、相似度 score、chunk_id、证据摘录）
- 患者病程时间线抽取与线性状态编码（`risk_level/risk_score`）
- Claim 级证据约束校验（引用完整性 + 语义重叠）
- 隐私审计日志（上传处理、聊天检索、TTL 过期销毁证明）

## 5. 挑战杯增强点（可演示实物）

- 医学时序认知引擎：
  - 上传病例后自动抽取病程事件（检验、分期、治疗事件）
  - 通过线性状态编码聚合风险，驱动检索路由提示
- 因果/证据约束推理（工程版）：
  - 每条医疗结论应绑定 `[证据#n]`
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
  - 当证据覆盖率过低时会自动触发二次改写（仅保留可证结论）

## 8. 关键配置

- `VECTOR_BACKEND=local_json`：向量库后端（已抽象，当前实现为本地 JSON）
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
- `LITERATURE_TOP_K=5`：每轮检索返回条数（上限5）
- `LITERATURE_TIMEOUT_SECONDS=8`：联网检索超时时间（秒）
- `LITERATURE_MEDICAL_ONCOLOGY_ONLY=true`：仅保留医学肿瘤文献
- `LITERATURE_MIN_RELEVANCE=0.18`：最低相关性阈值（低于阈值直接丢弃）
- `AUTO_EVIDENCE_REWRITE=true`：低充分度时自动二次改写答案
- `EVIDENCE_REWRITE_MIN_COVERAGE=0.75`：触发改写的最低证据覆盖率阈值
- `EVIDENCE_REWRITE_MAX_UNSUPPORTED=1`：触发改写的最大不支持结论阈值

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
- 增加用户身份与隐私隔离
- 增加多轮病程“时间线”结构化抽取
- 后续再加 LoRA 或私有微调模型
