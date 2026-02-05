# 结直肠病例助手 MVP（RAG）

目标：先跑通“上传病例/报告 -> 检索增强 -> 对话回答（科普+治疗路径建议）”。

## 1. 安装（uv 管理虚拟环境与依赖）

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 2. 配置

1. 将 `.env.example` 复制为 `.env`
2. 填入你的阿里云百炼 API Key（已兼容 `DASHSCOPE_API_KEY` / `BAILIAN_API_KEY`）

## 3. 启动

```bash
uv run --python .venv/bin/python uvicorn app.main:app --reload --port 8000
```

打开 `http://127.0.0.1:8000`

## 4. MVP 功能范围

- 上传 PDF / 图片 / txt 报告
- 自动抽取文本（图片走多模态模型 OCR 提取）
- 文本切片 + Embedding + 本地向量库（JSON 持久化）
- ChatGPT 风格单页对话
- 回答时附带引用片段来源

## 5. 后续可扩展

- 切到正式向量数据库（Milvus / pgvector / Elasticsearch）
- 增加用户身份与隐私隔离
- 增加多轮病程“时间线”结构化抽取
- 后续再加 LoRA 或私有微调模型
