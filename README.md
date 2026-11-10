# 掌柜智库

使用 uv 管理 Python 环境和依赖，Python 版本为 3.12。

## 初始化环境

```bash
uv sync
```

## 常用命令

```bash
# 添加项目依赖
uv add <包名>

# 添加开发依赖
uv add --dev <包名>

# 在项目虚拟环境中运行 Python
uv run python

# 激活虚拟环境（可选）
source .venv/bin/activate
```

提交 `pyproject.toml`、`uv.lock` 和 `.python-version`，以便复现项目环境。
本地虚拟环境和 `.env` 文件已加入 Git 忽略规则。

## 下载本地 BGE-M3 模型

在项目 `.env` 中配置模型保存目录和运行设备，例如当前 Linux 环境：

```ini
BGE_M3_PATH=/home/ava/ai_models/BAAI/bge-m3
BGE_M3=BAAI/bge-m3
BGE_DEVICE=cpu
BGE_FP16=0
```

初始化环境后，在项目根目录运行：

```bash
uv run python app/tool/download_bgem3.py
```

脚本读取项目 `.env`，将模型直接下载到 `BGE_M3_PATH`。下载完成后，
项目首次生成向量时会加载该目录中的模型，无需单独启动模型 API 服务。
仅修改配置并不会下载模型。CPU 运行时保持 `BGE_FP16=0`。

完整查询流程还使用本地重排序模型。在 `.env` 中将 `BGE_RERANKER_LARGE`
设为实际模型保存目录（当前为 `/home/ava/ai_models/BAAI/bge-reranker-large`），
然后下载：

```bash
uv run python app/tool/download_reranker.py
```
