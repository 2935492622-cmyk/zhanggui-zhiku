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
