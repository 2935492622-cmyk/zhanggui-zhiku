"""
app/import_process/agent/nodes/node_bge_embedding.py —— 切片向量化（BGE-M3）

【解决什么问题】
    Milvus 存的是向量不是文字。这一步把每个 chunk 变成两组向量：
      - dense（稠密，1024 维）：负责语义相似，"换色带" ≈ "更换碳带"
      - sparse（稀疏，词权重）：负责精确命中型号，"HAK180" 必须一字不差命中
    两组一起用才叫混合检索——这是本项目检索质量的核心卖点。

【为什么 Embedding 留在本地，其他都走云端】
    阿里云百炼的 text-embedding **只有稠密向量**，没有稀疏。
    如果用它，混合检索只能做一半，商品名精确命中的能力直接没了。
    所以生成/rerank/VLM 都走云端，唯独 embedding 留在本地 GPU 跑 BGE-M3。
    你本机 RTX 5060 Laptop / 8GB，BGE-M3 fp16 约 2GB 显存，够用。

【⚠️ 和教程的一个关键差异（不能照抄）】
    教程的 app/lm/embedding_utils.py 用的是
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
    这来自 pymilvus 的 [model] 额外依赖（会再拖进来一大堆包）。
    你的 .venv 里装的是 FlagEmbedding 1.4.2，用这个即可，不用再装任何东西：
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="cuda:0")
        out = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    注意 FlagEmbedding 返回的 sparse 是 {token_id: weight} 字典列表，
    Milvus 的 SPARSE_FLOAT_VECTOR 字段正好吃这个格式，不用自己转 CSR。

【入参 / 出参】
    读：state["chunks"]
    写：state["embeddings_content"]（每个 chunk 带上 dense / sparse）

【跑之前的准备】
    首次运行会从 HuggingFace 下载 BGE-M3 权重（约 2GB）。
    国内建议先设：export HF_ENDPOINT=https://hf-mirror.com
    或者把权重下到本地目录，改 .env 里的 BGE_M3_PATH 指向它。

【当前状态】空壳（return state）。
"""

from app.core.logger import node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    """
    add_running_task(state["task_id"], "node_bge_embedding")

    # TODO: 1) 取 settings.bge_m3_path 加载 BGEM3FlagModel（单例，别每次都加载）
    #      2) 按 settings.embedding_batch_size 批量 encode，同时出 dense + sparse
    #      3) 结果挂回每个 chunk，写进 state["embeddings_content"]
    # 坑：模型加载一次要十几秒，必须做单例缓存，否则 88 本手册每本都重载一次

    add_done_task(state["task_id"], "node_bge_embedding")
    return state
