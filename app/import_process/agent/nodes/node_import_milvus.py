"""
app/import_process/agent/nodes/node_import_milvus.py —— 写入 Milvus 双层索引

【解决什么问题】
    把向量化的切片真正写进向量库，并且要保证**幂等**：
    同一本手册重复导入两次，库里不能有两份（否则检索会出重复结果、
    统计数字也不对）。幂等靠 item_name / file_title 做标识：
    写入前先按标识删掉旧数据，再插新的。

【双层 collection（教程指定的字段名，照抄不改名）】
    kb_item_names  文档级：pk / file_title / item_name / dense / sparse
                   作用 = 一级筛选器，先定位「是哪本手册」
    kb_chunks      切片级：chunk_id / content / title / parent_title / part
                           / file_title / item_name / dense / sparse
                   作用 = 精准匹配层，再定位「是哪一段」

【入参 / 出参】
    读：state["embeddings_content"]、state["item_name"]、state["file_title"]
    写：无（副作用是 Milvus 里的数据）

【M0 阶段怎么跑】
    .env 里 MILVUS_URI=./data/milvus.db，走 Milvus Lite——一个单文件，
    零运维，不需要起 Docker。上生产再换 standalone 的 http URI，
    代码不用改，只改 .env。

【建表要点】
    两个向量字段的度量方式不同，别配错：
      dense  -> COSINE（BGE-M3 稠密向量已归一化，COSINE 等价内积且更直观）
      sparse -> IP（内积，Milvus 的稀疏向量标准用法）
    建索引：dense 用 HNSW / AUTOINDEX，sparse 用 SPARSE_INVERTED_INDEX。

【当前状态】空壳（return state）。
"""

from app.core.logger import node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    将处理好的向量数据写入 Milvus 数据库。
    """
    add_running_task(state["task_id"], "node_import_milvus")

    # TODO: 1) 建两个 collection（不存在才建，索引：dense=HNSW/COSINE，sparse=IP）
    #      2) 按 item_name / file_title 删旧数据（幂等）
    #      3) 批量 insert（Milvus Lite 也建议分批，别一次塞几万条）

    add_done_task(state["task_id"], "node_import_milvus")
    return state
