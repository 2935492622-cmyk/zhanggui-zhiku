"""
app/import_process/agent/nodes/node_document_split.py —— 文档切分

【解决什么问题】
    一本手册几十上百页，整篇塞进向量库会导致：向量把整本书的语义平均掉，
    检索「HAK180 怎么换色带」时命中不了具体那一段。
    切分就是把长文档切成语义完整的片段（chunk），让向量能精确对应到段落。

【入参 / 出参】
    读：state["md_content"]、state["file_title"]
    写：state["chunks"]（list[dict]，每个元素含 content + metadata）

【为什么这是本项目最值得你自己写的节点】
    切分策略直接决定检索质量上限，也是面试官最爱追问的地方：
      - 边界怎么定？按 Markdown 标题层级（H1/H2/H3）切，还是按固定字数滑窗？
      - 片段多大？太大语义模糊，太小丢失上下文；
      - 要不要把标题路径（"第3章 > 3.2 更换色带"）前置进 embedding 文本？
      - overlap 留多少？
    这些没有标准答案，必须靠你自己做实验、看检索结果调。
    参数先放 settings.chunk_max_size / chunk_min_size / chunk_overlap_ratio，
    方便你改 .env 做对比实验，不用改代码。

【一个针对本语料的具体问题】
    设备手册里有大量「安全须知 / 合规声明」样板段落，每本都有且内容雷同。
    它们会污染检索（不同型号的手册因为这些段落而互相相似）。
    要不要过滤掉、怎么识别，是你做实验时要回答的问题。

【当前状态】空壳（return state）。
"""

from app.core.logger import node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    将长文档切分成小的 Chunks (切片) 以便检索。
    """
    add_running_task(state["task_id"], "node_document_split")

    # TODO: 1) 按 Markdown 标题层级递归切分 -> 2) 超长段落二次切分
    #      -> 3) 过短片段贪心合并 -> 4) 生成 metadata（标题路径、页码、file_title）
    # 参数从 settings 读：chunk_max_size / chunk_min_size / chunk_overlap_ratio

    add_done_task(state["task_id"], "node_document_split")
    return state
