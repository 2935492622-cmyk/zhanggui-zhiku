"""
app/import_process/agent/state.py —— 导入流程的全局状态（LangGraph 的 State）

【解决什么问题】
    7 个节点串成一条流水线，每个节点「读一部分、写一部分」。
    这个 TypedDict 就是它们在之间传递的那张表：一次导入任务的全部数据都在这。

    用 TypedDict 而不是普通 dict 的原因：写 state["xxx"] 时 IDE 能自动补全、
    mypy 能检查出拼错的字段名。运行时它仍然就是一个 dict。

【上下游】
    上游：main_graph.py 用 create_default_state() 创建初始状态
    下游：7 个节点逐个读写

【字段是谁写的 / 谁读的】
    task_id                 node 外部传入（或自动生成）      所有节点用来打日志
    local_file_path         外部传入                        node_entry 读
    is_pdf_read_enabled      node_entry 写                   main_graph 的条件边读（决定走哪条路）
    is_md_read_enabled       node_entry 写                   main_graph 的条件边读
    pdf_path / md_path       node_entry / node_pdf_to_md 写   node_pdf_to_md / node_md_img 读
    file_title               node_entry 写                   后续所有节点写库时的文档标识
    md_content               node_pdf_to_md 或 node_md_img 写 node_document_split 读
    chunks                   node_document_split 写           node_bge_embedding 读
    item_name                node_item_name_recognition 写    node_import_milvus 读（做幂等清理）
    embeddings_content       node_bge_embedding 写            node_import_milvus 读（入库）

【和教程版的差异】
    教程里有个 local_dir 字段（"当前工作目录或输出目录"）。
    你的项目里路径统一由 settings 的派生属性给（settings.doc_path / parsed_path），
    不存在「每个任务一个工作目录」的概念，所以去掉了，避免两份真相。
"""

import copy
from typing import List, TypedDict


class ImportGraphState(TypedDict):
    """图的状态定义：包含所有节点产生和消费的数据字段。"""

    task_id: str  # 任务唯一ID，用于追踪日志和进度查询

    # --- 流程控制标记（node_entry 写，main_graph 的条件边读）---
    is_md_read_enabled: bool  # True -> 走 node_md_img
    is_pdf_read_enabled: bool  # True -> 走 node_pdf_to_md
    # 两个都是 False -> 直接 END（不支持的文件类型，优雅降级）

    # --- 路径相关 ---
    local_file_path: str  # 用户传入的原始文件路径
    pdf_path: str  # 输入是 PDF 时的路径
    md_path: str  # Markdown 路径（转换产物，或直接输入的 .md）
    file_title: str  # 文件名去后缀，如 "hak180使用说明书"

    # --- 内容数据 ---
    md_content: str  # Markdown 全文
    chunks: List[dict]  # 切片列表，每个元素 = {"content": ..., "metadata": ...}
    item_name: str  # 识别出的商品主体名，如 "HAK180 烫金机"

    # --- 向量数据 ---
    item_embedding: dict  # 品名自身的 dense/sparse 向量
    import_result: dict  # 真实写入统计，仅完整成功时填充
    embeddings_content: List[dict]  # 带 dense/sparse 向量的切片，准备写 Milvus


# 默认初始值。注意：可变类型（list）在这里是共享引用，
# 所以下面 create_default_state 用 deepcopy 拷贝，避免多个任务互相污染。
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": [],
    "item_embedding": {},
    "import_result": {},
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持用关键字参数覆盖。

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc/hak180使用说明书.pdf")
    """
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state
'''
这个函数的意思是：
1. 从默认模板复制一份全新的状态。
2. 把调用者传进来的参数覆盖到这份状态中。
3. 返回这份新状态。
'''

def get_default_state() -> ImportGraphState:
    """返回一个全新的默认状态（深拷贝，避免全局变量污染）"""
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    s1 = get_default_state()
    s2 = create_default_state(task_id="0001")
    print("默认状态:", s1)
    print("覆盖后:", s2["task_id"])
    # 验证 deepcopy 生效：改 s1 的 list 不影响 s2
    s1["chunks"].append("x")
    print("s2 未被污染:", s2["chunks"] == [])
