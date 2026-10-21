"""
app/import_process/agent/main_graph.py —— 知识库导入流程的主图

【解决什么问题】
    把「一本手册 → 向量库」这件事拆成 7 个节点，用 LangGraph 串起来。
    为什么不用普通函数顺序调用：因为流程里有分支（PDF 要先转换，Markdown 不用），
    而且后面要加「失败重试 / 人工审核 / 断点续跑」——状态图天然支持，顺序调用不支持。

【图的形状】
                        ┌─ is_md_read_enabled  ─→ node_md_img ─────────┐
    START → node_entry ─┤                                              ├→ node_document_split
                        └─ is_pdf_read_enabled ─→ node_pdf_to_md ──────┘        │
                                                                                 ↓
    END ← node_import_milvus ← node_bge_embedding ← node_item_name_recognition ←┘

    node_entry 后面是**条件边**：根据文件类型决定走哪条路；两个标记都是 False
    （比如传进来一个 .txt）就直接 END，不报错——这叫优雅降级。

【上下游】
    上游：将来 FastAPI 的上传接口，调 kb_import_app.invoke(state)
    下游：Milvus 双层 collection

【怎么用】
    from app.import_process.agent.main_graph import kb_import_app
    from app.import_process.agent.state import create_default_state
    result = kb_import_app.invoke(create_default_state(task_id="001", local_file_path="doc/xxx.pdf"))

【目前的状态】
    7 个节点里只有 node_entry 是真实实现，其余 6 个是空壳（return state）。
    这是故意的：先把「图能跑通、能测试」验证掉，再逐个填实现。
    这样每一步都有能运行的东西，不会出现写了 500 行却一次都跑不起来的情况。
"""

from langgraph.graph import END, START, StateGraph

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
# 导入 7 个业务节点
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus

# 1. 定义状态图对象，并指定全局 state 类型
workflow = StateGraph(ImportGraphState)

# 2. 注册节点（第一个参数是节点名，日志和进度追踪用的就是它）
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# 3. 指定入口节点
workflow.set_entry_point("node_entry")


# 4. 入口节点之后的条件边：按文件类型分流
def after_entry_node(state: ImportGraphState):
    """
    路由函数：读 node_entry 写进 state 的两个标记，决定下一步走哪。

    返回值必须是节点名（或 END）。这里返回的字符串和节点名完全一致，
    所以下面的 path_map 其实可以省略——但显式写出来有两个好处：
      1. 打印图结构时能显示分支名；
      2. 以后想把返回值改成语义化名字（如 "to_pdf"）时不用改调用方。
    """
    if state["is_md_read_enabled"]:
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    else:
        logger.warning("node_entry 未识别出受支持的文件类型，流程提前结束")
        return END


workflow.add_conditional_edges(
    "node_entry",
    after_entry_node,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END,
    },
)
'''
先根据状态决定路径，将PDF和markdown进行分流，然后根据运算结果选择。
然后剩下的节点是：识别图片、识别标题、向 Milvus 插入数据。只需要建立固定节点，不需要路由函数。
add_edge:建立固定联系   add_conditional_edges:建立可变联系,根据状态决定路径
'''

# 5. 静态边：这两条路最终汇合到 node_document_split
workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF 转完 MD 也要处理图片
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# 6. 编译成可执行对象
kb_import_app = workflow.compile()


if __name__ == "__main__":
    # 直接跑这个文件可以看图结构，不需要任何外部服务
    from app.import_process.agent.state import create_default_state

    kb_import_app.get_graph().print_ascii()

    demo = kb_import_app.invoke(
        create_default_state(task_id="demo", local_file_path="doc/hak180使用说明书.pdf")
    )
    print("\n执行结果:")
    for k, v in demo.items():
        print(f"  {k} = {v!r}")
