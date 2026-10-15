"""
app/import_process/agent/nodes/node_entry.py —— 导入流程的入口节点

【解决什么问题】
    外部只给一个 local_file_path。这个节点负责：
      1. 判断这是个什么文件（PDF / Markdown / 都不是）；
      2. 把判断结果写进 state 的两个开关，供 main_graph 的条件边分流；
      3. 抽出文件名作为 file_title（后面写库时用它做文档标识）。

    它**不读文件内容**——「文件到底存不存在、能不能解析」是下游节点的事。
    这样职责单一，测试也简单：不用准备真实文件就能测这个节点。

【入参 / 出参】
    读：state["local_file_path"]、state["task_id"]
    写：state["is_pdf_read_enabled"]、state["is_md_read_enabled"]
        state["pdf_path"] 或 state["md_path"]、state["file_title"]

【降级策略（重要设计）】
    路径为空、或不是 md/pdf 时，**不抛异常**，而是两个开关都保持 False，
    让 main_graph 的条件边把它送到 END。
    为什么：批量导入 88 本手册时，混进一个 .txt 不该让整个批次崩掉。
    代价：静默跳过，所以这里打了 warning 日志，事后能从日志里捞出来。

【和教程版的差异】
    1. 后缀判断加了 .lower()：Windows 下文件名常是 .PDF / .Pdf，
       教程的 endswith(".pdf") 会漏判，然后静默走 END，很难排查；
    2. 多打一条文件是否存在的 warning：教程注释说「不考虑文件是否真存在」，
       但 Ava 调试时写错路径会很困惑，加一条日志成本几乎为零。
"""

from pathlib import Path

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 受支持的扩展名（小写）
SUPPORTED_MD = ".md"
SUPPORTED_PDF = ".pdf"


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """
    add_running_task(state["task_id"], "node_entry")

    local_file_path = state["local_file_path"]

    # --- 情况 1：路径为空 -> 降级，交由条件边走到 END ---
    if not local_file_path:
        logger.warning("node_entry: local_file_path 为空，直接跳转到 END 节点")
        add_done_task(state["task_id"], "node_entry")
        return state

    # --- 情况 2：按后缀分流（大小写不敏感）---
    suffix = Path(local_file_path).suffix.lower()
    if suffix == SUPPORTED_MD:
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True
    elif suffix == SUPPORTED_PDF:
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True
    else:
        # --- 情况 3：不支持的类型 -> 降级 ---
        logger.warning(
            f"node_entry: 文件 {local_file_path} 不是 md/pdf（后缀={suffix or '无'}），"
            f"无法识别，直接跳转到 END 节点"
        )
        add_done_task(state["task_id"], "node_entry")
        return state

    # --- 抽出文件名去后缀，作为文档标题 ---
    # Path.stem 对 "huawei.xxx.pdf" 会取 "huawei.xxx"，符合预期
    state["file_title"] = Path(local_file_path).stem

    # 文件是否真的存在：这里只提示不阻断（真正读取在下游节点）
    if not Path(local_file_path).exists():
        logger.warning(
            f"node_entry: 文件类型已识别，但路径当前不存在：{local_file_path}。"
            f"（仅提示，不阻断流程；若后续节点报找不到文件，先检查这个路径）"
        )

    add_done_task(state["task_id"], "node_entry")
    return state


if __name__ == "__main__":
    # 不依赖 pytest 的快速自检：覆盖不支持类型 / MD / PDF / 空路径四种场景
    from app.import_process.agent.state import create_default_state

    cases = [
        ("空路径", ""),
        ("不支持的类型", "联想海豚用户手册.txt"),
        ("Markdown", "小米用户手册.md"),
        ("PDF", "万用表RS-12的使用.pdf"),
        ("大写后缀PDF", "HAK180.PDF"),
    ]
    for name, path in cases:
        st = create_default_state(task_id=f"self_{name}", local_file_path=path)
        node_entry(st)
        print(
            f"{name:12s} -> md={st['is_md_read_enabled']!s:5s} "
            f"pdf={st['is_pdf_read_enabled']!s:5s} title={st['file_title']!r}"
        )
