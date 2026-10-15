"""
app/import_process/agent/nodes/node_pdf_to_md.py —— PDF 转 Markdown

【解决什么问题】
    语料是 88 本设备手册 PDF（约 707MB）。PDF 是非结构化数据：有分栏、表格、
    页眉页脚、图片，直接按文本抽出来会乱成一团。
    这一步把 PDF 转成 Markdown，保留标题层级和表格结构，后面才能按标题切分。

【入参 / 出参】
    读：state["pdf_path"]、state["file_title"]
    写：state["md_content"]（全文 Markdown）、state["md_path"]（落盘位置）

【实现路线（两步走，先易后难）】
    第一步（先跑通）：pymupdf / pdfplumber 直接抽文本，快、零依赖、
        对纯文本型手册够用。缺点是丢表格结构、多栏顺序会错。
    第二步（再替换）：MinerU（magic-pdf）。它是专门做 PDF→Markdown 的，
        能还原表格和阅读顺序，是本项目简历上的技术点。
        但它下载和 GPU 依赖重，所以先用第一步把整条链路打通，
        确认链路没问题了再换 MinerU——换的时候只改这一个文件，其他地方不用动。

【当前状态】空壳（return state），等 main_graph 跑通后实现。
"""

from app.core.logger import node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """
    add_running_task(state["task_id"], "node_pdf_to_md")

    # TODO(实现顺序): 1) pymupdf 抽文本 -> 2) 换 MinerU(magic-pdf) 保留表格结构
    # 产出写到 state["md_content"]，并落盘到 settings.parsed_path / f"{file_title}.md"

    add_done_task(state["task_id"], "node_pdf_to_md")
    return state
