"""
tests/test_import_maingraph.py —— 导入主图的端到端测试

【测什么】
    不是测某个节点，而是测「图本身」：
      1. 图能编译（节点名、边、条件边都接对了）；
      2. PDF 走 PDF 分支，Markdown 走 MD 分支，不支持的类型走 END；
      3. 全流程跑完不抛异常，state 字段符合预期。

    这一步的价值：现在 6 个节点还是空壳，但**图结构已经被测试钉死了**。
    后面你逐个填实现时，跑这个测试就能立刻知道是自己写坏了还是结构没变。

【为什么用假路径而不读真实 PDF】
    现在 node_pdf_to_md 还是空壳，读真实文件也没意义。
    等它实现后，再加一个「真实 PDF 全流程」的集成测试。

【运行】
    uv run pytest tests/test_import_maingraph.py -v
"""

import pytest

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state


def test_graph_compiles():
    """图必须能编译出来，且 7 个节点都注册上了"""
    graph = kb_import_app.get_graph()
    node_names = set(graph.nodes.keys())
    expected = {
        "node_entry",
        "node_pdf_to_md",
        "node_md_img",
        "node_document_split",
        "node_item_name_recognition",
        "node_bge_embedding",
        "node_import_milvus",
    }
    # LangGraph 内部会加 __start__ / __end__ 之类的节点，用子集判断
    assert expected.issubset(node_names), f"缺少节点: {expected - node_names}"


def test_pdf_branch():
    """PDF 输入：走 node_entry -> node_pdf_to_md 分支"""
    state = create_default_state(task_id="g01", local_file_path="doc/hak180使用说明书.pdf")
    result = kb_import_app.invoke(state)
    assert result["is_pdf_read_enabled"] is True
    assert result["is_md_read_enabled"] is False
    assert result["file_title"] == "hak180使用说明书"


def test_md_branch():
    """Markdown 输入：跳过 PDF 转换，直接走 node_md_img"""
    state = create_default_state(task_id="g02", local_file_path="doc/小米用户手册.md")
    result = kb_import_app.invoke(state)
    assert result["is_md_read_enabled"] is True
    assert result["is_pdf_read_enabled"] is False


def test_unsupported_type_ends_early():
    """不支持的类型：条件边直接送 END，两个开关都是 False"""
    state = create_default_state(task_id="g03", local_file_path="doc/联想海豚用户手册.txt")
    result = kb_import_app.invoke(state)
    assert result["is_pdf_read_enabled"] is False
    assert result["is_md_read_enabled"] is False
    assert result["file_title"] == ""


def test_empty_path_ends_early():
    """空路径同样提前结束，不抛异常"""
    state = create_default_state(task_id="g04", local_file_path="")
    result = kb_import_app.invoke(state)
    assert result["is_pdf_read_enabled"] is False


def test_task_progress_is_tracked():
    """跑完图之后，任务进度里应该记录到 node_entry 已完成"""
    from app.utils.task_utils import get_done_task_list

    task_id = "g05"
    state = create_default_state(task_id=task_id, local_file_path="doc/万用表RS-12的使用.pdf")
    kb_import_app.invoke(state)
    done = get_done_task_list(task_id)
    assert "检查文件" in done, f"进度追踪没有记录 node_entry，实际: {done}"


@pytest.mark.parametrize(
    "filename,expect_pdf,expect_md",
    [
        ("a.pdf", True, False),
        ("a.PDF", True, False),
        ("a.md", False, True),
        ("a.txt", False, False),
        ("a", False, False),
    ],
)
def test_type_routing_table(filename, expect_pdf, expect_md):
    """用参数化把类型路由的所有分支一次测完"""
    state = create_default_state(task_id="g06", local_file_path=f"doc/{filename}")
    result = kb_import_app.invoke(state)
    assert result["is_pdf_read_enabled"] is expect_pdf
    assert result["is_md_read_enabled"] is expect_md


def test_real_corpus_runs_end_to_end():
    """拿 doc/ 里一本真实手册跑完整图，确认不会中途炸掉"""
    from app.core.config import settings

    pdfs = sorted(settings.doc_path.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"doc/ 下没有 PDF，跳过（语料目录: {settings.doc_path}）")

    state = create_default_state(task_id="g07", local_file_path=str(pdfs[0]))
    result = kb_import_app.invoke(state)
    assert result["is_pdf_read_enabled"] is True
    assert result["file_title"] == pdfs[0].stem
