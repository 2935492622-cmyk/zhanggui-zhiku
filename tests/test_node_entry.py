"""
tests/test_node_entry.py —— node_entry 节点的单元测试

【为什么先测这个节点】
    它是整条链路的入口，也是唯一一个真实实现的节点。
    它判错类型，后面所有节点都会跑错分支，所以必须先钉死。

【测试不依赖外部环境】
    不读真实文件、不连 Milvus、不调 LLM —— 纯字符串判断，毫秒级。
    「文件是否真的存在」只打 warning 不阻断，所以这里可以用假路径。

【运行】
    uv run pytest tests/test_node_entry.py -v
"""

from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.state import create_default_state


def _run(local_file_path: str):
    state = create_default_state(task_id="test_entry", local_file_path=local_file_path)
    return node_entry(state)


def test_md_file_sets_md_flag():
    """Markdown 文件：只打开 is_md_read_enabled，并填好 md_path"""
    state = _run("小米用户手册.md")
    assert state["is_md_read_enabled"] is True
    assert state["is_pdf_read_enabled"] is False
    assert state["md_path"] == "小米用户手册.md"
    assert state["file_title"] == "小米用户手册"


def test_pdf_file_sets_pdf_flag():
    """PDF 文件：只打开 is_pdf_read_enabled，并填好 pdf_path"""
    state = _run("万用表RS-12的使用.pdf")
    assert state["is_pdf_read_enabled"] is True
    assert state["is_md_read_enabled"] is False
    assert state["pdf_path"] == "万用表RS-12的使用.pdf"
    assert state["file_title"] == "万用表RS-12的使用"


def test_unsupported_type_degrades_silently():
    """不支持的类型：两个开关都保持 False（由条件边送到 END），不抛异常"""
    state = _run("联想海豚用户手册.txt")
    assert state["is_pdf_read_enabled"] is False
    assert state["is_md_read_enabled"] is False
    assert state["file_title"] == ""


def test_empty_path_degrades_silently():
    """空路径：同样降级，不抛异常"""
    state = _run("")
    assert state["is_pdf_read_enabled"] is False
    assert state["is_md_read_enabled"] is False


def test_suffix_is_case_insensitive():
    """.PDF 大写后缀也要能识别（Windows 下常见，教程版会漏判）"""
    state = _run("HAK180.PDF")
    assert state["is_pdf_read_enabled"] is True


def test_multi_dot_filename_keeps_full_stem():
    """带多个点的文件名：stem 只去掉最后一个后缀"""
    state = _run("HUAWEI.MateBook.B5-430.pdf")
    assert state["file_title"] == "HUAWEI.MateBook.B5-430"


def test_real_corpus_file_is_recognized():
    """用 doc/ 里真实存在的语料跑一遍（确认语料确实拷进来了）"""
    from app.core.config import settings

    pdfs = sorted(settings.doc_path.glob("*.pdf"))
    assert pdfs, f"doc/ 下没有找到 PDF，检查语料是否拷贝到 {settings.doc_path}"

    state = _run(str(pdfs[0]))
    assert state["is_pdf_read_enabled"] is True
    assert state["file_title"] == pdfs[0].stem
