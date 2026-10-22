"""
tests/test_import_maingraph.py —— 导入主图的端到端测试

【测什么】
    不是测某个节点，而是测「图本身」：
      1. 图能编译（节点名、边、条件边都接对了）；
      2. PDF 走 PDF 分支，Markdown 走 MD 分支，不支持的类型走 END；
      3. 全流程跑完不抛异常，state 字段符合预期。

    入口和 PDF 节点已实现，其余节点仍为占位；本文件验证当前图编排。
    后面你逐个填实现时，跑这个测试就能立刻知道是自己写坏了还是结构没变。

【测试边界】
    PDF 路由使用真实生成的 PDF 并实际解析，产物写入临时目录。
    模型和数据库使用明确的 Mock，测试通过不代表真实模型或入库成功。

【运行】
    uv run pytest tests/test_import_maingraph.py -v
"""

from types import SimpleNamespace

import pytest

import app.import_process.agent.nodes.node_pdf_to_md as pdf_node

pytestmark = [pytest.mark.integration, pytest.mark.mock]


@pytest.fixture(autouse=True)
def isolated_pdf_output(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_node, "settings", SimpleNamespace(parsed_path=tmp_path / "parsed"))


@pytest.fixture(autouse=True)
def mock_external_nodes(monkeypatch):
    import app.import_process.agent.nodes.node_item_name_recognition as names
    import app.import_process.agent.nodes.node_bge_embedding as embeddings
    import app.import_process.agent.nodes.node_import_milvus as storage
    monkeypatch.setattr(names, 'recognize', lambda text, title: title)
    class Model:
        tokenizer = staticmethod(lambda text, **kw: {'input_ids': [1]})
        def encode(self, texts, **kw):
            return {'dense_vecs': [[1.0] * 1024 for _ in texts],
                    'lexical_weights': [{1: 1.0} for _ in texts]}
    monkeypatch.setattr(embeddings, 'get_model', lambda *args: Model())
    monkeypatch.setattr(storage, 'check_schema', lambda *args: None)
    class Client:
        def upsert(self, collection_name, data):
            return {'upsert_count': len(data)}
        def flush(self, **kw): pass
        def close(self): pass
    monkeypatch.setattr(storage, 'get_client', Client)


from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state


def _input_file(tmp_path, filename):
    path = tmp_path / filename
    if path.suffix.lower() == ".pdf":
        engine = pytest.importorskip("pymupdf")
        with engine.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 100), "Routing test manual", fontsize=12)
            doc.save(path)
    else:
        path.write_text("# Test\nMarkdown body", encoding="utf-8")
    return str(path)


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


def test_pdf_branch(tmp_path):
    """PDF 输入：走 node_entry -> node_pdf_to_md 分支"""
    state = create_default_state(task_id="g01", local_file_path=_input_file(tmp_path, "hak180使用说明书.pdf"))
    result = kb_import_app.invoke(state)
    assert result["is_pdf_read_enabled"] is True
    assert result["is_md_read_enabled"] is False
    assert result["file_title"] == "hak180使用说明书"


def test_md_branch(tmp_path):
    """Markdown 输入：跳过 PDF 转换，直接走 node_md_img"""
    state = create_default_state(task_id="g02", local_file_path=_input_file(tmp_path, "小米用户手册.md"))
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


def test_task_progress_is_tracked(tmp_path):
    """跑完图之后，任务进度里应该记录到 node_entry 已完成"""
    from app.utils.task_utils import get_done_task_list

    task_id = "g05"
    state = create_default_state(task_id=task_id, local_file_path=_input_file(tmp_path, "万用表RS-12的使用.pdf"))
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
def test_type_routing_table(tmp_path, filename, expect_pdf, expect_md):
    """用参数化把类型路由的所有分支一次测完"""
    state = create_default_state(task_id="g06", local_file_path=_input_file(tmp_path, filename))
    result = kb_import_app.invoke(state)
    assert result["is_pdf_read_enabled"] is expect_pdf
    assert result["is_md_read_enabled"] is expect_md


def test_real_corpus_parses_through_current_graph():
    """真实语料验证入口与 PDF 解析；后续仍是占位节点"""
    from app.core.config import settings

    pdfs = sorted(settings.doc_path.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"doc/ 下没有 PDF，跳过（语料目录: {settings.doc_path}）")

    state = create_default_state(task_id="g07", local_file_path=str(pdfs[0]))
    result = kb_import_app.invoke(state)
    assert result["is_pdf_read_enabled"] is True
    assert result["file_title"] == pdfs[0].stem
    assert result["md_content"]
    assert result["md_path"]


def test_missing_file_stops_before_downstream(tmp_path):
    from app.utils.task_utils import get_done_task_list, get_task_status, get_running_task_list

    task_id = "g_missing"
    state = create_default_state(task_id=task_id, local_file_path=str(tmp_path / "missing.pdf"))
    with pytest.raises(FileNotFoundError):
        kb_import_app.invoke(state)
    assert get_done_task_list(task_id) == []
    assert get_running_task_list(task_id) == []
    assert get_task_status(task_id) == "failed"


@pytest.mark.parametrize("filename,expected", [
    ("route.pdf", ["node_entry", "node_pdf_to_md", "node_md_img", "node_document_split",
                   "node_item_name_recognition", "node_bge_embedding", "node_import_milvus"]),
    ("route.md", ["node_entry", "node_md_img", "node_document_split",
                  "node_item_name_recognition", "node_bge_embedding", "node_import_milvus"]),
    ("route.txt", ["node_entry"]),
])
def test_actual_node_order(tmp_path, filename, expected):
    state = create_default_state(task_id="g_order", local_file_path=_input_file(tmp_path, filename))
    updates = list(kb_import_app.stream(state, stream_mode="updates"))
    assert [name for update in updates for name in update] == expected


def test_corrupt_pdf_stops_before_image_node(tmp_path):
    from app.utils.task_utils import get_done_task_list, get_task_status, get_running_task_list

    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-corrupt")
    state = create_default_state(task_id="g_corrupt", local_file_path=str(path))
    engine = pytest.importorskip("pymupdf")
    with pytest.raises(engine.FileDataError):
        kb_import_app.invoke(state)
    assert get_done_task_list("g_corrupt") == ["检查文件"]
    assert get_running_task_list("g_corrupt") == []
    assert get_task_status("g_corrupt") == "failed"
