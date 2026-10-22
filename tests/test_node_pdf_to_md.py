"""PDF 节点测试：纯算法、标注 mock 的故障测试、标注 integration 的真实解析。

所有解析产物写入 tmp_path，不修改 doc/ 语料或正式 data/parsed。
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import app.import_process.agent.nodes.node_pdf_to_md as mod
from app.utils.task_utils import get_task_status, get_task_result, get_done_task_list, get_running_task_list

from app.import_process.agent.state import create_default_state
from app.import_process.agent.nodes.node_pdf_to_md import (
    _PAGE_NUM_RE,
    _body_font_size,
    _detect_boilerplate,
    _page_to_markdown,
    node_pdf_to_md,
    pdf_to_markdown,
)

# ---------------- 纯函数：页码行识别 ----------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("12", True),          # 纯数字页码
        ("  7  ", True),       # 带空格
        ("- 3 -", True),       # 常见装饰写法
        ("第 3 页", True),
        ("第 3 页 / 共 20 页", True),
        ("HAK180", False),     # 型号，绝不能当页码丢掉
        ("12.5V", False),      # 正文里的数字
        ("更换色带步骤 3", False),
        ("", False),
    ],
)
def test_page_num_regex(text, expected):
    """核心风险：型号/参数里的数字被误判成页码丢掉。这条测试专门守这个边界。"""
    assert bool(_PAGE_NUM_RE.match(text)) is expected


# ---------------- 纯函数：正文字号统计 ----------------

def test_body_font_size_is_weighted_by_chars():
    """
    一页里标题 5 个字（字号 20），正文 500 字（字号 10）。
    按字符数加权 -> 正文赢，返回 10.0。
    如果按行数统计，两者各 1 行，结果不确定 —— 这条测试锁住加权行为。
    """
    pages = [[("第一章", 20.0), ("正" * 500, 10.0)]]
    assert _body_font_size(pages) == 10.0


def test_body_font_size_empty():
    assert _body_font_size([]) == 10.0


# ---------------- 纯函数：页眉页脚识别 ----------------

def test_detect_boilerplate():
    """5 页里 4 页首尾都有同一句话 -> 判定为页眉页脚（阈值 50%）。"""
    pages = [
        [("HAK180 使用说明书", 9.0), ("正文内容A", 10.0), ("www.brother.com", 9.0)],
        [("HAK180 使用说明书", 9.0), ("正文内容B", 10.0), ("www.brother.com", 9.0)],
        [("HAK180 使用说明书", 9.0), ("正文内容C", 10.0), ("www.brother.com", 9.0)],
        [("HAK180 使用说明书", 9.0), ("正文内容D", 10.0), ("www.brother.com", 9.0)],
        [("完全不同的首页", 9.0), ("正文内容E", 10.0), ("别的页脚", 9.0)],
    ]
    boiler = _detect_boilerplate(pages)
    assert "HAK180 使用说明书" in boiler
    assert "www.brother.com" in boiler
    assert "正文内容A" not in boiler


def test_detect_boilerplate_not_triggered_by_normal_text():
    """正文各行都不同 -> 不该误杀任何内容。"""
    pages = [[(f"第{i}段独有的内容", 10.0)] for i in range(5)]
    assert _detect_boilerplate(pages) == set()


# ---------------- 纯函数：标题分级 ----------------

def test_page_to_markdown_heading_levels():
    """字号 1.3 倍以上 -> #；1.15 倍以上 -> ##；其余原样。"""
    lines = [("大标题", 16.0), ("小标题", 12.0), ("这是正文", 10.0)]
    out = _page_to_markdown(lines, body_size=10.0, boilerplate=set())
    assert out == ["# 大标题", "## 小标题", "这是正文"]


def test_page_to_markdown_preserves_body_numbers():
    lines = [("页眉文字", 9.0), ("7", 9.0), ("真正的内容", 10.0)]
    out = _page_to_markdown(lines, body_size=10.0, boilerplate={"页眉文字"})
    assert out == ["7", "真正的内容"]


def test_page_to_markdown_long_text_is_not_heading():
    """字号很大但很长 -> 是加粗正文，不是标题。"""
    long_text = "这" * 100
    out = _page_to_markdown([(long_text, 20.0)], body_size=10.0, boilerplate=set())
    assert out == [long_text]


# ---------------- 节点单元测试：Mock 解析，验证状态及 IO 失败 ----------------

@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "settings", SimpleNamespace(parsed_path=tmp_path / "parsed"))


def _state(path):
    return create_default_state(task_id="pdf_test", pdf_path=str(path), file_title="manual")


def _assert_failed(state):
    assert state["md_content"] == state["md_path"] == ""
    assert get_task_status(state["task_id"]) == "failed"
    assert get_task_result(state["task_id"], "error")
    assert get_running_task_list(state["task_id"]) == []
    assert "PDF转Markdown" not in get_done_task_list(state["task_id"])


def test_node_empty_pdf_path_fails():
    state = _state("")
    with pytest.raises(ValueError, match="pdf_path"):
        node_pdf_to_md(state)
    _assert_failed(state)


def test_node_missing_file_fails(tmp_path):
    state = _state(tmp_path / "missing.pdf")
    with pytest.raises(FileNotFoundError):
        node_pdf_to_md(state)
    _assert_failed(state)


@pytest.mark.mock
def test_node_raises_when_pymupdf_absent(tmp_path, monkeypatch):
    path = tmp_path / "input.pdf"
    path.write_bytes(b"input")
    monkeypatch.setattr(mod, "fitz", None)
    state = _state(path)
    with pytest.raises(RuntimeError, match="pymupdf"):
        node_pdf_to_md(state)
    _assert_failed(state)


@pytest.mark.mock
@pytest.mark.parametrize("result", ["", "   ", None])
def test_node_rejects_empty_parser_output(tmp_path, monkeypatch, result):
    path = tmp_path / "input.pdf"
    path.write_bytes(b"input")
    monkeypatch.setattr(mod, "pdf_to_markdown", lambda path: result)
    state = _state(path)
    state.update(md_content="old", md_path="old.md")
    with pytest.raises(ValueError, match="为空"):
        node_pdf_to_md(state)
    _assert_failed(state)
    assert not mod.settings.parsed_path.exists()


@pytest.mark.mock
def test_mock_parser_failure(tmp_path, monkeypatch):
    path = tmp_path / "input.pdf"
    path.write_bytes(b"input")

    def fail(path):
        raise RuntimeError("parser failure")

    monkeypatch.setattr(mod, "pdf_to_markdown", fail)
    state = _state(path)
    with pytest.raises(RuntimeError, match="parser failure"):
        node_pdf_to_md(state)
    _assert_failed(state)


@pytest.mark.mock
@pytest.mark.parametrize("stage", ["mkdir", "write"])
def test_mock_output_failure(tmp_path, monkeypatch, stage):
    path = tmp_path / "input.pdf"
    path.write_bytes(b"input")
    monkeypatch.setattr(mod, "pdf_to_markdown", lambda path: "# Title\nBody")

    def fail(*args, **kwargs):
        raise PermissionError("output denied")

    monkeypatch.setattr(Path, "mkdir" if stage == "mkdir" else "write_text", fail)
    state = _state(path)
    with pytest.raises(PermissionError):
        node_pdf_to_md(state)
    _assert_failed(state)
    if stage == "write":
        assert list(mod.settings.parsed_path.iterdir()) == []


def test_short_documents_do_not_lose_boilerplate_candidates():
    for n in (1, 2):
        assert _detect_boilerplate([[("Important instruction", 10.0)]] * n) == set()


# ---------------- 真正调用 PyMuPDF 的本地集成测试 ----------------

@pytest.fixture
def sample_pdf(tmp_path):
    engine = pytest.importorskip("pymupdf")
    path = tmp_path / "manual.pdf"
    with engine.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 100), "Machine Manual", fontsize=18)
        page.insert_text((72, 150), "Instructions and operating details for the machine. " * 2, fontsize=10)
        page.insert_text((72, 180), "180", fontsize=10)
        page.insert_text((72, 210), "Disconnect power before maintenance. " * 2, fontsize=10)
        page.insert_text((72, 820), "1", fontsize=9)
        doc.save(path)
    return path


@pytest.mark.integration
def test_real_pdf_preserves_single_page_body(sample_pdf):
    md = pdf_to_markdown(sample_pdf)
    assert "<!-- page 1 -->" in md
    assert "# Machine Manual" in md
    assert "Instructions" in md
    assert "\n180\n" in md  # 正文数字不能当页码删除
    assert not md.rstrip().endswith("\n1")


@pytest.mark.integration
def test_node_real_conversion_and_unique_outputs(sample_pdf):
    first = node_pdf_to_md(_state(sample_pdf))
    original = Path(first["md_path"]).read_text(encoding="utf-8")
    second_state = _state(sample_pdf)
    second_state["file_title"] = "../../outside"
    second = node_pdf_to_md(second_state)
    assert first["md_path"] != second["md_path"]
    assert Path(first["md_path"]).read_text(encoding="utf-8") == original
    assert second["md_content"] == original
    assert Path(second["md_path"]).is_relative_to(mod.settings.parsed_path)
    assert get_running_task_list("pdf_test") == []
    assert get_done_task_list("pdf_test") == ["PDF转Markdown"]
    assert get_task_status("pdf_test") == "processing"


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["corrupt", "blank", "encrypted", "not_pdf"])
def test_real_invalid_pdf_fails(tmp_path, kind):
    engine = pytest.importorskip("pymupdf")
    path = tmp_path / "invalid.pdf"
    if kind == "corrupt":
        path.write_bytes(b"%PDF-1.7 garbage")
    elif kind == "not_pdf":
        # PyMuPDF 可以读取其他格式，这里必须明确拒绝。
        with engine.open() as doc:
            doc.new_page()
            doc[0].get_pixmap().save(tmp_path / "image.png")
        path.write_bytes((tmp_path / "image.png").read_bytes())
    else:
        with engine.open() as doc:
            doc.new_page()
            options = dict(encryption=engine.PDF_ENCRYPT_AES_256,
                           owner_pw="owner", user_pw="reader") if kind == "encrypted" else {}
            doc.save(path, **options)
    state = _state(path)
    expected = engine.FileDataError if kind == "corrupt" else ValueError
    with pytest.raises(expected):
        node_pdf_to_md(state)
    _assert_failed(state)
    assert not mod.settings.parsed_path.exists()


@pytest.mark.integration
def test_position_filter_keeps_same_text_in_body(tmp_path):
    engine = pytest.importorskip("pymupdf")
    path = tmp_path / "repeat.pdf"
    with engine.open() as doc:
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 30), "Repeated label", fontsize=10)
            page.insert_text((72, 150), "Repeated label", fontsize=10)
            page.insert_text((72, 180), f"Unique body section {i}", fontsize=10)
        doc.save(path)
    md = pdf_to_markdown(path)
    assert md.count("Repeated label") == 3
    assert all(f"<!-- page {i} -->" in md for i in (1, 2, 3))


@pytest.mark.integration
def test_corpus_pdf_conversion(first_real_pdf):
    state = node_pdf_to_md(_state(first_real_pdf))
    assert len(state["md_content"]) > 100
    assert "<!-- page " in state["md_content"]
    assert Path(state["md_path"]).read_text(encoding="utf-8") == state["md_content"]


@pytest.mark.integration
def test_empty_page_keeps_original_page_numbers(tmp_path):
    engine = pytest.importorskip("pymupdf")
    path = tmp_path / "mixed.pdf"
    with engine.open() as doc:
        doc.new_page()  # 没有文本层；当前不做 OCR。
        page = doc.new_page()
        page.insert_text((72, 120), "Text on original page two", fontsize=12)
        doc.save(path)
    md = pdf_to_markdown(path)
    assert "<!-- page 2 -->" in md
    assert "<!-- page 1 -->" not in md
    assert "Text on original page two" in md


@pytest.mark.integration
def test_retry_clears_stale_outputs(sample_pdf, monkeypatch):
    state = node_pdf_to_md(_state(sample_pdf))
    previous_output = Path(state["md_path"])
    engine = mod.fitz
    monkeypatch.setattr(mod, "fitz", None)
    with pytest.raises(RuntimeError, match="pymupdf"):
        node_pdf_to_md(state)
    _assert_failed(state)
    assert previous_output.is_file()  # 失败重试不删除上次已成功产物。
    monkeypatch.setattr(mod, "fitz", engine)
    node_pdf_to_md(state)
    assert get_task_result("pdf_test", "error") == ""
    assert get_task_status("pdf_test") == "processing"
