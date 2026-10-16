"""
tests/test_node_pdf_to_md.py —— PDF 转 Markdown 的测试

【为什么分两类测试】
    纯函数测试（正则 / 字号统计 / 页眉页脚 / 标题分级）：不需要 pymupdf、不需要真实 PDF，
    毫秒级，任何环境都能跑 —— 这部分钉死的是**算法逻辑**。
    冒烟测试：需要 pymupdf + doc/ 里至少一本真实 PDF，验证**端到端真的能解析**。
    pymupdf 没装时冒烟测试自动 skip，不会让整个测试变红。
"""

import pytest

from app.core.config import settings
from app.import_process.agent.state import create_default_state
from app.import_process.agent.nodes.node_pdf_to_md import (
    _PAGE_NUM_RE,
    _body_font_size,
    _detect_boilerplate,
    _page_to_markdown,
    fitz,
    node_pdf_to_md,
    pdf_to_markdown,
)

_HAS_PYMUPDF = fitz is not None
_REAL_PDFS = sorted(settings.doc_path.glob("*.pdf")) if settings.doc_path.exists() else []
_NEEDS_REAL_PDF = pytest.mark.skipif(
    not _HAS_PYMUPDF or not _REAL_PDFS,
    reason="需要 pymupdf 且 doc/ 里至少有一本 PDF",
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


def test_page_to_markdown_drops_boilerplate_and_page_num():
    lines = [("页眉文字", 9.0), ("7", 9.0), ("真正的内容", 10.0)]
    out = _page_to_markdown(lines, body_size=10.0, boilerplate={"页眉文字"})
    assert out == ["真正的内容"]


def test_page_to_markdown_long_text_is_not_heading():
    """字号很大但很长 -> 是加粗正文，不是标题。"""
    long_text = "这" * 100
    out = _page_to_markdown([(long_text, 20.0)], body_size=10.0, boilerplate=set())
    assert out == [long_text]


# ---------------- 节点：降级路径 ----------------

def test_node_empty_pdf_path_degrades_silently():
    """pdf_path 为空 -> 不抛异常，md_content 保持空。"""
    state = create_default_state(task_id="t_empty", pdf_path="")
    result = node_pdf_to_md(state)
    assert result["md_content"] == ""
    assert result["md_path"] == ""


def test_node_missing_file_logs_error_and_degrades(tmp_path):
    """文件不存在 -> 打 error 日志并降级，不抛异常（批量导入需要这个行为）。"""
    missing = tmp_path / "not_exist.pdf"
    state = create_default_state(task_id="t_missing", pdf_path=str(missing))
    result = node_pdf_to_md(state)
    assert result["md_content"] == ""
    assert result["md_path"] == ""


def test_node_raises_when_pymupdf_absent(monkeypatch):
    """pymupdf 没装时，节点不该静默返回空内容 —— 必须抛，否则下游很难排查。"""
    import app.import_process.agent.nodes.node_pdf_to_md as mod

    monkeypatch.setattr(mod, "fitz", None)
    state = create_default_state(task_id="t_nofitz", pdf_path=str(_REAL_PDFS[0] if _REAL_PDFS else "x.pdf"))
    if _REAL_PDFS:
        with pytest.raises(RuntimeError, match="pymupdf"):
            node_pdf_to_md(state)


# ---------------- 端到端冒烟（需要真实 PDF） ----------------

@_NEEDS_REAL_PDF
def test_real_pdf_produces_non_empty_markdown():
    md = pdf_to_markdown(_REAL_PDFS[0])
    assert isinstance(md, str)
    assert len(md) > 100
    assert "<!-- page 1 -->" in md


@_NEEDS_REAL_PDF
def test_node_end_to_end_writes_file():
    """跑完整节点，验证落盘 + state 两个字段都被写上了。
    产物写到 data/parsed/（已被 .gitignore 忽略），不会进仓库。"""
    target = _REAL_PDFS[0]
    state = create_default_state(
        task_id="t_e2e", pdf_path=str(target), file_title=f"_test_{target.stem}"
    )
    result = node_pdf_to_md(state)
    assert result["md_content"]
    assert result["md_path"].endswith(".md")
    assert (settings.parsed_path / f"_test_{target.stem}.md").exists()
