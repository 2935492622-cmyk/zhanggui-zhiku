"""PDF → Markdown：复用 v1 的字号推断，保留页码和文本块边界。

只支持可提取文字的 PDF；不做 OCR、表格重建、图片提取或多模态识别。
读取 pdf_path / task_id；成功落盘后才更新 md_content / md_path。
失败记录 failed 并抛异常，禁止空正文继续进入后续节点。
"""

import re
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

from app.core.config import settings
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import (
    TASK_STATUS_PROCESSING, add_done_task, add_failed_task, add_running_task,
    set_task_result, update_task_status,
)

try:
    import pymupdf as fitz
except ImportError:
    fitz = None

_PAGE_NUM_RE = re.compile(
    r"^(?:[-\s\u2013\u2014]*\d{1,4}[-\s\u2013\u2014]*"
    r"|第\s*\d{1,4}\s*页(?:\s*/\s*共\s*\d{1,4}\s*页)?)$"
)
_H1_RATIO = 1.30
_H2_RATIO = 1.15
_MAX_HEADING_LEN = 60
_BOILERPLATE_RATIO = 0.6
_BOILERPLATE_MAX_LEN = 60


def _extract_page_data(page):
    """返回文本行及页边缘行的索引；空行表示原 PDF 中的文本块边界。"""
    lines: list[tuple[str, float]] = []
    margins: set[int] = set()
    rect = page.rect
    # 仅上下各 8% 的区域参与页眉/页脚判定，正文中的重复句不删除。
    top, bottom = rect.y0 + rect.height * 0.08, rect.y1 - rect.height * 0.08
    for block in page.get_text("dict", sort=True, flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
        if block.get("type") != 0:
            continue
        if lines:
            lines.append(("", 0.0))
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            _, y0, _, y1 = line["bbox"]
            if y1 <= top or y0 >= bottom:
                margins.add(len(lines))
            lines.append((text, max(s["size"] for s in spans)))
    return lines, margins


def _extract_page_lines(page) -> list[tuple[str, float]]:
    return _extract_page_data(page)[0]


def _body_font_size(pages: list[list[tuple[str, float]]]) -> float:
    """
    正文字号 = 出现最多的字号，按**字符数**加权。

    为什么加权：按「行数」统计，标题字号和正文字号票数会很接近；
    按字符数统计，占绝对多数的正文稳赢。
    """
    counter: Counter = Counter()
    for page in pages:
        for text, size in page:
            counter[round(size, 1)] += len(text)
    if not counter:
        return 10.0
    return counter.most_common(1)[0][0]


def _detect_boilerplate(pages: list[list[tuple[str, float]]]) -> set[str]:
    """输入各页的页边缘行；至少重复 3 页且覆盖 60% 页面才认为是页眉页脚。"""
    if len(pages) < 3:
        return set()
    counter = Counter()
    for page in pages:
        counter.update({text for text, _ in page if 0 < len(text) <= _BOILERPLATE_MAX_LEN})
    return {text for text, count in counter.items()
            if count >= 3 and count / len(pages) >= _BOILERPLATE_RATIO}


def _page_to_markdown(page_lines, body_size: float, boilerplate: set[str]) -> list[str]:
    """按字号推断标题；正文中的纯数字保留，页码由调用方按位置过滤。"""
    out = []
    for text, size in page_lines:
        if text in boilerplate:
            continue
        if text and len(text) <= _MAX_HEADING_LEN:
            if size >= body_size * _H1_RATIO:
                out.append(f"# {text}")
                continue
            if size >= body_size * _H2_RATIO:
                out.append(f"## {text}")
                continue
        out.append(text)
    return out


def pdf_to_markdown(pdf_path: str | Path) -> str:
    """真正解析 PDF，不写文件；非 PDF、加密或整篇无文本时明确报错。"""
    if fitz is None:
        raise RuntimeError("pymupdf 未安装，请先同步项目依赖")
    with fitz.open(pdf_path) as doc:
        if not doc.is_pdf:
            raise ValueError("输入内容不是 PDF")
        if doc.needs_pass:
            raise ValueError("暂不支持需要密码的 PDF")
        if not doc.page_count:
            raise ValueError("PDF 没有页面")
        pages = [_extract_page_data(page) for page in doc]

    margin_pages = [[lines[i] for i in sorted(margins)] for lines, margins in pages]
    boilerplate = _detect_boilerplate(margin_pages)
    # 排除空白分隔行，按正文字符数统计常见字号。
    body_size = _body_font_size([lines for lines, _ in pages])
    parts = []
    empty_pages = 0
    for number, (lines, margins) in enumerate(pages, start=1):
        filtered = [(text, size) for i, (text, size) in enumerate(lines)
                    if not (i in margins and (text in boilerplate or _PAGE_NUM_RE.fullmatch(text)))]
        md = "\n".join(_page_to_markdown(filtered, body_size, set())).strip()
        if not md:
            empty_pages += 1
            continue
        parts.append(f"<!-- page {number} -->\n{md}")
    if not parts:
        raise ValueError("PDF 未提取到有效文本，可能是扫描件，需要 OCR 或人工检查")
    if empty_pages:
        logger.warning(f"node_pdf_to_md: {empty_pages} 页未提取到正文；当前不执行 OCR")
    return "\n\n".join(parts)


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """解析并持久化 Markdown；只有文件写入成功后才更新输出字段。"""
    task_id = state["task_id"]
    state.update(md_content="", md_path="")
    add_running_task(task_id, "node_pdf_to_md")
    update_task_status(task_id, TASK_STATUS_PROCESSING)
    set_task_result(task_id, "error", "")
    try:
        pdf_path = state["pdf_path"]
        if not isinstance(pdf_path, str) or not pdf_path.strip():
            raise ValueError("pdf_path 必须是非空字符串")
        path = Path(pdf_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("PDF 路径不存在或不是普通文件")
        md_content = pdf_to_markdown(path)
        if not isinstance(md_content, str) or not md_content.strip():
            raise ValueError("PDF 解析结果为空")

        out_dir = settings.parsed_path
        out_dir.mkdir(parents=True, exist_ok=True)
        # 每次转换独立目录，不使用外部 file_title 拼路径，避免同名覆盖或路径穿越。
        directory = Path(mkdtemp(prefix="pdf-", dir=out_dir))
        out_file = directory / "document.md"
        try:
            out_file.write_text(md_content, encoding="utf-8")
            output_path = str(out_file.resolve(strict=True))
        except Exception:
            # 只清理由本次创建的产物目录，不触碰其他文档。
            out_file.unlink(missing_ok=True)
            directory.rmdir()
            raise
    except Exception:
        add_failed_task(task_id, "node_pdf_to_md", "PDF 转换失败，请检查文件、解析依赖或输出目录权限")
        raise

    state["md_content"] = md_content
    state["md_path"] = output_path
    logger.info(f"node_pdf_to_md: 已生成 {len(md_content)} 字符的 Markdown")
    add_done_task(task_id, "node_pdf_to_md")
    return state
