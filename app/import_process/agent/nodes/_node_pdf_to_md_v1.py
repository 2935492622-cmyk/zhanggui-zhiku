"""
app/import_process/agent/nodes/node_pdf_to_md.py —— PDF 转 Markdown

【解决什么问题】
    语料是 88 本设备手册 PDF（约 707MB）。PDF 是非结构化数据：有分栏、表格、
    页眉页脚、图片，直接按文本抽出来会乱成一团。
    这一步把 PDF 转成 Markdown，保留标题层级，后面才能按标题切分。

【入参 / 出参】
    读：state["pdf_path"]、state["file_title"]
    写：state["md_content"]（全文 Markdown）、state["md_path"]（落盘位置）

【实现路线（两步走，先易后难）】
    第一步（当前实现）：pymupdf 抽文本 + **按字号推断标题层级** + 去页眉页脚。
        快、轻量、纯 CPU，对纯文本型手册够用。缺点：表格会散架、多栏顺序可能错。
    第二步（之后替换）：MinerU（magic-pdf）。它是专门做 PDF→Markdown 的，
        能还原表格和阅读顺序，是本项目简历上的技术点。
        但它下载和 GPU 依赖重，所以先用第一步把整条链路打通，
        确认链路没问题了再换 MinerU——换的时候只改这一个文件，其他地方不用动。

【⚠️ 需要先装包】
    uv pip install pymupdf --index-url https://mirrors.aliyun.com/pypi/simple/
    （没装时 fitz = None，跑 pdf_to_markdown 会给出明确报错，不是莫名其妙的 ImportError）

【三个关键设计（面试能讲的细节）】
    1. 为什么用 get_text("dict") 而不是 get_text("text")：
       "text" 只给纯文本，字号信息全丢。PDF 里没有 Markdown 的 # 标记，
       **字号是我们识别标题的唯一线索**，所以必须拿到 span 级的 size。
    2. 正文字号按「字符数」加权统计，不是按行数：
       一页里标题可能只有 5 个字而正文有 500 字。按行数统计会让两者票数接近。
    3. 页眉页脚必须去掉：
       设备手册每页都有页眉（"HAK180 使用说明书"）和页脚。
       留着会让 88 本手册因为页眉相似而向量撞车，检索精度直接下降。
"""

import re
from collections import Counter
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

# pymupdf >= 1.24 官方推荐 `import pymupdf`；老版本只有 `import fitz`。两种都兼容。
# 都没装时置为 None，让调用方拿到清晰报错，而不是 import 阶段就崩。
try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    try:
        import fitz  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        fitz = None  # type: ignore[assignment]

# 页码行：纯数字、"第 3 页"、"第 3 页 / 共 20 页" 这类
_PAGE_NUM_RE = re.compile(
    r"^(?:[-\s\u2013\u2014]*\d{1,4}[-\s\u2013\u2014]*"
    r"|第\s*\d{1,4}\s*页(?:\s*/\s*共\s*\d{1,4}\s*页)?)$"
)

# 字号超过正文多少倍算标题。这两个是经验值，不同手册要调（做成模块常量方便改）
_H1_RATIO = 1.30
_H2_RATIO = 1.15
# 标题不能太长，否则只是加粗的正文段落
_MAX_HEADING_LEN = 60
# 页眉页脚判定：一行在多少比例的页里出现在首/末两行
_BOILERPLATE_RATIO = 0.5
_BOILERPLATE_MAX_LEN = 60


def _extract_page_lines(page) -> list[tuple[str, float]]:
    """
    从一页里抽出 [(文本行, 这行最大字号)]。

    block["type"]: 0=文本块, 1=图片块。只要文本，图片留给 node_md_img 处理。
    """
    lines: list[tuple[str, float]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            size = max(s["size"] for s in spans)
            lines.append((text, size))
    return lines


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
    """
    找出页眉页脚：在每页**开头两行或结尾两行**里反复出现的短文本。

    判定逻辑：出现在 >= 50% 页面的首尾位置、且长度不超过 60 字的行，认为是样板。
    """
    counter: Counter = Counter()
    for page in pages:
        if not page:
            continue
        candidates = {t for t, _ in page[:2]} | {t for t, _ in page[-2:]}
        for text in candidates:
            if len(text) <= _BOILERPLATE_MAX_LEN:
                counter[text] += 1
    total = max(len(pages), 1)
    return {t for t, c in counter.items() if c / total >= _BOILERPLATE_RATIO}


def _page_to_markdown(
    page_lines: list[tuple[str, float]],
    body_size: float,
    boilerplate: set[str],
) -> list[str]:
    """把一页的行转成 Markdown 行：标题加 # / ##，页眉页脚与页码丢弃。"""
    out: list[str] = []
    for text, size in page_lines:
        if text in boilerplate or _PAGE_NUM_RE.match(text):
            continue
        if len(text) <= _MAX_HEADING_LEN:
            if size >= body_size * _H1_RATIO:
                out.append(f"# {text}")
                continue
            if size >= body_size * _H2_RATIO:
                out.append(f"## {text}")
                continue
        out.append(text)
    return out


def pdf_to_markdown(pdf_path: str | Path) -> str:
    """
    PDF -> Markdown 字符串。单独抽成纯函数，是为了能脱离 LangGraph 直接测。

    页分隔用 `<!-- page N -->` 而不是 `---`：
        `---` 在 Markdown 里是分隔线，会被后续切分逻辑当噪音；
        HTML 注释既不影响渲染，又能让 document_split 反查出
        「这段落在原文第几页」，写进 chunk 的 metadata 里。
    """
    if fitz is None:
        raise RuntimeError(
            "pymupdf 未安装。先执行："
            "uv pip install pymupdf --index-url https://mirrors.aliyun.com/pypi/simple/"
        )

    pdf_path = Path(pdf_path)
    pages: list[list[tuple[str, float]]] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(_extract_page_lines(page))

    body_size = _body_font_size(pages)
    boilerplate = _detect_boilerplate(pages)
    logger.debug(
        f"node_pdf_to_md: 共 {len(pages)} 页，正文字号≈{body_size}，"
        f"识别到 {len(boilerplate)} 条页眉页脚"
    )

    parts: list[str] = []
    for idx, page_lines in enumerate(pages, start=1):
        md_lines = _page_to_markdown(page_lines, body_size, boilerplate)
        if not md_lines:
            continue  # 整页无文本（纯图片页 / 扫描页），跳过
        parts.append(f"<!-- page {idx} -->\n" + "\n".join(md_lines))
    return "\n\n".join(parts)


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """
    add_running_task(state["task_id"], "node_pdf_to_md")

    pdf_path = state.get("pdf_path") or ""
    if not pdf_path:
        logger.warning("node_pdf_to_md: pdf_path 为空，跳过转换")
        add_done_task(state["task_id"], "node_pdf_to_md")
        return state

    path = Path(pdf_path)
    if not path.exists():
        # 降级而不是抛异常：批量导入时混进坏路径不该让整批崩掉。
        # 打 error 级日志，事后能从日志里捞出来是哪一本。
        logger.error(f"node_pdf_to_md: 文件不存在，跳过：{pdf_path}")
        add_done_task(state["task_id"], "node_pdf_to_md")
        return state

    try:
        md_content = pdf_to_markdown(path)
    except Exception:
        # 解析失败不吞异常：静默给个空 md_content，下游排查会非常难。
        # 将来做批量导入时，在批量那一层 try/except 逐本兜底，节点本身保持严格。
        logger.exception(f"node_pdf_to_md: PDF 解析失败：{pdf_path}")
        raise

    file_title = state.get("file_title") or path.stem
    out_dir = settings.parsed_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{file_title}.md"
    out_file.write_text(md_content, encoding="utf-8")

    state["md_content"] = md_content
    state["md_path"] = str(out_file)
    logger.info(
        f"node_pdf_to_md: {file_title} -> {len(md_content)} 字符，"
        f"{md_content.count('<!-- page')} 页有文本，落盘 {out_file}"
    )

    add_done_task(state["task_id"], "node_pdf_to_md")
    return state


if __name__ == "__main__":
    # 手动自检：python -m app.import_process.agent.nodes.node_pdf_to_md
    target = next(iter(sorted(settings.doc_path.glob("*.pdf"))), None)
    if target is None:
        print("doc/ 里没有 PDF，先拷语料进来")
    else:
        md = pdf_to_markdown(target)
        print(f"文件: {target.name}")
        print(f"产出: {len(md)} 字符")
        print("-" * 60)
        print("\n".join(md.splitlines()[:40]))
