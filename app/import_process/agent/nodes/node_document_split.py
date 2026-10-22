"""标题段落 + 字符窗口切分；offset 对应原 Markdown，页码缺失时为 None。"""
import hashlib
import re
from app.core.config import settings
from ._runtime import import_node


@import_node('node_document_split')
def node_document_split(state):
    state['chunks'] = []
    text = state['md_content']
    maximum, minimum = settings.chunk_max_size, settings.chunk_min_size
    ratio = settings.chunk_overlap_ratio
    if not text.strip() or not 0 < minimum <= maximum or not 0 <= ratio < 1:
        raise ValueError('正文或切分参数无效')
    source = state['local_file_path'] or state['md_path']
    document_id = hashlib.sha256((source + '\0' + text).encode()).hexdigest()
    headings, pages, fences = [], [], False
    offset = 0
    boundaries = [0]
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(('```', '~~~')):
            fences = not fences
        if not fences:
            h = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
            if h:
                headings.append((offset, len(h[1]), h[2]))
                boundaries.append(offset)
            p = re.search(r'<!--\s*page\s+(\d+)\s*-->', line)
            if p:
                pages.append((offset, int(p[1])))
        offset += len(line)
    boundaries = sorted(set(boundaries + [len(text)]))
    # 先按标题分段，将短段与下一段合并（不超过长度上限）。
    sections = []
    for start, end in zip(boundaries, boundaries[1:]):
        if sections and sections[-1][1] - sections[-1][0] < minimum and end - sections[-1][0] <= maximum:
            sections[-1] = (sections[-1][0], end)
        else:
            sections.append((start, end))
    chunks = []
    for start, stop in sections:
        while start < stop:
            end = min(start + maximum, stop)
            if end < stop:
                boundary = text.rfind('\n', start + minimum, end)
                if boundary > start:
                    end = boundary + 1
            content = text[start:end]
            if content.strip():
                stack = []
                for pos, level, title in headings:
                    if pos > start:
                        break
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                page_numbers = [number for pos, number in pages if pos <= start]
                page_end = [number for pos, number in pages if pos < end]
                chunks.append({'content': content, 'metadata': {
                    'document_id': document_id, 'source': source, 'file_title': state['file_title'],
                    'part': len(chunks), 'start': start, 'end': end,
                    'title': stack[-1][1] if stack else state['file_title'],
                    'parent_title': ' > '.join(t for _, t in stack[:-1]),
                    'page_start': page_numbers[-1] if page_numbers else None,
                    'page_end': page_end[-1] if page_end else None}})
            if end == stop:
                break
            start = max(start + 1, end - int(maximum * ratio))
    if not chunks:
        raise ValueError('无有效切片')
    state['chunks'] = chunks
    return state
