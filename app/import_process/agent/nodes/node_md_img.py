"""读取 UTF-8 Markdown，规范化内联图片路径；不上传、不下载、不做 VLM。

支持 ![alt](path) 和 ![alt](<path with spaces>)；引用式/HTML 图片保留原文。
"""
import re
from pathlib import Path
from urllib.parse import urlsplit, unquote
from app.core.logger import logger
from ._runtime import import_node

_IMAGE = re.compile(r'!\[([^\]\n]*)\]\((<[^>\n]+>|[^\s()]+)(\s+"[^"\n]*")?\)')


@import_node('node_md_img')
def node_md_img(state):
    text = state['md_content']
    path = Path(state['md_path']).expanduser()
    if not text:
        text = path.read_text(encoding='utf-8-sig')
    if not text.strip():
        raise ValueError('Markdown 为空')
    # 分离代码块/行内代码，示例里的图片语法不能被改写。
    parts = re.split(r'(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)', text)
    def replace(match):
        alt, target, title = match.groups()
        target = target.strip('<>')
        scheme = urlsplit(target).scheme.lower()
        if scheme in ('http', 'https', 'file') or target.startswith('//'):
            return match.group(0)
        if scheme:
            logger.warning('图片包含不支持的 URI scheme，已转为替代文本')
            return f'[图片：{alt or "无描述"}]'
        image = (path.parent / unquote(target)).resolve()
        if not image.is_file():
            logger.warning('Markdown 引用的本地图片不存在，保留引用供排查')
        return f'![{alt}](<{image.as_uri()}>{title or ""})'
    state['md_content'] = ''.join(p if i % 2 else _IMAGE.sub(replace, p)
                                  for i, p in enumerate(parts))
    return state
