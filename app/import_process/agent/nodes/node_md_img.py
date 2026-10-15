"""
app/import_process/agent/nodes/node_md_img.py —— Markdown 中的图片处理

【解决什么问题】
    设备手册里大量「按钮位置示意图」「接口定义图」。图片在 Markdown 里只是一个
    本地相对路径，向量化时这段文字对检索毫无帮助（甚至是有噪声）。
    这一步要做的事：把图片变成「能被检索到的文字描述」。

【入参 / 出参】
    读：state["md_content"]、state["file_title"]
    写：state["md_content"]（把图片链接替换成可访问 URL + 图片描述）

【三步实现】
    1. 扫描 Markdown 里的图片引用（![alt](path)）；
    2. 上传到对象存储（MinIO；M0 阶段 MINIO_ENABLED=false，用本地目录代替）；
    3. 调多模态模型（qwen-vl-flash）给图片生成一句话描述，
       把描述插回正文——这样「XX 接口长什么样」这种问题也能检索到。

【注意】这一步依赖 LLM key。没配 key 时（settings.has_llm_key == False）
    应该跳过第 3 步，只做链接替换，保证链路在无 key 环境下也能跑通。

【当前状态】空壳（return state）。
"""

from app.core.logger import node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    处理 Markdown 中的图片资源 (Image)。
    """
    add_running_task(state["task_id"], "node_md_img")

    # TODO: 1) 扫描图片链接 -> 2) 上传 MinIO/本地目录 -> 3) VLM 生成描述并插回正文
    # 注意：第 3 步要判 settings.has_llm_key，没 key 就跳过，保证链路能跑通

    add_done_task(state["task_id"], "node_md_img")
    return state
