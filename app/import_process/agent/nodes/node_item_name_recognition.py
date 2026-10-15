"""
app/import_process/agent/nodes/node_item_name_recognition.py —— 商品主体名识别

【解决什么问题 —— 这是本项目的核心难点】
    用户问「HAK180 怎么换色带」，如果只做向量检索，模型很可能召回
    「RS-12 万用表」里讲色带的段落——两本手册都提到「色带」，语义很近。
    要命中的准，必须先知道「这篇文档讲的是哪个商品」，把它作为一级筛选条件。

    所以这一步：从文档内容里识别出它讲的是哪个商品（型号），
    存进 state["item_name"]，后面写双层索引 + 检索时先筛商品再找段落。

【为什么难（这才是简历上能讲的东西）】
    型号在真实手册里的写法极不规范：
      - "HAK180" / "hak-180" / "HAK 180" / "兄弟HAK180烫金机"
      - 同一本手册里可能出现多个型号（兼容机型列表）
      - 有些手册标题里根本没有型号（"使用说明书.pdf"）
    所以单纯正则匹配必然漏，必须用 LLM + 规则兜底的组合。

【入参 / 出参】
    读：state["md_content"]（取前几段就够，省 token）、state["file_title"]
    写：state["item_name"]

【实现建议（两步）】
    1. 先用规则（文件名 + 正则抽型号骨架）跑 baseline，不依赖 key；
    2. 再用 LLM（qwen-flash）从正文前几段抽取，让规则结果做兜底校验。
    没有 key 时（settings.has_llm_key == False）走规则路径，链路照样能跑。

【当前状态】空壳（return state）。
"""

from app.core.logger import node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    识别文档核心描述的物品/商品名称 (Item Name)。
    """
    add_running_task(state["task_id"], "node_item_name_recognition")

    # TODO: 1) 规则路径：文件名 + 正则抽型号（无 key 也能跑）
    #      2) LLM 路径：取正文前几段让 qwen-flash 抽取（有 key 时启用，规则结果做兜底）
    # 结果写入 state["item_name"]

    add_done_task(state["task_id"], "node_item_name_recognition")
    return state
