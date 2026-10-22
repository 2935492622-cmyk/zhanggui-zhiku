"""
app/utils/task_utils.py —— 内存态的任务进度追踪

【解决什么问题】
    导入一本手册要跑 7 个节点，跑几十秒到几分钟。前端要显示
    「已完成的步骤 / 正在跑的步骤」，靠的就是这里维护的两张表。

    它回答两个问题：
      1. 这次任务（task_id）跑完了哪些节点、正在跑哪个节点？
      2. 最终结果是成功还是失败？

【上下游】
    上游：各节点调用 add_running_task / add_done_task
    下游：将来 FastAPI 的进度查询接口（M3 阶段）+ SSE 推送

【和教程版的差异】
    教程的 task_utils 依赖 app/utils/sse_utils.py 做 SSE 实时推送，
    你现在还没有 Web 层（FastAPI 都没起），SSE 队列是纯负担，
    所以这里把 sse_utils / is_stream 全部砍掉，只留内存态。
    等 M3 做 Web 层时，只需要在 add_running_task / add_done_task 末尾
    加一行推送，接口不用变。

【局限（要知道）】
    数据全在内存字典里：进程重启就没了，多进程/多副本部署也不共享。
    正式项目要把这张表挪到 MongoDB 或 Redis。M0 阶段够用。
"""

from typing import Dict, List

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_SKIPPED = "skipped"

# key: task_id -> value: 节点名列表
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}
# key: task_id -> value: 状态字符串
_tasks_status: Dict[str, str] = {}
# key: task_id -> value: 结果字段（如 answer / error）
_tasks_result: Dict[str, Dict[str, str]] = {}

# 节点名 -> 中文展示名（未来给前端用）
_NODE_NAME_TO_CN: Dict[str, str] = {
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_document_split": "文档切分",
    "node_item_name_recognition": "主体名称识别",
    "node_bge_embedding": "向量生成",
    "node_import_milvus": "导入向量库",
    "END": "处理完成",
}


def _ensure_task(task_id: str) -> None:
    """懒初始化：第一次见到这个 task_id 时把四个结构建出来"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    """标记某节点开始运行"""
    _ensure_task(task_id)
    # 同一节点重试时，上次的完成记录不能代表本次成功。
    _tasks_done_list[task_id] = [n for n in _tasks_done_list[task_id] if n != node_name]
    running = _tasks_running_list[task_id]
    if node_name not in running:  # 重跑同一节点时不重复追加
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    """标记某节点完成（同时从 running 里移走，running 里可能有重名，全清）"""
    _ensure_task(task_id)
    _tasks_running_list[task_id] = [n for n in _tasks_running_list[task_id] if n != node_name]
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存任务结果字段，例如 set_task_result(task_id, "error", "xxx")"""
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def add_failed_task(task_id: str, node_name: str, error: str) -> None:
    """结束失败节点的 running 状态，保留其他进度，不追加 done。"""
    _ensure_task(task_id)
    _tasks_running_list[task_id] = [n for n in _tasks_running_list[task_id] if n != node_name]
    _tasks_done_list[task_id] = [n for n in _tasks_done_list[task_id] if n != node_name]
    set_task_result(task_id, "error", error)
    update_task_status(task_id, TASK_STATUS_FAILED)


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def update_task_status(task_id: str, status_name: str) -> None:
    _tasks_status[task_id] = status_name


def get_task_status(task_id: str) -> str:
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """已完成节点列表（中文，给前端展示用）"""
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_running_task_list(task_id: str) -> List[str]:
    """正在运行节点列表（中文）"""
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def clear_task(task_id: str) -> None:
    """任务结束后清理，避免内存字典无限增长"""
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)


if __name__ == "__main__":
    tid = "demo_001"
    add_running_task(tid, "node_entry")
    print("运行中:", get_running_task_list(tid))
    add_done_task(tid, "node_entry")
    add_running_task(tid, "node_pdf_to_md")
    print("已完成:", get_done_task_list(tid))
    print("运行中:", get_running_task_list(tid))
    clear_task(tid)
    print("清理后已完成:", get_done_task_list(tid))
