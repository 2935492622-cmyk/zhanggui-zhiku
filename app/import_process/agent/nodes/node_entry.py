"""验证本地 PDF/MD，设置互斥路由和绝对路径。

空路径或不支持类型记录 skipped，沿现有条件边结束。
支持类型但文件无效时抛异常并记录 failed；格式解析由下游负责。
"""

from pathlib import Path
import stat

from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import (
    TASK_STATUS_PROCESSING, TASK_STATUS_SKIPPED,
    add_done_task, add_failed_task, add_running_task,
    set_task_result, update_task_status,
)

SUPPORTED_MD = ".md"
SUPPORTED_PDF = ".pdf"


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """读取路径和 task_id，验证成功后写路径、标题和唯一分支标记。"""
    task_id = state["task_id"]
    # 只重置入口负责的字段，避免上一次路由污染本次检查。
    state.update(
        is_md_read_enabled=False, is_pdf_read_enabled=False,
        pdf_path="", md_path="", file_title="",
    )
    add_running_task(task_id, "node_entry")
    update_task_status(task_id, TASK_STATUS_PROCESSING)
    set_task_result(task_id, "error", "")
    set_task_result(task_id, "skip_reason", "")

    try:
        local_file_path = state["local_file_path"]
        if not isinstance(local_file_path, str):
            raise TypeError("local_file_path 必须是字符串")
        if "\x00" in local_file_path:
            raise ValueError("文件路径不能包含空字符")

        reason = ""
        if not local_file_path.strip():
            reason = "文件路径为空"
        else:
            # 不 strip 有效路径，避免改动文件名中有意义的空格。
            path = Path(local_file_path).expanduser()
            suffix = path.suffix.lower()
            if suffix not in (SUPPORTED_MD, SUPPORTED_PDF):
                reason = "不支持的文件类型，仅支持 PDF/MD"
        if reason:
            logger.warning(f"node_entry: {reason}，跳过导入")
            set_task_result(task_id, "skip_reason", reason)
            update_task_status(task_id, TASK_STATUS_SKIPPED)
            add_done_task(task_id, "node_entry")
            return state

        # stat 跟随符号链接；排除目录/FIFO，避免打开非普通文件时阻塞。
        file_stat = path.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("导入路径必须指向普通文件")
        if file_stat.st_size == 0:
            raise ValueError("不能导入空文件")
        # 实际读取验证权限；不在入口解析 PDF 或解码 Markdown。
        with path.open("rb") as source:
            if not source.read(1):
                raise ValueError("不能导入空文件")
        normalized_path = str(path.resolve(strict=True))
    except (OSError, ValueError, TypeError):
        # 任务结果只存通用错误，不复制路径或可能敏感的原始异常内容。
        add_failed_task(task_id, "node_entry", "入口文件验证失败，请检查路径、类型、大小和读取权限")
        raise

    # 所有验证通过后才开启分支；失败时两个标记始终为 False。
    state["file_title"] = path.stem
    if suffix == SUPPORTED_MD:
        state["md_path"] = normalized_path
        state["is_md_read_enabled"] = True
    else:
        state["pdf_path"] = normalized_path
        state["is_pdf_read_enabled"] = True
    add_done_task(task_id, "node_entry")
    return state


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from app.import_process.agent.state import create_default_state

    # 入口只检查基本合法性；这里的 PDF 示例并非真正的可解析 PDF。
    with TemporaryDirectory() as directory:
        for name in ("小米用户手册.md", "HAK180.PDF"):
            path = Path(directory) / name
            path.write_text("入口验证示例", encoding="utf-8")
            result = node_entry(create_default_state(task_id=name, local_file_path=str(path)))
            print(name, result["is_md_read_enabled"], result["is_pdf_read_enabled"])
