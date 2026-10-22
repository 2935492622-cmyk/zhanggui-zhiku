"""节点生命周期：成功才记 done；失败清理 running 并抛出脱敏异常。"""
from functools import wraps
from app.core.logger import node_log
from app.utils.task_utils import (add_running_task, add_done_task, add_failed_task,
    update_task_status, set_task_result, TASK_STATUS_PROCESSING)


def import_node(name):
    def decorate(func):
        @node_log(name)
        @wraps(func)
        def run(state):
            task = state['task_id']
            add_running_task(task, name)
            update_task_status(task, TASK_STATUS_PROCESSING)
            set_task_result(task, 'error', '')
            try:
                result = func(state)
            except Exception:
                message = f'{name} 失败，请检查输入、依赖和服务配置'
                add_failed_task(task, name, message)
                # 不将 SDK 异常（可能包含请求头和凭据）传到日志。
                raise RuntimeError(message) from None
            add_done_task(task, name)
            return result
        return run
    return decorate
