"""
app/core/logger.py —— 全项目共用的日志对象

【解决什么问题】
    RAG 链路是「解析 → 切分 → 识别 → 向量化 → 入库」一串节点，
    出问题时要能一眼看出「哪个节点、哪个 task_id、耗了多久」。
    print 做不到这件事（没有级别、没有时间、没法关掉某部分）。

    这个文件做三件事：
      1. 控制台彩色输出（开发时看）+ 文件按天切分（事后回溯），由 .env 开关控制；
      2. 修正调用位置：loguru 默认会把位置指向 loguru 内部，这里穿透到业务代码；
      3. 提供 @node_log / @step_log 两个装饰器，自动打「开始 / 完成 / 异常 + 耗时」。

【上下游】
    上游：app/core/config.py（读 LOG_* 配置）
    下游：所有模块 `from app.core.logger import logger`

【和教程版的差异（按你的环境改的，不是照抄）】
    1. 教程每个配置项都是 os.getenv，这里统一从 settings 读（类型已校验）；
    2. 教程的 fix_log_position 用 inspect.stack()，每次写日志都会构建完整调用栈
       （含源码上下文），压测时这是热点；这里改成 sys._getframe 逐层向上，
       只取 frame 不取上下文，快一个数量级；
    3. 补了 LOG_FIX_POSITION 开关（你 .env 里有这个字段，但教程代码没实现它）。
"""

import os
import sys
import time
from functools import wraps
from typing import Mapping

from loguru import logger as _base_logger

from app.core.config import settings

_THIS_FILE = os.path.abspath(__file__)

# -------------------------- 第一步：从配置中心读日志配置 --------------------------
LOG_CONSOLE_ENABLE = settings.log_console_enable
LOG_CONSOLE_LEVEL = settings.log_console_level.upper()
LOG_FILE_ENABLE = settings.log_file_enable
LOG_FILE_LEVEL = settings.log_file_level.upper()
LOG_FILE_RETENTION = settings.log_file_retention
LOG_FIX_POSITION = settings.log_fix_position

# -------------------------- 第二步：日志路径（由 config 派生，不手拼） --------------------------
LOG_DIR = settings.log_path
LOG_FILE_PATH = LOG_DIR / "app_{time:YYYYMMDD}.log"

# -------------------------- 第三步：日志格式 --------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)


def init_logger():
    """
    初始化全局日志配置。
    必须先 remove() 掉 loguru 自带的那个默认 handler，否则每条日志打印两遍。
    """
    _base_logger.remove()

    if LOG_CONSOLE_ENABLE:
        _base_logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True,  # 多进程/多线程写日志不出乱序
        )

    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _base_logger.add(
            sink=str(LOG_FILE_PATH),
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",  # 每天零点换文件
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",  # 中文不乱码
            enqueue=True,
            backtrace=True,
            diagnose=False,  # 异常堆栈不展开局部变量，避免暴露配置和客户端凭据
        )

    return _base_logger


_logger = init_logger()


def fix_log_position(record):
    """
    穿透 loguru 内部帧和本文件自身的帧，把日志定位到真正调用它的业务代码。

    为什么需要：不修正的话，日志里 name/function/line 全是 loguru 内部的 _logger.py，
    排查时完全不知道是哪一行打的。
    """
    if not LOG_FIX_POSITION:
        return

    frame = sys._getframe()
    depth = 0
    while frame is not None and depth < 40:  # 40 层兜底，防止极端情况死循环
        filename = frame.f_code.co_filename
        funcname = frame.f_code.co_name
        # 跳过 loguru 内部帧 + 本文件自身的帧
        if (
            "_logger.py" in filename
            or funcname == "_log"
            or filename == _THIS_FILE
            or "site-packages" in filename
        ):

            frame = frame.f_back
            depth += 1
            continue
        record["name"] = os.path.basename(filename)
        record["function"] = funcname
        record["line"] = frame.f_lineno
        return

    # 走到底都没找到业务帧（极端情况），保持原样


logger = _logger.patch(fix_log_position)


# -------------------------- 装饰器：给节点/步骤自动加日志 --------------------------
def _trace_id(state) -> str:
    """从 state 里取追踪 ID，优先 session_id（问答流程），其次 task_id（导入流程）"""
    if isinstance(state, Mapping):
        return str(state.get("session_id") or state.get("task_id") or "-")
    return "-"


def node_log(node_name: str):
    """
    节点日志装饰器。用法：

        @node_log("node_entry")
        def node_entry(state):
            ...

    自动打印：节点开始 / 节点完成(带耗时) / 节点异常(带完整堆栈)。
    异常**不吞**，原样向上抛，避免把错误藏起来。
    """

    def deco(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            trace_id = _trace_id(state)
            start_ts = time.time()
            logger.info(f"[{node_name}] 节点开始，追踪ID={trace_id}")
            try:
                result = func(state, *args, **kwargs)
                cost_ms = int((time.time() - start_ts) * 1000)
                logger.info(f"[{node_name}] 节点完成，追踪ID={trace_id}，耗时={cost_ms}ms")
                return result
            except Exception:
                logger.exception(f"[{node_name}] 节点异常，追踪ID={trace_id}")
                raise

        return wrapper

    return deco


def step_log(step_name: str):
    """步骤级日志装饰器，粒度比 node_log 更细（给节点内部的子函数用）"""

    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_ts = time.time()
            logger.info(f"[{step_name}] 步骤开始")
            try:
                result = func(*args, **kwargs)
                cost_ms = int((time.time() - start_ts) * 1000)
                logger.info(f"[{step_name}] 步骤完成，耗时={cost_ms}ms")
                return result
            except Exception:
                logger.exception(f"[{step_name}] 步骤异常")
                raise

        return wrapper

    return deco


if __name__ == "__main__":
    logger.debug("【自检】debug 级别（控制台级别为 INFO 时这条不显示）")
    logger.info("【自检】info 级别")
    logger.warning("【自检】warning 级别")
    print(f"日志文件输出路径：{LOG_FILE_PATH}")
    try:
        _ = 10 / 0
    except Exception:
        logger.exception("【自检】异常堆栈演示")
