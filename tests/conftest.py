"""
tests/conftest.py —— pytest 的共享夹具（fixture）

【解决什么问题】
    现在每个测试都在重复同一件事：
        state = create_default_state(task_id="g01", local_file_path="doc/xxx.pdf")
    这有三个毛病：
      1. 重复代码，改一处要改七处；
      2. task_id 得靠人肉保证不重复（写重了测试之间会互相污染）；
      3. task_utils 的内存表会随测试越积越多，谁也没清理。

    fixture 就是把这类「每个用例都要的准备工作」抽出来，由 pytest 自动注入。

【fixture 是什么（一句话）】
    一个被 @pytest.fixture 装饰的函数；测试函数只要**在参数里写上它的名字**，
    pytest 就会在调用测试前先执行它，把 return 的值传进来。
    名字必须对得上——这是 pytest 唯一的"魔法"，靠参数名匹配，靠 import。

【为什么这个文件叫 conftest.py】
    pytest 硬性约定：这个名字的文件会被**自动加载**，目录下所有测试直接用，
    不需要 import。所以它是放共享夹具的唯一正确位置。

【autouse 是什么】
    加了 autouse=True 的 fixture **不需要在参数里写**，对当前目录下每个用例自动生效。
    适合放"清理、隔离"这类所有用例都需要、但逻辑里用不到的东西。

【上下游】
    上游：pytest 自身
    下游：tests/ 下所有测试文件
"""

import pytest

from app.import_process.agent.state import create_default_state
from app.utils import task_utils


@pytest.fixture
def make_state():
    """
    state 工厂：造一份全新默认 state，可按关键字覆盖任意字段。

    用法（注意：测试函数参数里写 make_state，pytest 就把这个函数塞进来）：

        def test_xxx(make_state):
            state = make_state(task_id="t1", local_file_path="doc/a.pdf")

    为什么返回「一个函数」而不是直接返回 state：
        因为每个用例要的字段不一样。直接返回 state 的话，覆盖字段就只能在 fixture 里写死，
        那就跟没抽一样。返回函数（这叫工厂模式 fixture）调用方才灵活。
    """

    def _make(**overrides):
        return create_default_state(**overrides)

    return _make


def _snapshot_task_ids() -> set:
    """当前内存里已存在的所有 task_id"""
    # 这里访问的是 task_utils 的私有字典：测试代码属于同一个项目内部，
    # 需要拿到全集才能做精确的交集清理。业务代码不要这么写。
    return set(task_utils._tasks_done_list) | set(task_utils._tasks_running_list)


@pytest.fixture(autouse=True)
def auto_clear_task_registry():
    """
    自动隔离：每个用例跑完，清掉它自己产生的任务记录。

    为什么需要：task_utils 是纯内存字典（见它的 docstring「局限」），
    进程不退出就一直在。跑几十个测试会攒几十个 task_id，
    而且下一个用例万一撞了 id，断言就直接被上游污染。

    为什么用「前后快照做差」而不是直接全清：
        全清会误伤别的 fixture / 别的用例正在用的数据；
        做差只清本次新增的，边界干净。

    yield 的作用：yield 之前是 setup，之后是 teardown。
    即使用例中途抛异常，teardown 也照样执行——这是 pytest 保证的。
    """
    before = _snapshot_task_ids()
    yield
    for task_id in _snapshot_task_ids() - before:
        task_utils.clear_task(task_id)


@pytest.fixture
def first_real_pdf():
    """
    doc/ 语料目录下的第一本真实 PDF；目录为空时自动跳过用到它的用例。

    用法：

        def test_something(first_real_pdf):
            state = make_state(task_id="t", local_file_path=str(first_real_pdf))

    为什么要在 fixture 里 skip 而不是在测试里 skip：
        以后会有多个测试需要真实 PDF，判断逻辑写一处就够了；
        而且语料没准备好时它们会统一显示成 skipped，不会伪装成通过。
    """
    from app.core.config import settings

    pdfs = sorted(settings.doc_path.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"doc/ 下没有 PDF（语料目录: {settings.doc_path}）")
    return pdfs[0]
