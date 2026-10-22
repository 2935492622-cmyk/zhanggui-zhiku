"""入口单元测试：真实临时文件；仅权限/IO 故障使用 Mock，不调用外部服务。"""

from pathlib import Path

import pytest

from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.state import create_default_state
from app.utils.task_utils import (
    get_done_task_list, get_running_task_list, get_task_result, get_task_status,
)


def _state(path):
    return create_default_state(task_id="test_entry", local_file_path=path)


@pytest.mark.parametrize("filename,is_md", [
    ("小米用户手册.md", True), ("万用表RS-12的使用.pdf", False),
    ("HAK180.PDF", False), ("说明.Md", True),
    ("HUAWEI.MateBook.B5-430.pdf", False), (" 带空格.md", True),
])
def test_valid_file(tmp_path, filename, is_md):
    path = tmp_path / filename
    path.write_bytes(b"nonempty input; PDF parsing belongs to the next node")
    result = node_entry(_state(str(path)))
    assert result["is_md_read_enabled"] is is_md
    assert result["is_pdf_read_enabled"] is (not is_md)
    assert result["md_path" if is_md else "pdf_path"] == str(path.resolve())
    assert result["pdf_path" if is_md else "md_path"] == ""
    assert result["file_title"] == path.stem
    assert get_running_task_list("test_entry") == []
    assert get_done_task_list("test_entry") == ["检查文件"]
    # 入口完成不等于整个导入任务完成。
    assert get_task_status("test_entry") == "processing"


@pytest.mark.parametrize("path", ["", "   ", "a.txt", "a"])
def test_skipped_input_clears_old_route(path):
    state = _state(path)
    state.update(is_md_read_enabled=True, is_pdf_read_enabled=True,
                 md_path="old.md", pdf_path="old.pdf", file_title="old")
    result = node_entry(state)
    _assert_no_route(result)
    assert get_task_status("test_entry") == "skipped"
    assert get_task_result("test_entry", "skip_reason")
    assert get_running_task_list("test_entry") == []


def _assert_no_route(state):
    assert not state["is_md_read_enabled"]
    assert not state["is_pdf_read_enabled"]
    assert state["md_path"] == state["pdf_path"] == state["file_title"] == ""


@pytest.mark.parametrize("kind,error", [
    ("missing", FileNotFoundError), ("directory", ValueError), ("empty", ValueError),
])
def test_invalid_file_fails(tmp_path, kind, error):
    path = tmp_path / "document.md"
    if kind == "directory":
        path.mkdir()
    elif kind == "empty":
        path.touch()
    state = _state(str(path))
    with pytest.raises(error):
        node_entry(state)
    _assert_no_route(state)
    _assert_failed()


def _assert_failed():
    assert get_task_status("test_entry") == "failed"
    assert get_task_result("test_entry", "error")
    assert get_running_task_list("test_entry") == []
    assert get_done_task_list("test_entry") == []


@pytest.mark.parametrize("value,error", [(None, TypeError), (123, TypeError), ("bad\x00.md", ValueError)])
def test_malformed_path(value, error):
    with pytest.raises(error):
        node_entry(_state(value))
    _assert_failed()


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_mock_file_open_failure(tmp_path, monkeypatch, error):
    path = tmp_path / "private.md"
    path.write_text("text", encoding="utf-8")

    def fail_open(*args, **kwargs):
        raise error("simulated sensitive detail")

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(error):
        node_entry(_state(str(path)))
    _assert_failed()
    assert "sensitive detail" not in get_task_result("test_entry", "error")


def test_relative_path_becomes_absolute(tmp_path, monkeypatch):
    (tmp_path / "说明.md").write_text("正文", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = node_entry(_state("说明.md"))
    assert result["md_path"] == str(tmp_path / "说明.md")


def test_reused_state_and_retry(tmp_path):
    pdf = tmp_path / "one.pdf"
    md = tmp_path / "two.md"
    pdf.write_bytes(b"PDF input")
    md.write_text("正文", encoding="utf-8")
    state = _state(str(pdf))
    node_entry(state)
    state["local_file_path"] = str(md)
    node_entry(state)
    assert state["is_md_read_enabled"] and not state["is_pdf_read_enabled"]
    assert state["pdf_path"] == ""
    # 已经完成过的节点，本次失败不能仍留在 done 中。
    md.unlink()
    with pytest.raises(FileNotFoundError):
        node_entry(state)
    _assert_no_route(state)
    _assert_failed()
    md.write_text("重试正文", encoding="utf-8")
    node_entry(state)
    assert get_task_result("test_entry", "error") == ""
    assert get_task_status("test_entry") == "processing"


def test_preserves_other_state_fields(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("正文", encoding="utf-8")
    state = _state(str(path))
    state["md_content"] = "downstream data"
    node_entry(state)
    assert state["task_id"] == "test_entry"
    assert state["local_file_path"] == str(path)
    assert state["md_content"] == "downstream data"


def test_real_corpus_file_is_recognized(first_real_pdf):
    result = node_entry(_state(str(first_real_pdf)))
    assert result["is_pdf_read_enabled"]
    assert result["file_title"] == first_real_pdf.stem
