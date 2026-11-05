import io
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.import_process.agent.nodes import node_pdf_to_md as node


class MineruFlowTests(unittest.TestCase):
    def run_upload(self, statuses, state="done"):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        pdf = directory / "sample.pdf"
        pdf.write_bytes(b"test PDF bytes")
        session = Mock()
        session.put.side_effect = [
            SimpleNamespace(status_code=status, text="upload error")
            for status in statuses
        ]
        stack.enter_context(patch.object(node.requests, "Session", return_value=session))
        post = stack.enter_context(patch.object(node.requests, "post"))
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "code": 0,
            "data": {"batch_id": "test-batch", "file_urls": ["https://example.invalid/upload"]},
        }
        get = stack.enter_context(patch.object(node.requests, "get"))
        get.return_value.status_code = 200
        get.return_value.json.return_value = {
            "code": 0,
            "data": {"extract_result": [{
                "state": state,
                "full_zip_url": "https://example.invalid/result.zip",
                "err_msg": "retry limit reached",
            }]},
        }
        sleep = stack.enter_context(patch.object(node.time, "sleep"))
        stack.enter_context(patch.object(node, "MINERU_BASE_URL", "https://example.invalid/api"))
        stack.enter_context(patch.object(node, "MINERU_API_TOKEN", "test-token"))
        stack.enter_context(patch.object(node.mineru_config, "model_version", "pipeline"))
        return pdf, directory, session, post, get, sleep

    def test_rate_limit_retry_preserves_request_and_uses_configured_model(self):
        pdf, directory, session, post, _, sleep = self.run_upload([503, 503, 200])
        result = node.step_2_upload_and_poll(pdf, directory)
        self.assertEqual(result, "https://example.invalid/result.zip")
        self.assertEqual(post.call_args.kwargs["json"]["model_version"], "pipeline")
        calls = session.put.call_args_list
        self.assertTrue(all(call == calls[0] for call in calls))
        self.assertNotIn("headers", calls[0].kwargs)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10])
        session.close.assert_called_once()

    def test_upload_failures_are_bounded_and_never_start_polling(self):
        for statuses, expected_calls in [([503] * 4, 4), ([403], 1)]:
            with self.subTest(statuses=statuses):
                pdf, directory, session, _, get, _ = self.run_upload(statuses)
                with self.assertRaisesRegex(RuntimeError, "上传失败"):
                    node.step_2_upload_and_poll(pdf, directory)
                self.assertEqual(session.put.call_count, expected_calls)
                get.assert_not_called()
                session.close.assert_called_once()

    def test_successful_json_response_can_contain_failed_parse(self):
        pdf, directory, session, _, _, _ = self.run_upload([200], state="failed")
        with self.assertRaisesRegex(RuntimeError, "解析任务失败.*test-batch"):
            node.step_2_upload_and_poll(pdf, directory)
        session.close.assert_called_once()

    def test_extracts_and_renames_markdown_while_preserving_images(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("full.md", "# Result\n![diagram](images/test.png)")
            z.writestr("images/test.png", b"image bytes")
            z.writestr("layout.json", "{}")
        with tempfile.TemporaryDirectory() as directory:
            response = SimpleNamespace(status_code=200, content=archive.getvalue())
            with patch.object(node.requests, "get", return_value=response):
                result = Path(node.step_3_download_and_extract(
                    "https://example.invalid/result.zip", Path(directory), "manual"
                ))
            self.assertEqual(result, Path(directory) / "manual" / "manual.md")
            self.assertIn("# Result", result.read_text())
            self.assertEqual((result.parent / "images/test.png").read_bytes(), b"image bytes")


if __name__ == "__main__":
    unittest.main()
