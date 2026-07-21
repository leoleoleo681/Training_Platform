import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from platform_runtime import JobRuntime


class JobRuntimeTest(unittest.TestCase):
    def test_writes_failed_status_and_error_event(self):
        temp_dir = (
            Path(__file__).resolve().parents[3]
            / ".codex_tmp_metrics"
            / "platform_runtime_tests"
            / uuid.uuid4().hex
        )
        runtime = None
        try:
            job_root = temp_dir / "evaluation" / "demo" / "tests" / "test-1"
            runtime = JobRuntime(
                job_root=job_root,
                job_type="validate",
                task_id="task-1",
                model_name="demo",
                test_id="test-1",
                status_interval_seconds=0,
            )
            runtime.start()
            runtime.change_phase("VALIDATING")
            runtime.fail("EVALUATION_FAILED", "invalid sample")

            status = json.loads(
                (job_root / "runtime" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "FAILED")
            self.assertEqual(status["phase"], "VALIDATING")
            self.assertEqual(status["error"]["code"], "EVALUATION_FAILED")
            events = [
                json.loads(line)
                for line in (job_root / "runtime" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(events[-1]["type"], "job_failed")
            self.assertEqual(events[-1]["level"], "ERROR")
        finally:
            if runtime is not None:
                runtime.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_writes_status_events_metrics_and_run_log(self):
        temp_dir = (
            Path(__file__).resolve().parents[3]
            / ".codex_tmp_metrics"
            / "platform_runtime_tests"
            / uuid.uuid4().hex
        )
        runtime = None
        try:
            job_root = temp_dir / "models" / "demo"
            runtime_dir = job_root / "runtime"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "status.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "task_id": "precreated-task",
                    }
                ),
                encoding="utf-8",
            )

            runtime = JobRuntime(
                job_root=job_root,
                job_type="train",
                task_id="new-task",
                model_name="demo",
                status_interval_seconds=0,
            )
            runtime.start()
            runtime.change_phase("TRAINING")
            runtime.update_progress(
                current=5,
                total=10,
                epoch=1,
                total_epochs=2,
                force=True,
            )
            runtime.emit_metrics(
                phase="train",
                step=5,
                epoch=1,
                values={"loss": 0.25},
            )
            print("runtime-log-check")
            runtime.succeed()

            status = json.loads(
                (runtime_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "SUCCEEDED")
            self.assertEqual(status["phase"], "FINALIZING")
            self.assertEqual(status["task_id"], "precreated-task")
            self.assertEqual(status["created_at"], "2026-01-01T00:00:00+00:00")
            self.assertEqual(status["progress"]["percentage"], 100.0)

            events = [
                json.loads(line)
                for line in (runtime_dir / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(events[0]["type"], "job_started")
            self.assertEqual(events[-1]["type"], "job_succeeded")
            self.assertEqual(
                [event["seq"] for event in events],
                list(range(1, len(events) + 1)),
            )

            metrics = [
                json.loads(line)
                for line in (runtime_dir / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(metrics[0]["values"]["loss"], 0.25)
            self.assertIn(
                "runtime-log-check",
                (job_root / "logs" / "run.log").read_text(encoding="utf-8"),
            )
        finally:
            if runtime is not None:
                runtime.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
