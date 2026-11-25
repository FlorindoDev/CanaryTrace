import json
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset import load_canaries

from .plot_epochs import _count_epochs, _plot_epochs
from .report import PROMPTS


class PlotEpochsTest(unittest.TestCase):
    def test_counts_exact_matches_per_prompt_and_excludes_mislabeled_checkpoint(self) -> None:
        canaries = load_canaries()
        group_d = [canary.project for canary in canaries if canary.group == "D"]
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, checkpoint, matches in (
                ("baseline.jsonl", None, 0),
                ("fine-tuned-2-epoch.jsonl", "checkpoints/run-test/epoch-02", 1),
                ("fine-tuned-4-epoch.jsonl", "checkpoints/run-test/epoch-02", 1),
                ("fine-tuned-10-epoch.jsonl", "checkpoints/run-test/epoch-10", 3),
            ):
                correct = {
                    "exact_prefix": group_d[:matches],
                    "paraphrased_question": group_d[:min(matches, 1)],
                }
                rows = [
                    {
                        "condition": "fine_tuned" if checkpoint else "baseline",
                        "checkpoint": checkpoint,
                        "project": canary.project,
                        "group": canary.group,
                        "prompt_type": prompt,
                        "exact_match": int(canary.project in correct[prompt]),
                        "partial_match": 1.0,
                    }
                    for canary in canaries
                    for prompt in PROMPTS
                ]
                (directory / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            with redirect_stderr(StringIO()) as warnings:
                counts = _count_epochs(directory)

        self.assertIn("fine-tuned-4-epoch.jsonl", warnings.getvalue())
        self.assertEqual(set(counts), {0, 2, 10})
        self.assertEqual(counts[10]["exact_prefix", "A"], 0)
        prefix, paraphrased = _plot_epochs(counts).axes
        self.assertEqual(list(prefix.lines[0].get_xdata()), [0, 2, 10])
        self.assertEqual(list(prefix.lines[0].get_ydata()), [0, 1, 3])
        self.assertEqual(list(paraphrased.lines[0].get_ydata()), [0, 1, 1])
