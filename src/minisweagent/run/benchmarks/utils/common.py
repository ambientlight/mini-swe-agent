"""Shared agent utilities for benchmark runners."""

import minisweagent.models
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager


class ProgressTrackingAgent(DefaultAgent):
    """Agent that reports per-step progress via :class:`RunBatchProgressManager`."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id
        # Stream live token count + reasoning/content text into this instance's status line.
        if hasattr(self.model, "_progress_callback"):
            self.model._progress_callback = self._on_stream_token

    def _on_stream_token(self, n_tok: int, text: str) -> None:
        total = minisweagent.models.GLOBAL_MODEL_STATS.format_tokens()
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.n_calls + 1:3d} | ⟳ {n_tok} gen | {total} out"
        )
        self.progress_manager.update_instance_stream_text(self.instance_id, text)

    def step(self) -> dict:
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} | calling LLM...")
        return super().step()
