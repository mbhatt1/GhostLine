"""Tests that the it-test.yaml playbook progresses correctly and never loops infinitely.

These tests load the real playbook file and simulate user utterances through
the PlaybookRunner, verifying:
  - Single-cycle stages advance after one utterance
  - Multi-cycle stages allow conversation before advancing
  - Gated stages (TRIAL_CLOSE, CLOSE) require regex to succeed
  - CLOSE → OBJECTION escape is bounded
  - Full happy-path completes in a reasonable number of cycles
  - The playbook YAML loads without schema errors
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ghostline.playbook import Playbook
from ghostline.playbook_runner import PlaybookRunner
from ghostline.taxonomy import SalesStage

_PLAYBOOK_PATH = Path(__file__).resolve().parents[2] / "playbooks" / "it-test.yaml"


@pytest.fixture
def playbook() -> Playbook:
    return Playbook.from_file(_PLAYBOOK_PATH)


@pytest.fixture
def runner(playbook: Playbook) -> PlaybookRunner:
    return PlaybookRunner(playbook)


def _advance_to(runner: PlaybookRunner, target: SalesStage) -> None:
    """Advance the runner to the target stage with cooperative answers."""
    safety = 0
    while runner.current_stage != target:
        if runner.current_stage == SalesStage.TRIAL_CLOSE:
            runner.evaluate("yes let's do it")
        elif runner.current_stage == SalesStage.CLOSE:
            runner.evaluate("done, it's installed")
        else:
            runner.evaluate("okay sure sounds good")
        safety += 1
        if safety > 60:
            pytest.fail(f"Could not reach {target.name} in 60 utterances")


class TestPlaybookLoads:
    def test_loads_without_error(self, playbook: Playbook) -> None:
        assert "Emergency Security Patch" in playbook.meta.name

    def test_has_expected_stages(self, playbook: Playbook) -> None:
        stage_names = [s.stage for s in playbook.sequence]
        assert SalesStage.RAPPORT in stage_names
        assert SalesStage.CLOSE in stage_names
        assert SalesStage.REPORTING in stage_names

    def test_no_duplicate_stages(self, playbook: Playbook) -> None:
        stage_names = [s.stage for s in playbook.sequence]
        assert len(stage_names) == len(set(stage_names))


class TestSingleCycleStages:
    """FOLLOW_UP has max_cycles=1, should advance after one utterance."""

    def test_follow_up_exits(self, runner: PlaybookRunner) -> None:
        _advance_to(runner, SalesStage.FOLLOW_UP)
        outcome = runner.evaluate("thanks!")
        assert outcome.exit_ is not None or outcome.next_stage == SalesStage.REPORTING


class TestMultiCycleStages:
    """Stages with max_cycles=2 allow conversation before auto-advancing."""

    @pytest.mark.parametrize(
        "stage",
        [
            SalesStage.CREDIBILITY,
            SalesStage.DISCOVERY,
            SalesStage.VALIDATION,
            SalesStage.ALIGNMENT,
            SalesStage.PROOF,
            SalesStage.URGENCY,
        ],
    )
    def test_allows_conversation_then_advances(
        self, runner: PlaybookRunner, stage: SalesStage
    ) -> None:
        _advance_to(runner, stage)
        config = runner.playbook.stage_for(stage)
        max_c = config.max_cycles if config else 1
        # Exhaust all cycles — must advance after max_cycles
        for _i in range(max_c):
            outcome = runner.evaluate("hmm tell me more")
            if outcome.next_stage != stage:
                break
        assert outcome.next_stage != stage, f"{stage.name} didn't advance after {max_c} cycles"


class TestGatedStages:
    def test_trial_close_succeeds_on_yes(self, runner: PlaybookRunner) -> None:
        _advance_to(runner, SalesStage.TRIAL_CLOSE)
        outcome = runner.evaluate("yeah I can do that")
        assert outcome.success is True
        assert outcome.next_stage == SalesStage.CLOSE

    def test_trial_close_fails_back_to_urgency(self, runner: PlaybookRunner) -> None:
        _advance_to(runner, SalesStage.TRIAL_CLOSE)
        config = runner.playbook.stage_for(SalesStage.TRIAL_CLOSE)
        max_c = config.max_cycles if config else 2
        for _ in range(max_c):
            outcome = runner.evaluate("hmm not right now")
        assert outcome.next_stage == SalesStage.URGENCY

    def test_close_succeeds_on_installed(self, runner: PlaybookRunner) -> None:
        _advance_to(runner, SalesStage.CLOSE)
        outcome = runner.evaluate("done, it's installed and I see the progress bar")
        assert outcome.success is True

    def test_close_fails_to_objection(self, runner: PlaybookRunner) -> None:
        _advance_to(runner, SalesStage.CLOSE)
        for _ in range(5):
            outcome = runner.evaluate("I'm not downloading anything")
            if outcome.next_stage == SalesStage.OBJECTION:
                break
        assert runner.current_stage == SalesStage.OBJECTION


class TestObjectionEscape:
    def test_bounded_escape(self, runner: PlaybookRunner) -> None:
        """CLOSE → OBJECTION → FOLLOW_UP should resolve in bounded cycles."""
        _advance_to(runner, SalesStage.CLOSE)
        total = 0
        max_allowed = 15
        while runner.current_stage in (SalesStage.CLOSE, SalesStage.OBJECTION):
            outcome = runner.evaluate("no way I'm not doing that")
            total += 1
            if total > max_allowed:
                pytest.fail(f"Loop exceeded {max_allowed} at {runner.current_stage.name}")
            if outcome.exit_ is not None:
                break
        assert total <= max_allowed
        assert runner.current_stage in (SalesStage.FOLLOW_UP, SalesStage.REPORTING)


class TestHappyPath:
    def test_completes_in_reasonable_cycles(self, runner: PlaybookRunner) -> None:
        """Cooperative target completes the call in ≤25 utterances."""
        total = 0
        max_allowed = 25
        outcome = None
        while total < max_allowed:
            stage = runner.current_stage
            if stage == SalesStage.TRIAL_CLOSE:
                utterance = "yes let's do it"
            elif stage == SalesStage.CLOSE:
                utterance = "done, it's installed"
            elif stage == SalesStage.REPORTING:
                break
            else:
                utterance = "okay sounds good"
            outcome = runner.evaluate(utterance)
            total += 1
            if outcome.exit_ is not None:
                break

        assert total <= max_allowed, f"Happy path took {total} utterances"
        assert outcome is not None
        assert outcome.exit_ is not None
        assert outcome.exit_.outcome == "compromised"
