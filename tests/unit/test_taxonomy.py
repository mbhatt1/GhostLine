"""Tests for the conversation-stage taxonomy."""

from __future__ import annotations

import pytest

from ghostline.taxonomy import (
    PROSODY_VARIATIONS,
    PSYCH_TRIGGERS,
    STAGE_CHECKINS,
    STAGE_PROMPTS,
    STAGE_TIMINGS,
    PersonaTrait,
    SalesStage,
)

STAGE_MEMBERS = {
    "RAPPORT",
    "CREDIBILITY",
    "DISCOVERY",
    "VALIDATION",
    "ALIGNMENT",
    "PROOF",
    "URGENCY",
    "TRIAL_CLOSE",
    "OBJECTION",
    "CLOSE",
    "FOLLOW_UP",
    "REPORTING",
}


class TestSalesStage:
    def test_has_twelve_stages(self) -> None:
        assert len(SalesStage) == 12

    def test_member_names_match_expected(self) -> None:
        assert {s.name for s in SalesStage} == STAGE_MEMBERS

    def test_canonical_progression_order(self) -> None:
        """int(stage) reflects rapport → reporting default progression."""
        names = [s.name for s in SalesStage]
        assert names[0] == "RAPPORT"
        assert names[-1] == "REPORTING"
        assert names.index("CLOSE") < names.index("FOLLOW_UP")

    @pytest.mark.parametrize("stage", tuple(SalesStage))
    def test_each_stage_has_unique_value(self, stage: SalesStage) -> None:
        assert sum(1 for s in SalesStage if s.value == stage.value) == 1


class TestPersonaTrait:
    def test_traits_exist(self) -> None:
        assert {t.name for t in PersonaTrait} == {
            "PACE",
            "FORMALITY",
            "ASSERTIVENESS",
            "EMOTIONALITY",
            "DETAIL",
        }


class TestPsychTriggers:
    @pytest.mark.parametrize("stage", tuple(SalesStage))
    def test_every_stage_has_at_least_one_trigger(self, stage: SalesStage) -> None:
        triggers = PSYCH_TRIGGERS[stage]
        assert len(triggers) >= 1, f"{stage.name} has no triggers"
        assert all(isinstance(t, str) and t for t in triggers)

    def test_triggers_keyed_by_every_stage(self) -> None:
        assert set(PSYCH_TRIGGERS.keys()) == set(SalesStage)

    def test_known_trigger_vocab_preserved(self) -> None:
        """A few canonical Cialdini triggers must remain (regression guard)."""
        flat = {t for triggers in PSYCH_TRIGGERS.values() for t in triggers}
        for required in ("reciprocity", "authority", "scarcity", "social_proof"):
            assert required in flat, f"lost canonical trigger: {required}"


class TestProsodyVariations:
    def test_default_persona_present(self) -> None:
        assert "professional" in PROSODY_VARIATIONS

    @pytest.mark.parametrize("name", tuple(PROSODY_VARIATIONS))
    def test_each_prosody_has_rate_and_pitch(self, name: str) -> None:
        prosody = PROSODY_VARIATIONS[name]
        assert "rate" in prosody
        assert "pitch" in prosody
        assert 0.5 <= prosody["rate"] <= 2.0
        assert 0.5 <= prosody["pitch"] <= 2.0

    def test_professional_is_neutral_baseline(self) -> None:
        assert PROSODY_VARIATIONS["professional"] == {"rate": 1.0, "pitch": 1.0}


class TestStageTimings:
    @pytest.mark.parametrize("stage", tuple(SalesStage))
    def test_every_stage_has_positive_timing(self, stage: SalesStage) -> None:
        assert STAGE_TIMINGS[stage] > 0

    def test_reporting_has_reasonable_window(self) -> None:
        assert STAGE_TIMINGS[SalesStage.REPORTING] >= 5

    def test_trial_close_is_short(self) -> None:
        """TRIAL_CLOSE should be a short pause, not an awkward silence."""
        assert STAGE_TIMINGS[SalesStage.TRIAL_CLOSE] <= 8


class TestStagePromptsAndCheckins:
    @pytest.mark.parametrize("stage", tuple(SalesStage))
    def test_every_stage_has_prompt(self, stage: SalesStage) -> None:
        prompt = STAGE_PROMPTS[stage]
        assert isinstance(prompt, str)
        assert prompt.strip()

    @pytest.mark.parametrize("stage", tuple(SalesStage))
    def test_every_stage_has_checkin(self, stage: SalesStage) -> None:
        checkin = STAGE_CHECKINS[stage]
        assert isinstance(checkin, str)
        assert checkin.strip()

    def test_reporting_checkin_is_operator_side(self) -> None:
        assert "operator" in STAGE_CHECKINS[SalesStage.REPORTING].lower()
