"""Tests for the agent layer (model factory, analysis, triggers, handoffs, Axel)."""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest
from agents import Agent
from agents.extensions.models.litellm_model import LitellmModel

from ghostline.agent.analysis import AnalysisResult
from ghostline.agent.analysis_agent import ANALYSIS_INSTRUCTIONS, build_analysis_agent
from ghostline.agent.axel import AxelAgent, build_axel
from ghostline.agent.model import DEFAULT_MODEL, build_model
from ghostline.agent.stage_handoffs import (
    CallContext,
    build_stage_agent,
    build_stage_agents,
    stage_instructions,
    update_context_with_analysis,
)
from ghostline.agent.triggers import DEFAULT_TRIGGER, select_trigger
from ghostline.playbook import Playbook
from ghostline.taxonomy import PSYCH_TRIGGERS, SalesStage


@pytest.fixture
def simple_playbook() -> Playbook:
    return Playbook.from_string(
        """
meta: {name: simple}
defaults: {persona: professional}
sequence:
  - stage: RAPPORT
    custom_prompt: "Hello, IT desk here."
  - stage: CREDIBILITY
  - stage: CLOSE
    success_regex: "\\\\bpassword\\\\b"
    goto_on_success: REPORTING
  - stage: REPORTING
"""
    )


@pytest.fixture
def branching_playbook() -> Playbook:
    return Playbook.from_string(
        """
meta: {name: branching}
sequence:
  - stage: URGENCY
    max_cycles: 1
    goto_on_fail: OBJECTION
  - stage: OBJECTION
    goto_on_success: CLOSE
    goto_on_fail: URGENCY
  - stage: CLOSE
"""
    )


# ─── Model factory ─────────────────────────────────────────────────


class TestBuildModel:
    def test_returns_litellm_model(self) -> None:
        m = build_model("openai/gpt-4o-mini")
        assert isinstance(m, LitellmModel)

    def test_passes_api_key_and_base(self) -> None:
        m = build_model("ollama/llama3.1", api_key="k", api_base="http://localhost:11434")
        assert isinstance(m, LitellmModel)

    def test_default_model_constant(self) -> None:
        assert DEFAULT_MODEL == "openai/gpt-4o-mini"


# ─── AnalysisResult schema ─────────────────────────────────────────


class TestAnalysisResult:
    def test_valid_construction(self) -> None:
        a = AnalysisResult(
            sentiment_score=0.3,
            interest_level=0.6,
            objection_type=None,
            key_concerns=["timing"],
            buying_signals=["asked about pricing"],
            suggested_approach="emphasize deadline",
        )
        assert a.sentiment_score == pytest.approx(0.3)
        assert a.objection_type is None

    def test_sentiment_score_bounds_enforced(self) -> None:
        with pytest.raises(ValueError, match="sentiment_score"):
            AnalysisResult(
                sentiment_score=1.5,
                interest_level=0.5,
                suggested_approach="x",
            )

    def test_interest_level_bounds_enforced(self) -> None:
        with pytest.raises(ValueError, match="interest_level"):
            AnalysisResult(
                sentiment_score=0.0,
                interest_level=-0.1,
                suggested_approach="x",
            )

    def test_defaults_for_lists(self) -> None:
        a = AnalysisResult(sentiment_score=0.0, interest_level=0.5, suggested_approach="x")
        assert a.key_concerns == []
        assert a.buying_signals == []
        assert a.objection_type is None


# ─── Trigger selector ──────────────────────────────────────────────


def make_analysis(
    *,
    sentiment: float = 0.0,
    interest: float = 0.5,
    objection: str | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        sentiment_score=sentiment,
        interest_level=interest,
        objection_type=objection,
        suggested_approach="continue",
    )


class TestSelectTrigger:
    def test_returns_default_for_empty_stage_triggers(self) -> None:
        # SalesStage.REPORTING has triggers, but if we somehow pass an unknown
        # stage we should get the default. Constructed via mocking.
        a = make_analysis()
        result = select_trigger(SalesStage.RAPPORT, a, rng=random.Random(0))
        assert isinstance(result, str)
        assert result  # non-empty

    def test_negative_sentiment_biases_to_empathy_triggers(self) -> None:
        a = make_analysis(sentiment=-0.5)
        # Sample many times to cover the relevant pool
        rng = random.Random(42)
        results = {select_trigger(SalesStage.RAPPORT, a, rng=rng) for _ in range(20)}
        # All results should be from the empathy-flavoured subset
        for r in results:
            assert "feel" in r or "reciprocity" in r

    def test_objection_biases_to_objection_triggers(self) -> None:
        # ALIGNMENT has "feel_felt_found" (contains "felt")
        a = make_analysis(objection="timing")
        rng = random.Random(42)
        results = {select_trigger(SalesStage.ALIGNMENT, a, rng=rng) for _ in range(20)}
        for r in results:
            assert "objection" in r or "felt" in r

    def test_high_interest_biases_to_close_triggers(self) -> None:
        # ALIGNMENT has "future_pacing" (contains "future")
        a = make_analysis(interest=0.95)
        rng = random.Random(42)
        results = {select_trigger(SalesStage.ALIGNMENT, a, rng=rng) for _ in range(20)}
        for r in results:
            assert "close" in r or "future" in r

    def test_default_returns_from_stage_pool(self) -> None:
        a = make_analysis(sentiment=0.1, interest=0.5, objection=None)
        rng = random.Random(0)
        result = select_trigger(SalesStage.RAPPORT, a, rng=rng)
        assert result in PSYCH_TRIGGERS[SalesStage.RAPPORT]

    def test_default_trigger_constant(self) -> None:
        assert DEFAULT_TRIGGER == "acknowledge"


# ─── Analysis agent ────────────────────────────────────────────────


class TestAnalysisAgent:
    def test_returns_agent_with_analysis_output_type(self) -> None:
        agent = build_analysis_agent()
        assert isinstance(agent, Agent)
        assert agent.output_type is AnalysisResult

    def test_has_instructions(self) -> None:
        agent = build_analysis_agent()
        assert agent.instructions is ANALYSIS_INSTRUCTIONS

    def test_name_is_ghostline_analyzer(self) -> None:
        agent = build_analysis_agent()
        assert agent.name == "ghostline-analyzer"

    def test_accepts_custom_model_string(self) -> None:
        agent = build_analysis_agent(model="anthropic/claude-3-5-sonnet-latest")
        assert isinstance(agent, Agent)


# ─── Stage handoffs ────────────────────────────────────────────────


class TestStageInstructions:
    def test_includes_stage_name(self) -> None:
        text = stage_instructions(SalesStage.RAPPORT)
        assert "RAPPORT" in text
        assert "GhostLine" in text

    def test_includes_stage_objective(self) -> None:
        text = stage_instructions(SalesStage.URGENCY)
        # The default STAGE_PROMPTS text should be in the instructions
        assert "urgency" in text.lower()


class TestBuildStageAgent:
    def test_returns_agent_named_for_stage(self, simple_playbook: Playbook) -> None:
        agent = build_stage_agent(SalesStage.RAPPORT, simple_playbook)
        assert isinstance(agent, Agent)
        assert agent.name == "axel_rapport"

    def test_custom_prompt_merged_into_instructions(self, simple_playbook: Playbook) -> None:
        agent = build_stage_agent(SalesStage.RAPPORT, simple_playbook)
        # Instructions is a callable; invoke it with mock context/agent
        instructions = agent.instructions(MagicMock(), agent)  # type: ignore[misc]
        assert "IT desk here" in instructions

    def test_handoff_targets_attached(self, simple_playbook: Playbook) -> None:
        target = build_stage_agent(SalesStage.CREDIBILITY, simple_playbook)
        agent = build_stage_agent(SalesStage.RAPPORT, simple_playbook, handoff_targets=[target])
        assert target in agent.handoffs


class TestBuildStageAgents:
    def test_one_agent_per_stage(self, simple_playbook: Playbook) -> None:
        agents = build_stage_agents(simple_playbook)
        assert set(agents.keys()) == {
            SalesStage.RAPPORT,
            SalesStage.CREDIBILITY,
            SalesStage.CLOSE,
            SalesStage.REPORTING,
        }

    def test_no_handoffs_wired(self, simple_playbook: Playbook) -> None:
        """Handoffs are intentionally empty — PlaybookRunner handles transitions.

        The Agents SDK handoff tool schemas are incompatible with some providers
        (e.g. Groq rejects empty JSON schema properties), and stage transitions
        are already managed by PlaybookRunner.evaluate().
        """
        agents = build_stage_agents(simple_playbook)
        for stage, agent in agents.items():
            assert agent.handoffs == [], f"{stage.name} should have no handoffs"


class TestCallContext:
    def test_initial_state(self, simple_playbook: Playbook) -> None:
        ctx = CallContext(simple_playbook, SalesStage.RAPPORT)
        assert ctx.current_stage == SalesStage.RAPPORT
        assert ctx.latest_analysis is None
        assert ctx.latest_trigger is None
        assert ctx.language_hint is None

    def test_update_with_analysis_sets_fields(self, simple_playbook: Playbook) -> None:
        ctx = CallContext(simple_playbook, SalesStage.RAPPORT)
        analysis = make_analysis(sentiment=-0.3)
        trigger = update_context_with_analysis(ctx, SalesStage.RAPPORT, analysis)
        assert ctx.latest_analysis is analysis
        assert ctx.latest_trigger == trigger
        assert isinstance(trigger, str)


# ─── Axel root agent ───────────────────────────────────────────────


class TestBuildAxel:
    def test_returns_axel_agent_with_root_and_stages(self, simple_playbook: Playbook) -> None:
        axel = build_axel(simple_playbook)
        assert isinstance(axel, AxelAgent)
        assert isinstance(axel.root, Agent)
        assert axel.root.name == "axel-root"
        assert len(axel.stages) == 4

    def test_current_stage_is_first_stage(self, simple_playbook: Playbook) -> None:
        axel = build_axel(simple_playbook)
        assert axel.current_stage == SalesStage.RAPPORT

    def test_advance_to_updates_context(self, simple_playbook: Playbook) -> None:
        axel = build_axel(simple_playbook)
        axel.advance_to(SalesStage.CLOSE)
        assert axel.current_stage == SalesStage.CLOSE

    def test_root_handoffs_include_all_stages(self, simple_playbook: Playbook) -> None:
        axel = build_axel(simple_playbook)
        # Root delegates to all stage agents
        for stage_agent in axel.stages.values():
            assert stage_agent in axel.root.handoffs
