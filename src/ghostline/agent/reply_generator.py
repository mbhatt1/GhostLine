"""Reply generator: orchestrates analysis + stage agent to produce a reply.

This is the bridge between the call handler (which receives utterances)
and the agent layer (which produces replies).

By default it runs a **single LLM call** per utterance (the stage agent)
for low latency. The analysis agent (sentiment/objection detection) is
optional and can be enabled via ``skip_analysis=False`` — but it adds
a second LLM round-trip which roughly doubles response time.

Failures degrade gracefully: on LLM error, a fallback reply is returned
so the call doesn't dead-air.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agents import Runner

from ghostline.agent.analysis import AnalysisResult
from ghostline.agent.analysis_agent import build_analysis_agent
from ghostline.agent.axel import AxelAgent, build_axel
from ghostline.agent.stage_handoffs import CallContext, update_context_with_analysis
from ghostline.agent.triggers import select_trigger
from ghostline.playbook import Playbook
from ghostline.taxonomy import PSYCH_TRIGGERS, SalesStage

if TYPE_CHECKING:
    from agents.extensions.models.litellm_model import LitellmModel

__all__ = ("ReplyGenerator", "ReplyResult")

_LOGGER = logging.getLogger(__name__)

_FALLBACK_REPLY = "I understand. Could you tell me more about that?"

# Default AnalysisResult used when the analysis agent is skipped.
# Neutral sentiment, moderate interest, no objection — conservative.
_SKIP_ANALYSIS_DEFAULT = AnalysisResult(
    sentiment_score=0.0,
    interest_level=0.5,
    objection_type=None,
    key_concerns=[],
    buying_signals=[],
    suggested_approach="Continue with current approach",
)


class ReplyResult:
    """The result of generating a reply for one utterance.

    Attributes:
        reply: The text to speak via TTS.
        analysis: The structured analysis (or None if analysis failed).
        trigger: The psychological trigger selected (or None).
        stage: The stage the reply was generated for.
    """

    __slots__ = ("analysis", "reply", "stage", "trigger")

    def __init__(
        self,
        reply: str,
        analysis: AnalysisResult | None,
        trigger: str | None,
        stage: SalesStage,
    ) -> None:
        self.reply = reply
        self.analysis = analysis
        self.trigger = trigger
        self.stage = stage


class ReplyGenerator:
    """Wraps the Axel + analysis agents for use by the call handler.

    Args:
        playbook: The loaded playbook (for stage agents + persona).
        model: LiteLLM model instance/string.
        skip_analysis: If True (default), skip the analysis LLM call and
            use a default AnalysisResult. This halves response latency at
            the cost of less precise trigger selection. Set False for
            richer sentiment-aware trigger selection.
    """

    def __init__(
        self,
        playbook: Playbook | None,
        model: LitellmModel | str | None = None,
        *,
        skip_analysis: bool = True,
    ) -> None:
        self._playbook = playbook
        self._model = model
        self._skip_analysis = skip_analysis
        self._analysis_agent = build_analysis_agent(model=model) if not skip_analysis else None
        self._axel: AxelAgent | None = None
        if playbook is not None:
            self._axel = build_axel(playbook, model=model)

    def new_context(self) -> CallContext | None:
        """Create a fresh per-call context (or None if no playbook)."""
        if self._axel is not None:
            return self._axel.context
        return None

    async def generate(
        self,
        utterance: str,
        current_stage: SalesStage,
        context: CallContext | None = None,
        history: str | None = None,
    ) -> ReplyResult:
        """Generate a reply for ``utterance`` in ``current_stage``.

        Args:
            utterance: The user's latest utterance (transcribed by Deepgram).
            current_stage: The current conversation stage.
            context: The per-call CallContext (for dynamic instructions).
            history: Optional conversation history string for context.

        Returns:
            A :class:`ReplyResult` with the reply text and metadata.
        """
        if self._skip_analysis:
            analysis: AnalysisResult | None = _SKIP_ANALYSIS_DEFAULT
            trigger: str | None = self._pick_random_trigger(current_stage)
        else:
            analysis = await self._run_analysis(utterance)
            trigger = self._select_trigger(current_stage, analysis)

        if context is not None and analysis is not None:
            update_context_with_analysis(context, current_stage, analysis)

        reply = await self._run_stage_agent(utterance, current_stage, context, history)

        return ReplyResult(
            reply=reply,
            analysis=analysis,
            trigger=trigger,
            stage=current_stage,
        )

    @staticmethod
    def _pick_random_trigger(stage: SalesStage) -> str:
        """Pick a trigger from the stage's list without an LLM analysis call."""
        import secrets

        triggers = PSYCH_TRIGGERS.get(stage, [])
        if not triggers:
            return "acknowledge"
        return triggers[secrets.randbelow(len(triggers))]

    async def _run_analysis(self, utterance: str) -> AnalysisResult | None:
        """Run the analysis agent to get structured AnalysisResult."""
        if self._analysis_agent is None:
            return None
        try:
            result = await Runner.run(self._analysis_agent, utterance)
            return result.final_output_as(AnalysisResult)
        except Exception:
            _LOGGER.exception("Analysis agent failed; continuing without analysis")
            return None

    @staticmethod
    def _select_trigger(stage: SalesStage, analysis: AnalysisResult | None) -> str | None:
        """Pick a psychological trigger (or None if no analysis)."""
        if analysis is None:
            return None
        return select_trigger(stage, analysis)

    async def _run_stage_agent(
        self,
        utterance: str,
        stage: SalesStage,
        context: CallContext | None,
        history: str | None,
    ) -> str:
        """Run the stage agent to generate a reply."""
        if self._axel is None:
            return _FALLBACK_REPLY

        stage_agent = self._axel.stages.get(stage)
        if stage_agent is None:
            stage_agent = self._axel.root

        input_text = utterance
        if history:
            input_text = f'Conversation so far:\n{history}\n\nThe person just said: "{utterance}"'

        try:
            result = await Runner.run(stage_agent, input_text, context=context)
            final = result.final_output
            if isinstance(final, str) and final.strip():
                return final.strip()
            _LOGGER.warning("Stage agent returned empty output; using fallback")
            return _FALLBACK_REPLY
        except Exception:
            _LOGGER.exception("Stage agent failed; using fallback reply")
            return _FALLBACK_REPLY
