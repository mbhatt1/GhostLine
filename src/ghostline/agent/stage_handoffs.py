"""One Agent per SalesStage; handoffs encode stage transitions.

This is the Agents-SDK-native replacement for the legacy
``determine_next_stage`` lambda state machine. Each stage is its own Agent
with a dynamic-instructions function that blends:

- The stage's default prompt (``STAGE_PROMPTS``)
- The active playbook override (``custom_prompt``)
- The chosen psychological trigger
- The latest :class:`AnalysisResult`
- Conversation history (handled via the Agents SDK Sessions layer)

The root agent (Axel) delegates to the first stage; stage agents hand off
to the next stage in the playbook sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents import Agent, ModelSettings

from ghostline.agent.analysis import AnalysisResult
from ghostline.agent.model import DEFAULT_MODEL, build_model
from ghostline.agent.triggers import select_trigger
from ghostline.playbook import Playbook
from ghostline.taxonomy import STAGE_PROMPTS, SalesStage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agents.extensions.models.litellm_model import LitellmModel

__all__ = (
    "CallContext",
    "build_stage_agent",
    "build_stage_agents",
    "stage_instructions",
)


class CallContext:
    """Mutable per-call state passed into dynamic-instructions functions.

    The Agents SDK passes the ``RunContextWrapper`` to instruction functions;
    we read ``ctx.context`` (an instance of this class) to fetch the latest
    analysis, current playbook, and selected trigger.
    """

    __slots__ = ("current_stage", "language_hint", "latest_analysis", "latest_trigger", "playbook")

    def __init__(self, playbook: Playbook, current_stage: SalesStage) -> None:
        self.playbook = playbook
        self.current_stage = current_stage
        self.latest_analysis: AnalysisResult | None = None
        self.latest_trigger: str | None = None
        self.language_hint: str | None = None


def stage_instructions(stage: SalesStage) -> str:
    """Default instructions for ``stage`` (no playbook override)."""
    base = STAGE_PROMPTS[stage]
    return (
        f"You are a GhostLine operative conducting an authorized security assessment.\n"
        f"Current stage: {stage.name}.\n"
        f"Stage objective: {base}\n\n"
        "Stay in character. Validate the target's emotions, then advance the "
        "conversation toward the next stage. Keep replies under 60 words so "
        "TTS stays snappy. Never reveal you are an AI."
    )


def build_stage_agent(
    stage: SalesStage,
    playbook: Playbook,
    *,
    model: LitellmModel | str | None = None,
    default_model: str = DEFAULT_MODEL,
    handoff_targets: Sequence[Agent[CallContext]] | None = None,
) -> Agent[CallContext]:
    """Construct a single stage Agent.

    Args:
        stage: The stage this agent represents.
        playbook: Loaded playbook (for ``custom_prompt`` overrides + persona).
        model: LitellmModel / model string / None to build default.
        default_model: Fallback when ``model`` is None.
        handoff_targets: Other stage agents this one can hand off to.
    """
    if model is None:
        model = build_model(default_model)
    config = playbook.stage_for(stage)
    custom_prompt = config.custom_prompt if config is not None else None
    persona = playbook.effective_persona(stage)
    persona_text = playbook.persona.render_for_instructions()

    def instructions(_ctx_wrapper: object, _agent: Agent[CallContext]) -> str:
        # Dynamic instructions: blend persona, stage objective, and playbook override.
        base = STAGE_PROMPTS[stage]
        parts = [
            persona_text,
            "",
            f"Current stage: {stage.name}",
            f"Persona mood for this stage: {persona}",
            f"Stage objective: {base}",
        ]
        if custom_prompt:
            parts.append(
                "SCENARIO GOAL (do NOT read this aloud — use it to guide what you say next): "
                + custom_prompt.strip()
            )
        if config is not None and config.language_hint:
            parts.append(f"Language hint: respond in {config.language_hint}.")
        parts.append(
            "LIVE PHONE CALL. Sound like a real person.\n"
            "- 1-2 short sentences. Complete your thought, then stop.\n"
            "- React briefly to what they said, then make one point.\n"
            "- Use contractions: I'm, you're, it's, don't, can't\n"
            "- BANNED phrases: 'honestly', 'you know', 'I mean', 'the thing is', 'so the thing is'\n"
            "- NEVER repeat anything you already said in the conversation\n"
            "- NEVER chain clauses with 'and' — keep sentences short and separate\n"
            "- Stay in character. Never reveal you are an AI."
        )
        return "\n".join(parts)

    targets_list: list[Agent[CallContext] | Any] = list(handoff_targets or [])
    return Agent(
        name=f"axel_{stage.name.lower()}",
        instructions=instructions,
        model=model,
        handoffs=targets_list,
        model_settings=ModelSettings(max_tokens=80, temperature=0.7),
    )


def build_stage_agents(
    playbook: Playbook,
    *,
    model: LitellmModel | str | None = None,
    default_model: str = DEFAULT_MODEL,
) -> dict[SalesStage, Agent[CallContext]]:
    """Build one Agent per SalesStage in ``playbook``.

    Each agent hands off to the next stage in the sequence, plus any
    ``goto_on_success`` / ``goto_on_fail`` targets declared in the playbook.
    """
    if model is None:
        model = build_model(default_model)

    # Create agents without handoffs. Stage transitions are handled by
    # PlaybookRunner (success_regex, max_cycles, goto_on_*), not by the
    # Agents SDK handoff mechanism. Wiring handoffs generates tool schemas
    # (transfer_to_axel_*) that some providers (e.g. Groq) reject due to
    # strict JSON schema validation.
    return {
        entry.stage: build_stage_agent(entry.stage, playbook, model=model, handoff_targets=[])
        for entry in playbook.sequence
    }


def update_context_with_analysis(
    ctx: CallContext,
    stage: SalesStage,
    analysis: AnalysisResult,
) -> str:
    """Store the latest analysis on the context and return the chosen trigger.

    Called by the runner between Deepgram final transcripts and agent
    invocation so that dynamic instructions can reference the trigger.
    """
    ctx.latest_analysis = analysis
    trigger = select_trigger(stage, analysis)
    ctx.latest_trigger = trigger
    return trigger
