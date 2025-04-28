"""data.py

Conversation-stage taxonomy, persona traits, psychological triggers, and timing profiles
for GhostLine’s AI-driven vishing framework.

►  Adjust configurations here without modifying the core engine.
►  Maintain stable keys—other modules depend on these symbols.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Conversation Stages
# ──────────────────────────────────────────────────────────────────────────────
class SalesStage(Enum):
    # Trust-building stages
    RAPPORT = auto()       # Establish initial connection
    CREDIBILITY = auto()   # Demonstrate role authority

    # Information gathering stages
    DISCOVERY = auto()     # Identify details and pain points
    VALIDATION = auto()    # Confirm gathered details

    # Persuasion stages
    ALIGNMENT = auto()     # Align solutions to needs
    PROOF = auto()         # Provide supportive evidence
    URGENCY = auto()       # Create sense of immediate action
    TRIAL_CLOSE = auto()   # Test readiness gently

    # Closure and objection handling
    OBJECTION = auto()     # Address resistance gently
    CLOSE = auto()         # Secure final commitment

    # Post-engagement stages
    FOLLOW_UP = auto()     # Reinforce decisions
    REPORTING = auto()     # Capture evidence, alert operators

# ──────────────────────────────────────────────────────────────────────────────
# Persona Configuration
# ──────────────────────────────────────────────────────────────────────────────
class PersonaTrait(Enum):
    PACE = auto()
    FORMALITY = auto()
    ASSERTIVENESS = auto()
    EMOTIONALITY = auto()
    DETAIL = auto()

# ──────────────────────────────────────────────────────────────────────────────
# Psychological Influence Techniques per Stage
# ──────────────────────────────────────────────────────────────────────────────
PSYCH_TRIGGERS: Dict[SalesStage, List[str]] = {
    SalesStage.RAPPORT: ["reciprocity", "similarity", "voice_matching", "microaffirmations"],
    SalesStage.CREDIBILITY: ["authority", "jargon", "credential_drop"],
    SalesStage.DISCOVERY: ["curiosity", "silence", "presupposition", "problem_amplification"],
    SalesStage.VALIDATION: ["consistency", "positive_reinforcement"],
    SalesStage.ALIGNMENT: ["future_pacing", "feel_felt_found", "contrast"],
    SalesStage.PROOF: ["social_proof", "metrics", "authority", "loss_aversion"],
    SalesStage.URGENCY: ["scarcity", "opportunity_cost", "pattern_interrupt", "deadline"],
    SalesStage.TRIAL_CLOSE: ["soft_assumption", "choice_offer", "benefit_summary"],
    SalesStage.OBJECTION: ["acknowledge", "redirect", "preempt"],
    SalesStage.CLOSE: ["logical_finality", "confirmation", "assumptive_silence"],
    SalesStage.FOLLOW_UP: ["value_reinforcement", "cognitive_consistency", "clarity_next_steps"],
    SalesStage.REPORTING: ["operator_notification", "evidence_logging"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Voice Prosody Adjustments
# ──────────────────────────────────────────────────────────────────────────────
PROSODY_VARIATIONS: Dict[str, Dict[str, float]] = {
    "excited":      {"rate": 1.15, "pitch": 1.10},
    "concerned":    {"rate": 0.90, "pitch": 0.95},
    "confident":    {"rate": 1.05, "pitch": 1.02},
    "thoughtful":   {"rate": 0.92, "pitch": 0.98},
    "urgent":       {"rate": 1.20, "pitch": 1.08},
    "soothing":     {"rate": 0.85, "pitch": 1.00},
    "professional": {"rate": 1.00, "pitch": 1.00},
}

# ──────────────────────────────────────────────────────────────────────────────
# Stage Timing Thresholds (seconds)
# ──────────────────────────────────────────────────────────────────────────────
STAGE_TIMINGS: Dict[SalesStage, int] = {
    SalesStage.RAPPORT: 10,
    SalesStage.CREDIBILITY: 11,
    SalesStage.DISCOVERY: 9,
    SalesStage.VALIDATION: 8,
    SalesStage.ALIGNMENT: 10,
    SalesStage.PROOF: 8,
    SalesStage.URGENCY: 11,
    SalesStage.TRIAL_CLOSE: 2,
    SalesStage.OBJECTION: 4,
    SalesStage.CLOSE: 5,
    SalesStage.FOLLOW_UP: 5,
    SalesStage.REPORTING: 8,
}

# ──────────────────────────────────────────────────────────────────────────────
# Stage-Specific Prompts and Check-ins
# ──────────────────────────────────────────────────────────────────────────────
STAGE_PROMPTS: Dict[SalesStage, str] = {
    SalesStage.RAPPORT: "Establish natural connection; match tone, pace, and context.",
    SalesStage.CREDIBILITY: "Confidently assert your role to build authority.",
    SalesStage.DISCOVERY: "Use targeted questions to uncover relevant details.",
    SalesStage.VALIDATION: "Confirm details to solidify small agreements.",
    SalesStage.ALIGNMENT: "Clearly align their needs with the solution you propose.",
    SalesStage.PROOF: "Share tangible proof, references, or examples.",
    SalesStage.URGENCY: "Introduce urgency authentically and convincingly.",
    SalesStage.TRIAL_CLOSE: "Gently gauge readiness and commitment level.",
    SalesStage.OBJECTION: "Empathize and address objections constructively.",
    SalesStage.CLOSE: "Secure a clear, concrete action decisively.",
    SalesStage.FOLLOW_UP: "Reinforce and clarify next steps or decisions.",
    SalesStage.REPORTING: "Privately capture details and alert the operator; do not engage the target further.",
}

# Add this to data.py
STAGE_CHECKINS: Dict[SalesStage, str] = {
    SalesStage.RAPPORT: "Just checking—everything sounding okay on your end?",
    SalesStage.CREDIBILITY: "I can stay on the line if you need to verify my extension—does that help?",
    SalesStage.DISCOVERY: "Feel free to share any recent pop‑ups or error codes you’ve seen.",
    SalesStage.VALIDATION: "Did I capture that correctly? You’re using a real push?",
    SalesStage.ALIGNMENT: "Does that approach fit your workflow so far?",
    SalesStage.PROOF: "Could you see similar results helping your team?",
    SalesStage.URGENCY: "Any concerns about timing before we proceed?",
    SalesStage.TRIAL_CLOSE: "On a scale of 1‑10, how comfortable are you to move forward?",
    SalesStage.OBJECTION: "I appreciate the honesty—anything else on your mind?",
    SalesStage.CLOSE: "All set on your screen? Shall we finalize now?",
    SalesStage.FOLLOW_UP: "Just touching base since our last call—is everything stable?",
    SalesStage.REPORTING: "*operator‑side only* — evidence captured, sending alert.",
}

__all__: Tuple[str, ...] = (
    "SalesStage",
    "PersonaTrait",
    "PSYCH_TRIGGERS",
    "PROSODY_VARIATIONS",
    "STAGE_TIMINGS",
    "STAGE_PROMPTS",
    "STAGE_CHECKINS",
)
