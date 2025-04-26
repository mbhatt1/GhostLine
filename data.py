# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data
# ─────────────────────────────────────────────────────────────────────────────
from enum import Enum


class SalesStage(Enum):
    RAPPORT = 1      # Building initial connection
    DISCOVERY = 2    # Exploring needs/pain points
    ALIGNMENT = 3    # Positioning solution to needs
    PROOF = 4        # Providing evidence/testimonials
    URGENCY = 5      # Creating time sensitivity
    TRIAL_CLOSE = 6  # Testing readiness to commit
    OBJECTION = 7    # Handling concerns
    CLOSE = 8        # Securing commitment
    FOLLOW_UP = 9    # Next steps and additional value

class PersonaTrait(Enum):
    PACE = 1         
    FORMALITY = 2    
    ASSERTIVENESS = 3
    EMOTIONALITY = 4
    DETAIL = 5       

PSYCH_TRIGGERS = {
    SalesStage.RAPPORT: [
        "reciprocity_personal_share",
        "similarity_establish",
        "voice_matching",
    ],
    SalesStage.DISCOVERY: [
        "open_loop_curiosity",
        "strategic_silence",
        "presupposition_question",
    ],
    SalesStage.ALIGNMENT: [
        "future_pacing",
        "feel_felt_found",
        "contrast_framing",
    ],
    SalesStage.PROOF: [
        "social_proof_specific",
        "authority_invocation",
        "loss_aversion",
    ],
    SalesStage.URGENCY: [
        "scarcity_authentic",
        "opportunity_cost",
        "pattern_interrupt",
    ],
    SalesStage.TRIAL_CLOSE: [
        "assumptive_close_soft",
        "choice_close",
        "summary_benefit_bridge",
    ],
    SalesStage.OBJECTION: [
        "feel_felt_found",
        "objection_redirect",
        "objection_preemption",
    ],
    SalesStage.CLOSE: [
        "logical_conclusion",
        "future_confirmation",
        "assumptive_paperwork",
    ],
    SalesStage.FOLLOW_UP: [
        "value_addition",
        "cognitive_consistency",
        "next_step_clarity",
    ]
}

PROSODY_VARIATIONS = {
    "excited": {"rate": 1.15, "pitch": 1.1},
    "concerned": {"rate": 0.9, "pitch": 0.95},
    "confident": {"rate": 1.05, "pitch": 1.02},
    "thoughtful": {"rate": 0.92, "pitch": 0.98},
    "urgent": {"rate": 1.2, "pitch": 1.08},
}

# Stage timings
STAGE_TIMINGS = {
    SalesStage.RAPPORT: 15,
    SalesStage.DISCOVERY: 8,
    SalesStage.ALIGNMENT: 6,
    SalesStage.PROOF: 10,
    SalesStage.URGENCY: 4,
    SalesStage.TRIAL_CLOSE: 7,
    SalesStage.OBJECTION: 5,
    SalesStage.CLOSE: 3,
    SalesStage.FOLLOW_UP: 12
}

STAGE_PROMPTS = {
    SalesStage.RAPPORT: (
        "You're in the RAPPORT stage with the customer. Focus on building connection and trust. "
        "Ask open-ended questions. Listen more than you speak."
    ),
    SalesStage.DISCOVERY: (
        "You're in the DISCOVERY stage. Uncover the customer's pain points. "
        "Validate concerns. Ask targeted questions about their situation."
    ),
    SalesStage.ALIGNMENT: (
        "You're in the ALIGNMENT stage. Connect their needs to your solution. "
        "Ask confirming questions to ensure alignment."
    ),
    SalesStage.PROOF: (
        "You're in the PROOF stage. Share success stories or metrics. "
        "Build credibility with relevant testimonials."
    ),
    SalesStage.URGENCY: (
        "You're in the URGENCY stage. Introduce time-sensitive elements. "
        "Highlight the cost of delay or inaction."
    ),
    SalesStage.TRIAL_CLOSE: (
        "You're in the TRIAL CLOSE stage. Test readiness with soft commitment questions. "
        "Use assumptive language and watch for hesitation."
    ),
    SalesStage.OBJECTION: (
        "You're in the OBJECTION handling stage. Use feel-felt-found or redirect. "
        "Turn objections into reasons to buy."
    ),
    SalesStage.CLOSE: (
        "You're in the CLOSE stage. Be direct and confident. Ask clearly for the business. "
        "Stay silent after asking the closing question."
    ),
    SalesStage.FOLLOW_UP: (
        "You're in the FOLLOW UP stage. Confirm next steps. Provide unexpected value. "
        "Reinforce their decision or keep the door open."
    )
}

STAGE_CHECKINS = {
    SalesStage.RAPPORT: (
        "Just checking in—I'm interested in hearing more about your situation. "
        "How do you feel about what we've discussed so far?"
    ),
    SalesStage.DISCOVERY: (
        "I noticed some hesitation—would you like me to clarify anything "
        "about your current challenges or our approach?"
    ),
    SalesStage.ALIGNMENT: (
        "Based on everything we've discussed, does our solution resonate with "
        "your vision so far?"
    ),
    SalesStage.PROOF: (
        "Could you see yourself benefiting from similar results?"
    ),
    SalesStage.URGENCY: (
        "Any questions about the timing or availability we discussed?"
    ),
    SalesStage.TRIAL_CLOSE: (
        "How comfortable do you feel with moving forward at this point?"
    ),
    SalesStage.OBJECTION: (
        "Thanks for sharing your concerns. Is there anything else holding you back?"
    ),
    SalesStage.CLOSE: (
        "I believe we've covered the key points. Shall we finalize the details?"
    ),
    SalesStage.FOLLOW_UP: (
        "Hope everything is going well since our last conversation. "
        "Is there anything else I can do to support you?"
    )
}