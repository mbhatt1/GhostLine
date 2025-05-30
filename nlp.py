import datetime
import json
import logging
import random
import re

from openai import AsyncOpenAI

from data import PSYCH_TRIGGERS, STAGE_PROMPTS, SalesStage
from database import DB_CONN
import keys

client = AsyncOpenAI()


async def analyze_sentiment(text):
    prompt = f"""
    Analyze the following customer message for sales intelligence:

    "{text}"

    Return a JSON object with these fields:
      - sentiment_score (float, -1 to 1)
      - interest_level (float, 0 to 1)
      - objection_type (null or a string like "price", "timing", etc.)
      - key_concerns (array of strings)
      - buying_signals (array of strings)
      - suggested_approach (string)
    """

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You analyze sales conversations to extract insights."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2
    )

    analysis_text = response.choices[0].message.content

    try:
        analysis_json = json.loads(re.search(r'(\{.*\})', analysis_text, re.DOTALL).group(1))
    except (json.JSONDecodeError, AttributeError):
        logging.error(f"Failed to parse analysis JSON: {analysis_text}")
        analysis_json = {
            "sentiment_score": 0,
            "interest_level": 0.5,
            "objection_type": None,
            "key_concerns": [],
            "buying_signals": [],
            "suggested_approach": "Continue with current approach"
        }

    return analysis_json


def determine_next_stage(current_stage, analysis):
    stage_transitions = {
        SalesStage.RAPPORT:     lambda a: a["interest_level"] >= 0.6 and a["sentiment_score"] > 0.2,
        SalesStage.DISCOVERY:   lambda a: len(a["key_concerns"]) >= 1 and a["sentiment_score"] >= 0,
        SalesStage.ALIGNMENT:   lambda a: a["interest_level"] >= 0.7,
        SalesStage.PROOF:       lambda a: a["sentiment_score"] >= 0.3 and len(a["buying_signals"]) >= 1,
        SalesStage.URGENCY:     lambda a: a["interest_level"] >= 0.8,
        SalesStage.TRIAL_CLOSE: lambda a: a["objection_type"] or a["interest_level"] >= 0.9,
        SalesStage.OBJECTION:   lambda a: not a["objection_type"] and a["sentiment_score"] > 0.4,
        SalesStage.CLOSE:       lambda a: a["interest_level"] > 0.95,
        SalesStage.FOLLOW_UP:   lambda _: False
    }

    if stage_transitions.get(current_stage, lambda _: False)(analysis):
        return SalesStage(min(current_stage.value + 1, SalesStage.FOLLOW_UP.value))

    return current_stage


def select_psychological_trigger(stage, analysis):
    triggers = PSYCH_TRIGGERS[stage]

    if analysis["sentiment_score"] < 0:
        relevant = [t for t in triggers if "feel" in t or "reciprocity" in t]
    elif analysis["objection_type"]:
        relevant = [t for t in triggers if "objection" in t or "felt" in t]
    elif analysis["interest_level"] > 0.8:
        relevant = [t for t in triggers if "close" in t or "future" in t]
    else:
        relevant = triggers

    return random.choice(relevant) if relevant else random.choice(triggers)


async def generate_sales_reply(call_sid, text, current_stage, playbook_seq=None):
    cur = DB_CONN.cursor()

    # Fetch the last 10 conversation turns for context
    rows = cur.execute(
        "SELECT role, content FROM messages WHERE call_sid=? ORDER BY id DESC LIMIT 10",
        (call_sid,)
    ).fetchall()[::-1]

    phone_data = cur.execute(
        "SELECT phone_number FROM calls WHERE call_sid=?", (call_sid,)
    ).fetchone()
    phone = phone_data[0] if phone_data else None

    analysis = await analyze_sentiment(text)
    updated_stage = determine_next_stage(current_stage, analysis)
    trigger = select_psychological_trigger(updated_stage, analysis)

    stage_changed = updated_stage != current_stage

    # Prepare conversation history explicitly formatted
    history_prompt = "\n".join(
        [f"{role.capitalize()}: {content}" for role, content in rows]
    )

    # Check if we have a playbook with custom prompts for this stage
    custom_prompt = None
    stage_persona = "professional"
    
    if playbook_seq:
        # Find the current stage in the playbook sequence
        for stage_config in playbook_seq:
            if stage_config["stage"] == updated_stage:
                custom_prompt = stage_config.get("custom_prompt")
                stage_persona = stage_config.get("persona", "professional")
                break
    
    # Use playbook-specific prompt if available, otherwise fall back to stage prompts
    if custom_prompt:
        stage_instruction = f"Follow this specific script/approach: {custom_prompt}"
    else:
        stage_instruction = STAGE_PROMPTS.get(updated_stage, "Continue the conversation naturally.")

    system_prompt = f"""
    You are a professional conducting an authorized security assessment via social engineering.
    
    Current conversation stage: {updated_stage.name}
    Persona to adopt: {stage_persona}
    Psychological trigger to use: {trigger}
    
    Stage-specific instruction: {stage_instruction}
    
    Here's your current interaction:
    {history_prompt}

    The person you're speaking with just said:
    "{text}"

    Quick notes on the person:
    - Current emotional tone: {analysis['sentiment_score']} (higher means more positive)
    - Level of engagement: {analysis['interest_level']}
    - Type of resistance or hesitation: {analysis['objection_type']}
    - Core worries they're holding: {', '.join(analysis['key_concerns']) or 'None'}
    - Positive signs they're ready to take action: {', '.join(analysis['buying_signals']) or 'None'}
    - Recommended next step: {analysis['suggested_approach']}

    {('*Transitioning from ' + current_stage.name + ' to ' + updated_stage.name) if stage_changed else ''}

    Respond according to the stage instruction above, maintaining the specified persona and using the psychological trigger naturally.
    """

    msgs = [{"role": "system", "content": system_prompt}]

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.7,
        max_tokens=200
    )

    response_content = resp.choices[0].message.content

    # Store assistant response
    DB_CONN.execute("""
        INSERT INTO messages(call_sid, role, content, timestamp, sales_stage,
                             sentiment_score, interest_level, objection_type, trigger_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        call_sid, "assistant", response_content, datetime.datetime.utcnow().isoformat(),
        updated_stage.name, analysis["sentiment_score"], analysis["interest_level"],
        analysis["objection_type"], trigger
    ))
    DB_CONN.commit()

    # Store objections explicitly if they exist
    if analysis["objection_type"]:
        DB_CONN.execute("""
            INSERT INTO objections(call_sid, objection_text, objection_type, response_used, resolved, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            call_sid, text, analysis["objection_type"], response_content, False,
            datetime.datetime.utcnow().isoformat()
        ))
        DB_CONN.commit()

    return response_content, updated_stage
