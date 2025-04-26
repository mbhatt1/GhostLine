# ─────────────────────────────────────────────────────────────────────────────
# NLP, Sentiment, Stage Logic
# ─────────────────────────────────────────────────────────────────────────────
import datetime
import json
import logging
import random
import re

import openai

from data import PSYCH_TRIGGERS, STAGE_PROMPTS, SalesStage
from database import DB_CONN


async def analyze_sentiment(text):
    """
    Analyze text with OpenAI for sentiment, interest, objections, etc.
    Return a structured dictionary with relevant fields.
    """
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

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You analyze sales convos to extract insights."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2
    )

    analysis_text = response.choices[0].message.content
    try:
        json_match = re.search(r'(\{.*\})', analysis_text, re.DOTALL)
        if json_match:
            analysis_json = json.loads(json_match.group(1))
        else:
            analysis_json = json.loads(analysis_text)
        return analysis_json
    except json.JSONDecodeError:
        logging.error(f"Failed to parse analysis JSON: {analysis_text}")
        return {
            "sentiment_score": 0,
            "interest_level": 0.5,
            "objection_type": None,
            "key_concerns": [],
            "buying_signals": [],
            "suggested_approach": "Continue with current approach"
        }

def determine_next_stage(current_stage, analysis):
    stage_transitions = {
        SalesStage.RAPPORT:     lambda a: (a["interest_level"] >= 0.6 and a["sentiment_score"] > 0.2),
        SalesStage.DISCOVERY:   lambda a: (len(a["key_concerns"]) >= 1 and a["sentiment_score"] >= 0),
        SalesStage.ALIGNMENT:   lambda a: (a["interest_level"] >= 0.7),
        SalesStage.PROOF:       lambda a: (a["sentiment_score"] >= 0.3 and len(a["buying_signals"]) >= 1),
        SalesStage.URGENCY:     lambda a: (a["interest_level"] >= 0.8),
        SalesStage.TRIAL_CLOSE: lambda a: (a["objection_type"] is not None or a["interest_level"] >= 0.9),
        SalesStage.OBJECTION:   lambda a: (a["objection_type"] is None and a["sentiment_score"] > 0.4),
        SalesStage.CLOSE:       lambda a: (a["interest_level"] > 0.95),
        SalesStage.FOLLOW_UP:   lambda a: False
    }

    transition_condition = stage_transitions.get(current_stage, lambda _: False)
    if transition_condition(analysis):
        nv = min(current_stage.value + 1, SalesStage.FOLLOW_UP.value)
        return SalesStage(nv)
    return current_stage

def select_psychological_trigger(stage, analysis):
    triggers = PSYCH_TRIGGERS[stage]
    # Some example conditions
    if analysis["sentiment_score"] < 0:
        relevant = [t for t in triggers if "feel" in t or "reciprocity" in t]
    elif analysis["objection_type"]:
        relevant = [t for t in triggers if "objection" in t or "felt" in t]
    elif analysis["interest_level"] > 0.8:
        relevant = [t for t in triggers if "close" in t or "future" in t]
    else:
        relevant = triggers
    return random.choice(relevant) if relevant else random.choice(triggers)

# Generate the AI's next sales reply
async def generate_sales_reply(call_sid, text, current_stage):
    cur = DB_CONN.cursor()
    rows = cur.execute(
        "SELECT role, content FROM messages WHERE call_sid=? ORDER BY id DESC LIMIT 10",
        (call_sid,)
    ).fetchall()[::-1]

    # Get phone number from calls
    phone_data = cur.execute("SELECT phone_number FROM calls WHERE call_sid=?", (call_sid,)).fetchone()
    phone = phone_data[0] if phone_data else None

    # Possibly load a customer profile if needed
    # ...
    
    analysis = await analyze_sentiment(text)
    updated_stage = determine_next_stage(current_stage, analysis)
    stage_changed = (updated_stage != current_stage)
    trigger = select_psychological_trigger(updated_stage, analysis)

    system_prompt = f"""
    You are a family friend having a natural phone conversation with a potential customer of a cracked software.

    {STAGE_PROMPTS[updated_stage]}

    Use this psychological technique in your response: {trigger}

    Customer analysis:
    - Sentiment: {analysis['sentiment_score']}
    - Interest: {analysis['interest_level']}
    - Objection: {analysis['objection_type']}
    - Key concerns: {analysis['key_concerns']}
    - Buying signals: {analysis['buying_signals']}
    - Approach: {analysis['suggested_approach']}

    {'*IMPORTANT* Transitioning from ' + current_stage.name + ' to ' + updated_stage.name if stage_changed else ''}
    """

    msgs = [{"role": "system", "content": system_prompt}]
    for r, c in rows:
        msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": text})

    resp = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=msgs,
        temperature=0.7,
        max_tokens=200
    )
    response_content = resp.choices[0].message.content

    DB_CONN.execute("""
        INSERT INTO messages(call_sid, role, content, timestamp, sales_stage, 
        sentiment_score, interest_level, objection_type, trigger_used)
        VALUES(?,?,?,?,?,?,?,?,?)
    """, (
        call_sid, "assistant", response_content, datetime.datetime.utcnow().isoformat(),
        updated_stage.name,
        analysis["sentiment_score"],
        analysis["interest_level"],
        analysis["objection_type"],
        trigger
    ))
    DB_CONN.commit()

    # If there's an objection, log it
    if analysis["objection_type"]:
        DB_CONN.execute("""
            INSERT INTO objections(call_sid, objection_text, objection_type, response_used, resolved, timestamp)
            VALUES(?,?,?,?,?,?)
        """, (
            call_sid, text, analysis["objection_type"], response_content, False,
            datetime.datetime.utcnow().isoformat()
        ))
        DB_CONN.commit()

    return response_content, updated_stage