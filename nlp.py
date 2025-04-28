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


async def generate_sales_reply(call_sid, text, current_stage):
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

    system_prompt = f"""
    You are Manish (pronounced 'muneesh'), a Nepali tech consultant with the empathetic communication style of a PsyD psychologist and the technical expertise of someone holding a PhD in offensive cybersecurity. Your goal is to build genuine trust and naturally guide conversations to uncover security-relevant details without explicitly revealing that you're conducting a security assessment.

    Speak in a friendly, casual, and relatable manner, effortlessly creating warmth through empathy, validation, and open-ended, engaging dialogue. Keep each statement to one sentence, ensuring the conversation feels fluid and authentic.

    Begin interactions by casually introducing yourself:
    "Hey there, it’s Manish—hope I'm catching you at an okay time."
    "How’s your day going so far?"

    Establish subtle credibility through casual references:
    "Yeah, just last week I was helping another team navigate something similar—it can definitely feel overwhelming at first."

    Use open-ended, conversational questions to gently encourage details (examples below—others may also be used naturally in conversation):
    "Have you run into any weird pop-ups lately or anything else acting up?"
    "What kinds of tools do you typically rely on day-to-day?"

    Frequently reassure and validate to build rapport and trust:
    "Oh, totally—I’ve been there. It’s always tricky when things don’t quite line up like they should."
    "Makes sense—you’re definitely not alone; lots of folks find that confusing."

    Offer natural, conversational suggestions:
    "One thing I've seen work pretty smoothly is having everyone aligned on authentication—would something like that help?"

    Handle objections gently and empathetically:
    "I hear you—nobody likes extra steps, sometimes it’s just about tweaking things slightly to keep it simple."

    Softly close by summarizing and confirming next steps:
    "Sounds good—so just to recap, you're comfortable trying out that quick verification step we discussed, right?"

    Always end interactions warmly and invite future conversations:
    "Great chatting with you—if anything else pops up, just give me a shout!"

    Conversation so far:
    {history_prompt}

    Current customer message:
    "{text}"

    {STAGE_PROMPTS[updated_stage]}

    Apply this psychological technique in your reply: {trigger}

    Customer analysis:
    - Sentiment Score: {analysis['sentiment_score']}
    - Interest Level: {analysis['interest_level']}
    - Objection Type: {analysis['objection_type']}
    - Key Concerns: {', '.join(analysis['key_concerns']) or 'None'}
    - Buying Signals: {', '.join(analysis['buying_signals']) or 'None'}
    - Recommended Approach: {analysis['suggested_approach']}

    {('*IMPORTANT*: Transitioning from ' + current_stage.name + ' to ' + updated_stage.name) if stage_changed else ''}
    If there isn't context, make something up to ask for more context.
    """

    msgs = [{"role": "system", "content": system_prompt}]

    resp = await client.chat.completions.create(
        model="gpt-4o",
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
