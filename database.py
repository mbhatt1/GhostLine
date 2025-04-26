# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────
import sqlite3



def init_db(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()

    # calls table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            call_sid TEXT PRIMARY KEY,
            start_time TEXT,
            end_time TEXT NULL,
            voice_id TEXT,
            campaign TEXT,
            persona TEXT,
            phone_number TEXT,
            outcome TEXT NULL,
            conversion_score REAL NULL,
            notes TEXT NULL
        )
    """)

    # messages table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            sales_stage TEXT NULL,
            sentiment_score REAL NULL,
            interest_level REAL NULL,
            objection_type TEXT NULL,
            trigger_used TEXT NULL,
            FOREIGN KEY (call_sid) REFERENCES calls(call_sid)
        )
    """)

    # objections table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS objections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid TEXT,
            objection_text TEXT,
            objection_type TEXT,
            response_used TEXT,
            resolved BOOLEAN,
            timestamp TEXT,
            FOREIGN KEY (call_sid) REFERENCES calls(call_sid)
        )
    """)

    # customer_profiles table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customer_profiles (
            phone_number TEXT PRIMARY KEY,
            communication_style TEXT NULL,
            pain_points TEXT NULL,
            response_rates TEXT NULL,
            preferred_persona TEXT NULL,
            last_updated TEXT
        )
    """)

    conn.commit()
    return conn


DB_CONN = init_db("sales_tracking.db")