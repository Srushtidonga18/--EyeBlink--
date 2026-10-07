import os
import sqlite3
import pandas as pd

# Path to the local SQLite database file
SQLITE_PATH = os.path.join(os.path.dirname(__file__), 'drowsiness_logs.db')

def init_db():
    """Initialize detection_logs table in local SQLite database."""
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id TEXT DEFAULT 'DRIVER_001',
            status TEXT NOT NULL,
            ear_value REAL NOT NULL,
            mor_value REAL NOT NULL,
            confidence REAL NOT NULL,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()
    print(f'[DB] SQLite database initialized successfully at: {SQLITE_PATH}')

def log_alert(status, ear, mor, confidence):
    """Log drowsiness or yawn events to the local SQLite database."""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO detection_logs (status, ear_value, mor_value, confidence)
            VALUES (?, ?, ?, ?);
        ''', (status, float(ear), float(mor), float(confidence)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'[DB Error] Failed to log event: {e}')

def get_logs_dataframe():
    """Helper function to load logs as Pandas DataFrame."""
    if not os.path.exists(SQLITE_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM detection_logs ORDER BY id DESC", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

if __name__ == '__main__':
    init_db()
