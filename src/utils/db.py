import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_FILE = 'patient_history.db'

def init_db():
    """Initialize the SQLite database and create the history table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            modality TEXT,
            filename TEXT,
            result TEXT,
            confidence REAL,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_record(modality, filename, result, confidence, details=""):
    """
    Add a new diagnostic record to the database.
    
    Args:
        modality (str): 'X-Ray' or 'ECG'
        filename (str): Name of the uploaded file
        result (str): Diagnostic result (e.g., 'Pneumonia')
        confidence (float): Confidence score (0.0 - 1.0)
        details (str): Additional info (e.g., specific arrhythmia type)
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO history (timestamp, modality, filename, result, confidence, details)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, modality, filename, result, confidence, details))
    conn.commit()
    conn.close()

def get_history():
    """Retrieve all records from the database as a Pandas DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    # Read into DataFrame for easy display in Streamlit
    try:
        df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
    except:
        df = pd.DataFrame(columns=['id', 'timestamp', 'modality', 'filename', 'result', 'confidence', 'details'])
    conn.close()
    return df

def get_stats():
    """Retrieve basic usage statistics."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM history")
    total_scans = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM history WHERE result != 'Normal' AND result != 'Normal Rhythm'")
    anomalies = c.fetchone()[0]
    
    conn.close()
    return total_scans, anomalies
