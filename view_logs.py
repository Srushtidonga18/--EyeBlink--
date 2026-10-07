import sqlite3
import pandas as pd
import os
import sys
from db.db_config import SQLITE_PATH, get_logs_dataframe

def view_logs(limit=20):
    print("==========================================================")
    print("       SMART DRIVER SYSTEM - DATABASE LOG VIEWER")
    print("==========================================================")
    print(f"[Database] SQLite: {SQLITE_PATH}\n")

    if not os.path.exists(SQLITE_PATH):
        print("[Notice] No logs found yet. Run 3_realtime_detect.py to generate events.")
        return

    df = get_logs_dataframe()

    if df.empty:
        print("[Notice] Database table is empty. No drowsiness/yawn events recorded yet.")
        return

    # Summary Statistics
    total_events = len(df)
    drowsy_count = (df['status'] == 'DROWSY').sum()
    yawn_count = (df['status'] == 'YAWNING').sum()
    avg_ear = df['ear_value'].mean()
    avg_conf = df['confidence'].mean() * 100

    print("----------------------------------------------------------")
    print("SUMMARY STATISTICS:")
    print(f"   * Total Logged Incidents : {total_events}")
    print(f"   * Drowsy Incidents       : {drowsy_count} ({(drowsy_count/total_events)*100:.1f}%)")
    print(f"   * Yawning Incidents      : {yawn_count} ({(yawn_count/total_events)*100:.1f}%)")
    print(f"   * Average EAR on Trigger : {avg_ear:.3f}")
    print(f"   * Average Model Conf     : {avg_conf:.1f}%")
    print("----------------------------------------------------------\n")

    # Display Recent Events
    print(f"LAST {min(limit, total_events)} LOGGED INCIDENTS:")
    print(df.head(limit).to_string(index=False))

    # Export to CSV for College Report
    os.makedirs('reports', exist_ok=True)
    export_csv = 'reports/detection_logs_export.csv'
    df.to_csv(export_csv, index=False)
    print(f"\n[Exported] Full audit log exported to: {export_csv}")
    print("==========================================================")

if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    view_logs(limit)
