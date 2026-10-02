import sqlite3

connection = sqlite3.connect("eval_dashboard.db")
cursor = connection.cursor()

columns = [
    row[1]
    for row in cursor.execute("PRAGMA table_info(runs)")
]

if "prompt_version" not in columns:
    cursor.execute(
        "ALTER TABLE runs ADD COLUMN prompt_version VARCHAR(50) DEFAULT 'v1'"
    )

if "temperature" not in columns:
    cursor.execute(
        "ALTER TABLE runs ADD COLUMN temperature FLOAT"
    )

if "max_tokens" not in columns:
    cursor.execute(
        "ALTER TABLE runs ADD COLUMN max_tokens INTEGER"
    )

connection.commit()
connection.close()

print("Experiment tracking columns updated successfully")