import sqlite3

connection = sqlite3.connect("eval_dashboard.db")
cursor = connection.cursor()


# ============================================================
# RUNS TABLE
# ============================================================

run_columns = [
    row[1]
    for row in cursor.execute("PRAGMA table_info(runs)")
]

if "prompt_version" not in run_columns:
    cursor.execute(
        "ALTER TABLE runs ADD COLUMN prompt_version VARCHAR(50) DEFAULT 'v1'"
    )

if "temperature" not in run_columns:
    cursor.execute(
        "ALTER TABLE runs ADD COLUMN temperature FLOAT"
    )

if "max_tokens" not in run_columns:
    cursor.execute(
        "ALTER TABLE runs ADD COLUMN max_tokens INTEGER"
    )


# ============================================================
# RESPONSES TABLE
# ============================================================

response_columns = [
    row[1]
    for row in cursor.execute("PRAGMA table_info(responses)")
]

if "status" not in response_columns:
    cursor.execute(
        "ALTER TABLE responses ADD COLUMN status VARCHAR(50) DEFAULT 'success'"
    )

if "error_message" not in response_columns:
    cursor.execute(
        "ALTER TABLE responses ADD COLUMN error_message TEXT"
    )


# ============================================================
# SAVE CHANGES
# ============================================================

connection.commit()
connection.close()

print("Database schema updated successfully.")