import json
import sqlite3

# BATCH_SIZE = 4096
BATCH_SIZE = 1000


def buildinWinsIterableFactory(db_path):
    def generator():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ascii_codes, active_bin FROM positions")

        while True:
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                yield dict(x=json.loads(row[0]), y=row[1])

        cursor.close()
        conn.close()

    return generator