import json
import sqlite3

PREFETCH_BATCH_SIZE = 8096


def iterableFactory(db_path, num_classes):
    assert num_classes > 1, "Bin size should be at least 1"
    def generator():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT padded_ascii_codes, stockfish_win_perc_20 FROM positions"
        )

        while True:
            rows = cursor.fetchmany(PREFETCH_BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                yield dict(x=json.loads(row[0]), y=get_bucket(row[1], num_classes))

        cursor.close()
        conn.close()

    return generator


def get_bucket(win_perc, num_buckets):
    return min(int(win_perc / (1 / num_buckets)), num_buckets - 1)
