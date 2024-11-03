import json
import sqlite3
import mlx.core as mx

from datasetGen.constants import BIN_SIZE, PREFETCH_BATCH_SIZE


def buildinWinsIterableFactory(db_path):
    def generator():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ascii_codes, active_bin_128, active_bin_64, win_perc FROM positions"
        )

        while True:
            rows = cursor.fetchmany(PREFETCH_BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                if BIN_SIZE == 128:
                    yield dict(x=json.loads(row[0]), y=row[1])
                elif BIN_SIZE == 64:
                    yield dict(x=json.loads(row[0]), y=row[2])
                elif BIN_SIZE == 1:
                    yield dict(x=json.loads(row[0]), y=mx.expand_dims(mx.array(row[3]), 0))
                else:
                    raise Exception("Unimplemented bin size")

        cursor.close()
        conn.close()

    return generator
