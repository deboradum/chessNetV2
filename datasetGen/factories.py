import json
import sqlite3
import mlx.core as mx

from datasetGen.constants import BIN_SIZE, PREFETCH_BATCH_SIZE


# TODO: get bin not from active_bin, but transform the appropriate win chance here directly, this offers more flexibility.
def buildinWinsIterableFactory(db_path):
    def generator():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ascii_codes, win_perc FROM positions")

        while True:
            rows = cursor.fetchmany(PREFETCH_BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                if BIN_SIZE == 128:
                    yield dict(x=json.loads(row[0]), y=get_bucket(row[1], 128))
                elif BIN_SIZE == 64:
                    yield dict(x=json.loads(row[0]), y=get_bucket(row[1], 64))
                elif BIN_SIZE == 1:
                    yield dict(
                        x=json.loads(row[0]), y=mx.expand_dims(mx.array(row[1]), 0)
                    )
                else:
                    raise Exception("Unimplemented bin size")

        cursor.close()
        conn.close()

    return generator


# TODO: get bin not from active_bin, but transform the appropriate win chance here directly, this offers more flexibility.
def normedBuildinWinsIterableFactory(db_path):
    def generator():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT norm_ascii_codes, win_perc FROM positions")

        while True:
            rows = cursor.fetchmany(PREFETCH_BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                if BIN_SIZE == 128:
                    yield dict(x=json.loads(row[0]), y=get_bucket(row[1], 128))
                elif BIN_SIZE == 64:
                    yield dict(x=json.loads(row[0]), y=get_bucket(row[1], 64))
                elif BIN_SIZE == 1:
                    yield dict(
                        x=json.loads(row[0]), y=mx.expand_dims(mx.array(row[1]), 0)
                    )
                else:
                    raise Exception("Unimplemented bin size")

        cursor.close()
        conn.close()

    return generator


def get_bucket(win_perc, num_buckets):
    return min(int(win_perc / (1 / num_buckets)), num_buckets - 1)


# TODO: get bin not from active_bin, but transform the appropriate win chance here directly, this offers more flexibility.
def stockfishIterableFactory(db_path):
    def generator():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT norm_ascii_codes, stockfish_win_perc_20 FROM positions")

        while True:
            rows = cursor.fetchmany(PREFETCH_BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                if BIN_SIZE == 128:
                    yield dict(x=json.loads(row[0]), y=get_bucket(row[1], 128))
                elif BIN_SIZE == 64:
                    yield dict(x=json.loads(row[0]), y=get_bucket(row[1], 64))
                elif BIN_SIZE == 1:
                    yield dict(
                        x=json.loads(row[0]), y=mx.expand_dims(mx.array(row[1]), 0)
                    )
                else:
                    raise Exception("Unimplemented bin size")

        cursor.close()
        conn.close()

    return generator
