import sqlite3
import random


def create_database(db_name):
    conn = sqlite3.connect(db_name)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
               fen TEXT PRIMARY KEY,
               padded_fen TEXT,
               eval_value REAL,
               win_perc REAL,
               active_bin_128 INT,
               active_bin_64 INT,
               ascii_codes TEXT,
               norm_ascii_codes TEXT,
               stockfish_eval_20 REAL,
               stockfish_win_perc_20 REAL
           )"""
    )
    return conn


def split_database(input_db):
    conn = sqlite3.connect(input_db)
    cursor = conn.cursor()

    train_conn = create_database("datasetTrain.db")
    val_conn = create_database("datasetVal.db")
    test_conn = create_database("datasetTest.db")
    overfit_conn = create_database("datasetOverfit.db")

    train_count, val_count, test_count, overfit_count = 0, 0, 0, 0
    batch_size = 8192

    cursor.execute("SELECT * FROM positions")
    while True:
        rows = cursor.fetchmany(batch_size)

        for idx, row in enumerate(rows):
            # Randomly assign to one of the three databases
            rand_choice = random.random()
            if rand_choice < 0.8:
                # 80% chance for train
                train_count += 1
                train_conn.execute(
                    "INSERT INTO positions (fen, padded_fen, eval_value, win_perc, active_bin_128, active_bin_64, ascii_codes, norm_ascii_codes, stockfish_eval_20, stockfish_win_perc_20) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
            elif rand_choice < 0.9:
                # 10% chance for validation
                val_count += 1
                val_conn.execute(
                    "INSERT INTO positions (fen, padded_fen, eval_value, win_perc, active_bin_128, active_bin_64, ascii_codes, norm_ascii_codes, stockfish_eval_20, stockfish_win_perc_20) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
            else:
                # 10% chance for test
                test_count += 1
                test_conn.execute(
                    "INSERT INTO positions (fen, padded_fen, eval_value, win_perc, active_bin_128, active_bin_64, ascii_codes, norm_ascii_codes, stockfish_eval_20, stockfish_win_perc_20) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
            if rand_choice < 0.005 and overfit_count<8192:
                # .5% chance for overfit db, used for testing
                overfit_count += 1
                overfit_conn.execute(
                    "INSERT INTO positions (fen, padded_fen, eval_value, win_perc, active_bin_128, active_bin_64, ascii_codes, norm_ascii_codes, stockfish_eval_20, stockfish_win_perc_20) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )

            # Commit in batches
            if (idx + 1) % batch_size == 0:
                train_conn.commit()
                val_conn.commit()
                test_conn.commit()
                overfit_conn.commit()

    # Final commit for any remaining rows
    train_conn.commit()
    val_conn.commit()
    test_conn.commit()
    overfit_conn.commit()

    # Close all connections
    train_conn.close()
    val_conn.close()
    test_conn.close()
    overfit_conn.close()
    conn.close()

    print(f"Total entries: {len(rows)}")
    print(
        f"Train entries: {train_count}, Validation entries: {val_count}, Test entries: {test_count}, Overfit count: {overfit_count}"
    )


input_database = "dataset.db"
split_database(input_database)
