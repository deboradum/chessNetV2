import sqlite3
import random


def create_database(db_name):
    conn = sqlite3.connect(db_name)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
               fen TEXT PRIMARY KEY,
               padded_fen TEXT,
               win_perc REAL,
               active_bin INT,
               ascii_codes TEXT
           )"""
    )
    return conn


def split_database(input_db):
    conn = sqlite3.connect(input_db)
    cursor = conn.cursor()

    train_conn = create_database("builtinWinsTrain.db")
    val_conn = create_database("builtinWinsVal.db")
    test_conn = create_database("builtinWinsTest.db")

    train_count, val_count, test_count = 0, 0, 0
    total_count = 0
    batch_size = 1024  # Adjust batch size as necessary

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
                    "INSERT INTO positions (fen, padded_fen, win_perc, active_bin, ascii_codes) VALUES (?, ?, ?, ?, ?)",
                    row,
                )
            elif rand_choice < 0.9:
                # 10% chance for validation
                val_count += 1
                val_conn.execute(
                    "INSERT INTO positions (fen, padded_fen, win_perc, active_bin, ascii_codes) VALUES (?, ?, ?, ?, ?)",
                    row,
                )
            else:
                # 10% chance for test
                test_count += 1
                test_conn.execute(
                    "INSERT INTO positions (fen, padded_fen, win_perc, active_bin, ascii_codes) VALUES (?, ?, ?, ?, ?)",
                    row,
                )

            # Commit in batches
            if (idx + 1) % batch_size == 0:
                train_conn.commit()
                val_conn.commit()
                test_conn.commit()

    # Final commit for any remaining rows
    train_conn.commit()
    val_conn.commit()
    test_conn.commit()

    # Close all connections
    train_conn.close()
    val_conn.close()
    test_conn.close()
    conn.close()

    print(f"Total entries: {len(rows)}")
    print(
        f"Train entries: {train_count}, Validation entries: {val_count}, Test entries: {test_count}"
    )


input_database = "builtinWins.db"
split_database(input_database)
