import os
import time
import math
import json
import random
import tqdm
import sqlite3
import chess.pgn
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor


def create_database(db_name):
    conn = sqlite3.connect(db_name)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
            fen TEXT PRIMARY KEY,
            padded_fen TEXT,
            padded_ascii_codes TEXT,
            stockfish_eval_20 REAL,
            stockfish_win_perc_20 REAL
        )"""
    )
    return conn


def get_win_perc(centipawns):
    centipawns = max(min(centipawns, 1000), -1000)
    return 0.5 * 2 / (1 + math.exp(-0.00368208 * centipawns))


def pad_fen(fen):
    parts = fen.split()
    # 1. Piece placement (pad to 72 characters)
    piece_placement = parts[0].ljust(72, ".")
    # 2. Active color (already 1 character, no padding needed)
    active_color = parts[1]
    # 3. Castling availability (pad to 4 characters)
    castling_availability = parts[2].ljust(4, ".")
    # 4. En passant target square (pad to 2 characters)
    en_passant = parts[3].ljust(2, ".")
    # 5. Halfmove clock (pad to 3 digits)
    halfmove_clock = parts[4].zfill(3)
    # 6. Fullmove number (pad to 4 digits)
    fullmove_number = parts[5].zfill(4)

    return f"{piece_placement}{active_color}{castling_availability}{en_passant}{halfmove_clock}{fullmove_number}"


def get_stockfish_eval(fen, stockfish):
    stockfish.set_fen_position(fen)
    evaluation = stockfish.get_evaluation()
    if evaluation["type"] == "cp":
        return evaluation["value"] / 100
    elif evaluation["type"] == "mate":
        mate_score = 10000 * (1 if evaluation["value"] > 0 else -1)
        return mate_score
    else:
        raise ValueError("Unexpected evaluation type returned by Stockfish")


# Parses a Stockfish database PGN file into .db files, with <max_games> per
# database. Evals are manually set to -1, they will be calculated later on a
# different machine for speed reasons.
def create_db_no_evals(filepath, max_games, label, start_offset, end_offset):
    conn = None

    pgn = open(filepath)
    g = chess.pgn.read_game(pgn)
    start = time.perf_counter()
    num_games = 0
    s = time.perf_counter()
    while True:
        if num_games == end_offset:
            break
        positions = []
        g = chess.pgn.read_game(pgn)
        if not g:
            break
        if num_games < start_offset:
            num_games += 1
            continue

        if num_games % 1000 == 0:
            taken = round(time.perf_counter() - s, 2)
            print(f"Did 1000 games in {taken}s")
            s = time.perf_counter()

        if num_games % max_games == 0:
            print("Max encountered! New name:")
            print(f"{label}_{num_games}-{num_games+max_games}.db")
            conn = sqlite3.connect(f"{label}_{num_games}-{num_games+max_games}.db")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS positions (
                    fen TEXT PRIMARY KEY,
                    padded_fen TEXT,
                    padded_ascii_codes TEXT,
                    stockfish_eval_20 REAL,
                    stockfish_win_perc_20 REAL,
                    game_num INT
                )"""
            )

        num_games += 1

        board = g.board()
        for node in g.mainline():
            move = node.move
            board.push(move)

            fen = board.fen()
            padded_fen = pad_fen(fen)
            ascii_list = [ord(c) for c in padded_fen]
            padded_ascii_codes = json.dumps(ascii_list)
            stockfish_eval_20 = -1
            stockfish_win_perc_20 = -1

            positions.append(
                (
                    fen,
                    padded_fen,
                    padded_ascii_codes,
                    stockfish_eval_20,
                    stockfish_win_perc_20,
                    num_games,
                )
            )

            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20, game_num) VALUES (?, ?, ?, ?, ?, ?)",
                positions,
            )
            conn.commit()
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20, game_num) VALUES (?, ?, ?, ?, ?, ?)",
            positions,
        )
        conn.commit()
    taken = round(time.perf_counter() - start, 2)
    print(f"Parsed file, took {taken}s")
    conn.close()


def shuffle_database(db_name, table_name):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        temp_table_name = f"{table_name}_shuffled"

        cursor.execute(f"""
            CREATE TABLE {temp_table_name} AS
            SELECT * FROM {table_name} ORDER BY RANDOM();
        """)

        # Backup the original table
        backup_table_name = f"{table_name}_backup"
        cursor.execute(f"ALTER TABLE {table_name} RENAME TO {backup_table_name};")

        cursor.execute(f"ALTER TABLE {temp_table_name} RENAME TO {table_name};")
        conn.commit()

        print(
            f"Table '{table_name}' has been shuffled successfully. "
            f"Original table is backed up as '{backup_table_name}'."
        )
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()


def evalulate_db(db_path):
    stockfish = Stockfish(
        "/opt/homebrew/bin/stockfish", depth=15, parameters={"Threads": 2}
    )
    stockfish.set_skill_level(20)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Select rows where stockfish_win_perc_20 is -1.0
    cursor.execute("SELECT fen FROM positions WHERE stockfish_win_perc_20 = -1.0")
    # cursor.execute("SELECT fen FROM positions")
    rows = cursor.fetchall()
    count = 0
    print("Starting.")
    s = time.perf_counter()
    for row in rows:
        fen = row[0]
        try:
            new_eval = get_stockfish_eval(fen, stockfish)
            parts = fen.split()
            if not parts[1] == "w":
                new_eval = -new_eval
            new_win_perc = get_win_perc(new_eval * 100)
        except Exception as e:
            print(f"Error evaluating FEN {fen}: {e}")
            continue
        cursor.execute(
            "UPDATE positions SET stockfish_eval_20 = ?, stockfish_win_perc_20 = ? WHERE fen = ?",
            (new_eval, new_win_perc, fen),
        )
        count += 1

        if count % 1000 == 0:
            conn.commit()
            taken = round(time.perf_counter() - s, 2)
            print(f"[{db_path}]: Committed {count} updates in {taken} seconds.")
            s = time.perf_counter()
    conn.commit()
    print(f"[{db_path}] Completed updating {count} rows.")
    conn.close()


def merge_dbs(source_dbs, new_db):
    dest_conn = sqlite3.connect(new_db)
    dest_conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
                fen TEXT PRIMARY KEY,
                padded_fen TEXT,
                padded_ascii_codes TEXT,
                stockfish_eval_20 REAL,
                stockfish_win_perc_20 REAL,
                game_num INT
            )"""
    )
    dest_conn.commit()
    dest_cursor = dest_conn.cursor()

    batch_size = 65536
    for db in source_dbs:
        source_conn = sqlite3.connect(db)
        source_cursor = source_conn.cursor()

        # Get number of rows
        source_cursor.execute("SELECT COUNT(*) FROM positions")
        total_rows = source_cursor.fetchone()[0]

        with tqdm.tqdm(total=total_rows, desc=f"Processing {db}") as pbar:
            try:
                source_cursor.execute("SELECT * FROM positions")
                while True:
                    rows = source_cursor.fetchmany(batch_size)
                    if not rows:
                        break  # Exit the loop when there are no more rows to fetch

                    # s = time.perf_counter()
                    for row in rows:
                        (
                            _,
                            _,
                            _,
                            stockfish_eval_20,
                            stockfish_win_perc_20,
                            _,
                        ) = row
                        if abs(stockfish_eval_20) == 10 or stockfish_win_perc_20 < 0:
                            continue
                        dest_cursor.execute(
                            "INSERT OR IGNORE INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20, game_num) VALUES (?, ?, ?, ?, ?, ?)",
                            row,
                        )
                    dest_conn.commit()
                    # taken = round(time.perf_counter() - s, 2)
                    pbar.update(len(rows))
                    # print(f"Parsed {len(rows)} rows, took {taken} seconds")
            except sqlite3.DatabaseError as e:
                print(f"Error processing {db}: {e}")
                source_conn.close()
                continue
        source_conn.close()
    dest_conn.close()


def split_database(input_db, train_games, eval_games, test_games, hptune_games):
    conn = sqlite3.connect(input_db)
    cursor = conn.cursor()

    train_conn = create_database("train.db")
    val_conn = create_database("val.db")
    test_conn = create_database("test.db")
    hptune_conn = create_database("hptune.db")

    train_count, val_count, test_count, hptune_count = 0, 0, 0, 0
    batch_size = 65536

    cursor.execute("SELECT COUNT(*) FROM positions")
    total_rows = cursor.fetchone()[0]

    cursor.execute("SELECT * FROM positions")
    with tqdm.tqdm(total=total_rows, desc="Splitting databases") as pbar:
        while True:
            rows = cursor.fetchmany(batch_size)

            for idx, row in enumerate(rows):
                (
                    _,
                    _,
                    _,
                    stockfish_eval_20,
                    stockfish_win_perc_20,
                    game_num,
                ) = row

                # Randomly assign to one of the three databases
                train_game_num_start, train_game_num_end = train_games
                if game_num >= train_game_num_start and game_num <= train_game_num_end:
                    train_count += 1
                    train_conn.execute(
                        "INSERT INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20, game_num) VALUES (?, ?, ?, ?, ?, ?)",
                        row,
                    )
                eval_game_num_start, eval_game_num_end = eval_games
                if game_num >= eval_game_num_start and game_num <= eval_game_num_end:
                    # 10% chance for validation
                    val_count += 1
                    val_conn.execute(
                        "INSERT INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20, game_num) VALUES (?, ?, ?, ?, ?, ?)",
                        row,
                    )
                test_game_num_start, test_game_num_end = test_games
                if game_num >= test_game_num_start and game_num <= test_game_num_end:
                    # 10% chance for test
                    test_count += 1
                    test_conn.execute(
                        "INSERT INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20, game_num) VALUES (?, ?, ?, ?, ?, ?)",
                        row,
                    )
                hptune_game_num_start, hptune_game_num_end = hptune_games
                if game_num >= hptune_game_num_start and game_num <= hptune_game_num_end:
                    hptune_count += 1
                    hptune_conn.execute(
                        "INSERT INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20, game_num) VALUES (?, ?, ?, ?, ?, ?)",
                        row,
                    )

                if (idx + 1) % batch_size == 0:
                    train_conn.commit()
                    val_conn.commit()
                    test_conn.commit()
                    hptune_conn.commit()
            pbar.update(len(rows))

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
        f"Train entries: {train_count}, Validation entries: {val_count}, Test entries: {test_count}, hptune count: {hptune_count}"
    )


if __name__ == "__main__":
    # Parse PGN into databases
    # games_per_db = 100
    # pgn_path = "../data/train_2022-02.pgn"
    # create_db_no_evals(pgn_path, games_per_db, "train", 0, 201)

    dbs = [
        # "train_0-100000.db",
        # "train_100000-200000.db",
        # "train_200000-300000.db",
        # "train_300000-400000.db",
        # "train_400000-500000.db",
        # "train_500000-600000.db",
        # "train_600000-700000.db",
        # "train_700000-800000.db",
        "train_800000-900000.db",
        "train_900000-1000000.db",
        "train_1000000-1100000.db",
    ]

    # Evaluate db positions
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     # Submit evaluate_db tasks to the executor
    #     futures = {executor.submit(evalulate_db, db): db for db in dbs}
    #     for future in futures:
    #         db = futures[future]
    #         try:
    #             print("Starting", db)
    #             # Wait for the task to complete and check for errors
    #             future.result()
    #             print(f"Successfully processed {db}")
    #         except Exception as e:
    #             print(f"Error processing {db}: {e}")

    # merge databases to eliminate duplicates
    complete_path = "all.db"
    merge_dbs(dbs, complete_path)

    # Split database into train, test, val set
    train_games = (0, 1000000)
    eval_games = (1000001, 1050000)
    test_games = (1050001, 1100000)
    hptune_games = (0, 100000)
    split_database("all.db", train_games, eval_games, test_games, hptune_games)

    # Shuffle database
    # db_path = "train.db"
    # shuffle_database(db_path, "positions")
