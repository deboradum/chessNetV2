import os
import time
import math
import json
import sqlite3
import chess.pgn
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor


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
            taken = round(time.perf_counter()-s, 2)
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
    stockfish = Stockfish("/opt/homebrew/bin/stockfish", depth=15, parameters={"Threads": 2})
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


def merge_dbs(dbs, new):

    return


if __name__ == "__main__":
    # Parse PGN into databases
    # games_per_db = 100
    # pgn_path = "../data/train_2022-02.pgn"
    # create_db_no_evals(pgn_path, games_per_db, "train", 0, 201)

    # Evaluate db positions
    dbs = [
        "train_0-100000.db",
        "train_100000-200000.db",
        "train_200000-300000.db",
        # "train_300000-400000.db",
        # "train_400000-500000.db",
        # "train_500000-600000.db",
        # "train_600000-700000.db",
        # "train_700000-800000.db",
        # "train_800000-900000.db",
        # "train_900000-1000000.db",
        # "train_1000000-1100000.db",
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit evaluate_db tasks to the executor
        futures = {executor.submit(evalulate_db, db): db for db in dbs}

        for future in futures:
            db = futures[future]
            try:
                print("Starting", db)
                # Wait for the task to complete and check for errors
                future.result()
                print(f"Successfully processed {db}")
            except Exception as e:
                print(f"Error processing {db}: {e}")

    # merge databases to eliminate duplicates
    complete_path = "all.db"
    merge_dbs(dbs, complete_path)

    # Split database into train, test, val set
    # TODO: Based on game_num: 10000 games in test & val set

    # Shuffle database
    # db_path = "train.db"
    # shuffle_database(db_path, "positions")
