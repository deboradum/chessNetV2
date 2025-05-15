import os
import time
import math
import json
import tqdm
import sqlite3
import chess.pgn
from stockfish import Stockfish
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
DB_PATH = "pgn_dbs/lichess_db_standard_rated_2024-11.db"
CHUNK_SIZE = 500  # Adjust based on your evaluation speed and memory


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
def create_db_no_evals(filepath, db_path, skip_early_game=False):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
            fen TEXT PRIMARY KEY,
            padded_fen TEXT,
            padded_ascii_codes TEXT,
            stockfish_eval_20 REAL,
            stockfish_win_perc_20 REAL
        )"""
    )

    pgn = open(filepath)
    g = chess.pgn.read_game(pgn)
    start = time.perf_counter()
    num_games = 0

    while True:
        positions = []
        g = chess.pgn.read_game(pgn)
        if not g:
            break

        num_games += 1

        board = g.board()
        for turn, node in enumerate(g.mainline()):
            move = node.move
            board.push(move)

            # Skip first 15 moves to get more mid / endgame positions in the dataset
            if skip_early_game and turn < 30:
                continue

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
                )
            )

        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20) VALUES (?, ?, ?, ?, ?)",
            positions,
        )
        conn.commit()
    taken = round(time.perf_counter() - start, 2)
    print(f"Parsed {filepath}, took {taken}s")
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
        "/opt/homebrew/bin/stockfish", depth=15, parameters={"Threads": 4}
    )
    stockfish.set_skill_level(20)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Select rows where stockfish_win_perc_20 is -1.0
    cursor.execute("SELECT fen FROM positions WHERE stockfish_win_perc_20 = -1.0")
    rows = cursor.fetchall()
    count = 0
    print("Starting", db_path)
    updates = []
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
        updates.append((new_eval, new_win_perc, fen))
        count += 1

        if count % 1000 == 0:
            cursor.executemany(
                "UPDATE positions SET stockfish_eval_20 = ?, stockfish_win_perc_20 = ? WHERE fen = ?",
                updates
            )
            updates.clear()
            conn.commit()
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
                stockfish_win_perc_20 REAL
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
                source_cursor.execute(
                    "SELECT fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20 FROM positions"
                )
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
                        ) = row
                        if (abs(stockfish_eval_20) > 100) or stockfish_win_perc_20 < 0:
                            continue
                        dest_cursor.execute(
                            "INSERT OR IGNORE INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20) VALUES (?, ?, ?, ?, ?)",
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



def get_all_ext_files_in_dir(dir_name, ext=".pgn"):
    all_files = []
    for root, _, files in os.walk(dir_name):
        for file in files:
            if file.lower().endswith(ext):
                all_files.append(os.path.join(root, file))
    return all_files


def init_stockfish():
    sf = Stockfish(STOCKFISH_PATH, depth=15, parameters={"Threads": 2})
    sf.set_skill_level(20)
    return sf


def evaluate_chunk(fens):
    stockfish = init_stockfish()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    updates = []
    for fen in fens:
        try:
            new_eval = get_stockfish_eval(fen, stockfish)
            parts = fen.split()
            if parts[1] != "w":
                new_eval = -new_eval
            new_win_perc = get_win_perc(new_eval * 100)
            updates.append((new_eval, new_win_perc, fen))
        except Exception as e:
            print(f"Error evaluating FEN {fen}: {e}")
            continue

    if updates:
        cursor.executemany(
            "UPDATE positions SET stockfish_eval_20 = ?, stockfish_win_perc_20 = ? WHERE fen = ?",
            updates,
        )
        conn.commit()
    conn.close()


def evaluate_db_parallel(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("SELECT fen FROM positions WHERE stockfish_win_perc_20 = -1.0")
    fens = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"Starting evaluation of {len(fens)} positions using {cpu_count()} cores.")

    # Split into chunks
    chunks = [fens[i : i + CHUNK_SIZE] for i in range(0, len(fens), CHUNK_SIZE)]

    start_time = time.perf_counter()
    with Pool(processes=4) as pool:
        pool.map(evaluate_chunk, chunks)

    elapsed = round(time.perf_counter() - start_time, 2)
    print(f"Completed evaluation in {elapsed} seconds.")


if __name__ == "__main__":
    # Parse PGN into databases
    subdir_name = "pgn_dbs"
    os.makedirs(subdir_name, exist_ok=True)
    pgns = get_all_ext_files_in_dir("../data", ".pgn")
    for pgn_path in pgns:
        db_name = subdir_name + pgn_path.replace(".pgn", ".db").replace("../data", "")
        if os.path.isfile(db_name):
            print(f"{db_name} already done")
            continue
        create_db_no_evals(pgn_path, db_name, skip_early_game=True)

    # evaluate_db_parallel("pgn_dbs/lichess_db_standard_rated_2024-11.db")

    # # Evaluate db positions
    dbs = get_all_ext_files_in_dir(subdir_name, ".db")
    dbs.remove("pgn_dbs/lichess_db_standard_rated_2024-11.db")
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit evaluate_db tasks to the executor
        futures = {executor.submit(evalulate_db, db): db for db in dbs}
        for future in futures:
            db = futures[future]
            try:
                # Wait for the task to complete and check for errors
                future.result()
                print(f"Successfully processed {db}")
            except Exception as e:
                print(f"Error processing {db}: {e}")

    # Merge database using cli:
    # sqlite3 dest.db "ATTACH 'source.db' AS db2; INSERT OR IGNORE INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20) SELECT fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20 FROM db2.positions WHERE stockfish_eval_20 != -1; DETACH db2;";

    # Split database into train, test, val set
    # TODO

    # # Shuffle database
    # db_path = "train.db"
    # shuffle_database(db_path, "positions")
