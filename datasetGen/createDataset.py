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


def get_all_pgn_files(root_dir, max_depth=10):
    files = []

    def search_directory(current_dir, current_depth):
        if current_depth > max_depth:
            return
        for entry in os.scandir(current_dir):
            if entry.is_file() and entry.name.endswith(".pgn"):
                files.append(os.path.abspath(entry.path))
            elif entry.is_dir():
                search_directory(entry.path, current_depth + 1)

    search_directory(root_dir, 0)
    return files


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


def process_file(filepath, s, e):
    conn = sqlite3.connect(f"dataset{s}{e}.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
               fen TEXT PRIMARY KEY,
               padded_fen TEXT,
               padded_ascii_codes TEXT,
               stockfish_eval_20 REAL,
               stockfish_win_perc_20 REAL
           )"""
    )

    stockfish = Stockfish("/opt/homebrew/bin/stockfish", depth=20)
    stockfish.set_skill_level(20)

    pgn = open(filepath)
    g = chess.pgn.read_game(pgn)
    start = time.perf_counter()
    while True:
        positions = []
        g = chess.pgn.read_game(pgn)
        if not g:
            break

        board = g.board()
        for node in g.mainline():
            move = node.move
            board.push(move)

            fen = board.fen()
            padded_fen = pad_fen(fen)
            ascii_list = [ord(c) for c in padded_fen]
            padded_ascii_codes = json.dumps(ascii_list)
            stockfish_eval_20 = get_stockfish_eval(fen, stockfish)

            if not board.turn:
                stockfish_eval_20 = -stockfish_eval_20
            stockfish_win_perc_20 = get_win_perc(stockfish_eval_20 * 100)

            positions.append(
                (
                    fen,
                    padded_fen,
                    padded_ascii_codes,
                    stockfish_eval_20,
                    stockfish_win_perc_20,
                )
            )

        # print("Adding game with", len(positions), "positions")
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO positions (fen, padded_fen, padded_ascii_codes, stockfish_eval_20, stockfish_win_perc_20) VALUES (?, ?, ?, ?, ?)",
            positions,
        )
        conn.commit()
    taken = round(time.perf_counter()-start, 2)
    print(f"Parsed file, took {taken}s")
    conn.close()


def main(filepaths, s, e):
    with ThreadPoolExecutor(max_workers=7) as executor:
        executor.map(lambda filepath: process_file(filepath, s, e), filepaths)


# Get and write paths to file
# paths = get_all_pgn_files("../data")
# with open("all_paths.txt", "w") as f:
#     for path in paths:
#         f.write(path + "\n")

# Read paths from file
with open("all_paths.txt", "r") as f:
    paths = [line.strip() for line in f]

s = 200
e = 300
main(paths[s:e], s, e)
