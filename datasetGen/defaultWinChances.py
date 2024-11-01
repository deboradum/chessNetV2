import os
import math
import json
import sqlite3
import chess.pgn
from constants import BIN_SIZE


def get_win_perc(centipawns):
    return 0.5 * 2 / (1 + math.exp(-0.00368208 * centipawns))


def extract_eval_from_comment(comment):
    if "[%eval " in comment:
        try:
            eval_str = comment.split("[%eval ")[1].split("]")[0]
            eval_value = float(eval_str)
            return eval_value
        except Exception as e:
            return None
    return None


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


def main(filepaths):
    conn = sqlite3.connect("builtinWins.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS positions (
               fen TEXT PRIMARY KEY,
               padded_fen TEXT,
               win_perc REAL,
               active_bin_128 INT,
               active_bin_64 INT,
               ascii_codes TEXT
           )"""
    )

    for filepath in filepaths:
        pgn = open(filepath)
        g = chess.pgn.read_game(pgn)
        # Read all games in the file
        while True:
            positions = []
            g = chess.pgn.read_game(pgn)
            if not g:
                break

            board = g.board()
            for node in g.mainline():
                move = node.move
                eval_value = extract_eval_from_comment(node.comment)

                board.push(move)

                if eval_value is None:
                    continue

                if not board.turn:
                    eval_value = -eval_value

                fen = board.fen()
                padded_fen = pad_fen(fen)
                win_perc = get_win_perc(eval_value * 100)
                active_bin_128 = min(int(win_perc / (1/128)), 127)
                active_bin_64 = min(int(win_perc / (1/64)), 127)
                ascii_codes = json.dumps([ord(c) for c in padded_fen])

                positions.append((fen, padded_fen, win_perc, active_bin_128, active_bin_64, ascii_codes))

            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO positions (fen, padded_fen, win_perc, active_bin_128, active_bin_64, ascii_codes) VALUES (?, ?, ?, ?, ?, ?)",
                positions,
            )
            conn.commit()
    conn.close()


paths = get_all_pgn_files("../data")
main(paths)
