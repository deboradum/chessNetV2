import os
import math
import json
import sqlite3
import chess.pgn

from stockfish import Stockfish

MAX_ASCII = 119
MIN_ASCII = 45


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


def get_stockfish_eval(fen, stockfish):
    stockfish.set_fen_position(fen)
    evaluation = stockfish.get_evaluation()

    if evaluation["type"] == "cp":
        return evaluation["value"]/100
    elif evaluation["type"] == "mate":
        mate_score = 10000 * (1 if evaluation["value"] > 0 else -1)
        return mate_score
    else:
        raise ValueError("Unexpected evaluation type returned by Stockfish")


def norm_ascii(ascii_list):
    return [round((x - MIN_ASCII) / (MAX_ASCII - MIN_ASCII), 5) for x in ascii_list]


def main(filepaths):
    conn = sqlite3.connect("dataset.db")
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
               stockfish_eval_15 REAL,
               stockfish_win_perc_15 REAL
           )"""
    )

    stockfish = Stockfish("/opt/homebrew/bin/stockfish", depth=15)
    stockfish.set_skill_level(20)

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
                active_bin_128 = min(int(win_perc / (1 / 128)), 127)
                active_bin_64 = min(int(win_perc / (1 / 64)), 127)
                ascii_list = [ord(c) for c in padded_fen]
                ascii_codes = json.dumps(ascii_list)
                norm_ascii_codes = json.dumps(norm_ascii(ascii_list))
                stockfish_eval_15 = get_stockfish_eval(fen, stockfish)

                if not board.turn:
                    stockfish_eval_15 = -stockfish_eval_15
                stockfish_win_perc_15 = get_win_perc(stockfish_eval_15 * 100)

                positions.append(
                    (
                        fen,
                        padded_fen,
                        eval_value,
                        win_perc,
                        active_bin_128,
                        active_bin_64,
                        ascii_codes,
                        norm_ascii_codes,
                        stockfish_eval_15,
                        stockfish_win_perc_15,
                    )
                )

            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO positions (fen, padded_fen, eval_value, win_perc, active_bin_128, active_bin_64, ascii_codes, norm_ascii_codes, stockfish_eval_15, stockfish_win_perc_15) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                positions,
            )
            conn.commit()
    conn.close()


paths = get_all_pgn_files("../data")
main(paths)
