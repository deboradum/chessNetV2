import os
import sys
import time
import torch
import chess
import chess.pgn

from EngineTorch import Engine
from ModelTorch import ChessNet
from stockfish import Stockfish


class Player:
    def __init__(self, is_stockfish, engine):
        # True if stockfish, False if custom engine
        self.is_stockfish = is_stockfish
        self.engine = engine
        self.name = f"stockfish - {engine.get_parameters()['UCI_Elo']} elo" if is_stockfish else "chessNet"

    def get_move(self, fen):
        if self.is_stockfish:
            self.engine.set_fen_position(fen)
            move = chess.Move.from_uci(self.engine.get_best_move())
        else:
            move = self.engine.get_best_move(fen)

        return move


def clear_screen():
    if sys.platform.startswith("win"):
        os.system("cls")  # For Windows
    else:
        os.system("clear")  # For Unix/Linux/MacOS


def get_stockfish(depth, elo_rating):
    stockfish = Stockfish(
        "/opt/homebrew/bin/stockfish", depth=depth, parameters={"Threads": 2, "UCI_Elo": elo_rating}
    )
    stockfish.set_elo_rating(elo_rating)

    return stockfish


def run_game(board, black, white, pgn_path=""):
    game = chess.pgn.Game()
    node = game
    game.headers["White"] = white.name
    game.headers["Black"] = black.name

    while not board.is_game_over():
        fen = board.fen()
        if board.turn == chess.WHITE:
            move = white.get_move(fen)
        else:
            move = black.get_move(fen)

        clear_screen()
        print(board)

        board.push(move)

        node = node.add_variation(move)

    if pgn_path == "":
        return

    with open(pgn_path, "w") as pgn_file:
        print(game, file=pgn_file)


if __name__ == "__main__":
    for elo in range(100, 3000, 100):
        stockfish = get_stockfish(10, elo)

        net = ChessNet(4, 4, 128, 1024, 128)
        net.load_state_dict(
            torch.load(
                "torch_adam_4e-05_128_4_4_1024_128_128_epoch_1_batch_300000.pt",
                map_location=torch.device("cpu"),
            )
        )
        e = Engine(net, verbose=False)

        sf = Player(True, stockfish)
        eng = Player(False, e)

        run_game(chess.Board(), sf, eng, f"stockfish_{elo}.pgn")
