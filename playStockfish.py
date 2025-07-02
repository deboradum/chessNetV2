import os
import sys
import yaml
import torch
import chess
import chess.pgn
import argparse

from engine import Engine
from train import Config
from ModelTransformer import ChessNet
from stockfish import Stockfish

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to config file", required=False
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint file", required=True
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = Config(**config_dict)

    for elo in range(100, 3000, 100):
        stockfish = get_stockfish(10, elo)

        net = ChessNet(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            vocab_size=config.vocab_size,
            embed_dim=config.embedding_dim,
            num_classes=config.num_classes,
            rms_norm=True,
        ).to(device)
        net.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        e = Engine(net, verbose=args.verbose)

        sf = Player(True, stockfish)
        eng = Player(False, e)

        run_game(chess.Board(), sf, eng, f"stockfish_{elo}.pgn")
