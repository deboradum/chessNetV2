import time
import chess

import mlx.core as mx

from Model import ChessNet
from datasetGen.createDataset import pad_fen


def board_after_move(board, move):
    temp_board = board.copy()
    temp_board.push(move)

    return temp_board


def board_to_input_(board):
    fen = board.fen()
    padded_fen = pad_fen(fen)
    ascii_list = [ord(c) for c in padded_fen]

    return ascii_list


class Engine:
    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose

    def get_best_move(self, fen):
        s = time.perf_counter()
        board = chess.Board(fen)
        if self.verbose:
            print("Getting optimal move for board:")
            print(board)

        legal_moves = [m for m in board.legal_moves]
        if len(legal_moves) == 0:
            print("No legal moves possible.")
            return

        if self.verbose:
            print(f"Found {len(legal_moves)} legal moves")

        boards_after_moves = mx.stack(
            [mx.array(board_to_input_(board_after_move(board, m))) for m in legal_moves]
        )

        logits = self.model(boards_after_moves)
        # Get most likely win chance
        win_chance_per_move = mx.argmax(logits, axis=1)
        if self.verbose:
            print("Found", win_chance_per_move, win_chance_per_move.shape)

        # Best move is the move with the lowest win chance for the opponent.
        best_move = legal_moves[int(mx.argmin(win_chance_per_move))]

        if self.verbose:
            print("Move with the best win chance:", best_move)
            print("Board after best move:")
            board.push(best_move)
            print(board)

            taken = round(time.perf_counter() - s, 5)
            print(f"Took {taken} seconds")

        return best_move


if __name__ == "__main__":
    net = ChessNet(4, 4, 128, 1024, 128)
    net.load_weights("best.npz")
    e = Engine(net, verbose=True)

    while True:
        f = input("\nType fen string: ")
        try:
            best_move = e.get_best_move(f)
            print(best_move)
        except Exception as e:
            print("Unexpected error:", e)
