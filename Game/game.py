from Game.board import Board


class Game:

    def __init__(self, players: list, boardsize: int = 4):
        self.board = Board(boardsize)
        self.numBoxes = 0
        self.players = players

    def is_valid(self, idx: int) -> bool:
        return not self.board.vectorBoard[idx]  # (1==True gives False, 0 == False gives True)

    def play(self):

        currentPlayer = self.players[0]
        otherPlayer = self.players[1]
        turn = 0
        PlayerTurn = 0
        N = self.board.size
        newNumBoxes = 0

        self.board.print_board()

        while turn < (2 * N + 2) * N:

            move = currentPlayer.get_move(self.board.vectorBoard)

            while not self.is_valid(move):
                currentPlayer.invalidMove()
                move = currentPlayer.get_move(self.board.vectorBoard)
                # print("Invalid Move")

            self.board.set_board(move)
            turn += 1

            newNumBoxes = self.board.count_boxes()
            # print(newNumBoxes)

            if newNumBoxes - self.numBoxes == 0:
                PlayerTurn += 1
                currentPlayer = self.players[PlayerTurn % 2]
                otherPlayer = self.players[(PlayerTurn + 1) % 2]
            else:
                currentPlayer.scored(newNumBoxes - self.numBoxes)
                self.numBoxes = newNumBoxes

            self.board.print_board()

        # print("Players score: " + str([p.score for p in self.players]))
