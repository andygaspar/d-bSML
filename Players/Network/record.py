from Game.board import Board


class Record:
    state: Board
    action: int
    nextState: Board
    reward: float

    def __init__(self, state: Board, action: int, nextState: Board, reward: float):
        self.state = state
        self.action = action
        self.nextState = nextState
        self.reward = reward
