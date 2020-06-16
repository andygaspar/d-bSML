from GameTraining.Gym.AItrainer import AITrainer
from GameTraining.gameTraining import GameTraining
from Players.randomPlayer import RandomPlayer

CAPACITY = 30_000
SAMPLE_SIZE = 500
HIDDEN = 100
EPOCHS = 1000

REWARD_SCORE: float = 10
REWARD_OPPONENT_SCORE: float = -10
REWARD_INVALID_SCORE: float = -1000
REWARD_WIN = 50
REWARD_LOOSE = -50

boardsize = 3

players = [RandomPlayer(1, boardsize), AITrainer(2, boardsize)]
game = GameTraining(players, boardsize)
