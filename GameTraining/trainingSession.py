from GameTraining.Gym.AItrainer import AITrainer
from GameTraining.gameTraining import GameTraining
from Players.randomPlayer import RandomPlayer

CAPACITY = 30_000
SAMPLE_SIZE = 20
HIDDEN = 100
EPOCHS = 1000
GAMMA = 0.9

REWARD_NO_SCORE:float = 0
REWARD_SCORE: float = 10
REWARD_OPPONENT_SCORE: float = -10
REWARD_INVALID_SCORE: float = -1000
REWARD_WIN = 50
REWARD_LOOSE = -50
UPDATE_STEP = 1

boardsize = 3

use_invalid = False
players = [RandomPlayer(1, boardsize), AITrainer(2, boardsize, HIDDEN, EPOCHS, REWARD_NO_SCORE, REWARD_SCORE,
                                                 REWARD_OPPONENT_SCORE, REWARD_INVALID_SCORE, use_invalid, SAMPLE_SIZE,
                                                 CAPACITY, GAMMA)]
game = GameTraining(players, boardsize)

batch = []
for i in range(200):
    game.play()
    game.reset()
    if i % 20 == 0:
        players[1].train_network()

print(players[1].replayMemory.rewards)
