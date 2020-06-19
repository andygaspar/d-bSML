from GameTraining.Gym.AItrainer import AITrainer
from GameTraining.gameTraining import GameTraining
from Players.randomPlayer import RandomPlayer

from time import time

CAPACITY = 30_000
SAMPLE_SIZE = 500
HIDDEN = 100
EPOCHS = 20
GAMMA = 0.9

REWARD_NO_SCORE: float = 1
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
t = time()
for i in range(2000):
    game.play(train=False)
    game.reset()

print(time()-t, " go finio de zogar")

for i in range(200):
    game.play(train=True)
    print("gicata ",i," finita +++++++++    ++++++++   ++++++++")
    game.reset()
print(" 1 *****************************")
for i in range(2000):
    game.play(train=False)
    game.reset()

for i in range(200):
    game.play(train=True)
    game.reset()
print(" 2 *****************************")
for i in range(2000):
    game.play(train=False)
    game.reset()

for i in range(200):
    game.play(train=True)
    game.reset()
print(" 3 *****************************")
for i in range(2000):
    game.play(train=False)
    game.reset()

for i in range(200):
    game.play(train=True)
    game.reset()
print(" 4 *****************************")