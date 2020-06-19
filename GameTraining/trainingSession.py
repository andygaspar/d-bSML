from GameTraining.Gym.AItrainer import AITrainer
from GameTraining.gameTraining import GameTraining
from Players.AIPlayer import AIPlayer
from Players.randomPlayer import RandomPlayer
from Game.game import Game
from time import time
import numpy as np


def network_experience(game: GameTraining, num_games: int, get_wins: bool = False):
    wins = []
    for i in range(num_games):
        game.play(train=False)
        if get_wins:
            wins.append(players[1].score >= players[0].score)
        game.reset()
    print("win rate  ", sum(wins) / len(wins))


def training_cycle(game: GameTraining, num_games: int):
    losses = []
    interval: int = 100
    for i in range(num_games):
        game.play(train=True)
        losses.append(game.players[1].network.loss)
        game.reset()
        if i % interval == 0:
            print("Mean loss in previous " + str(interval) + " games ", np.mean(losses))
            losses = []


CAPACITY = 30_000
SAMPLE_SIZE = 1_000
HIDDEN = 100
GAMMA = 0.5

REWARD_NO_SCORE: float = 0.5
REWARD_SCORE: float = 10
REWARD_OPPONENT_SCORE: float = -10
REWARD_INVALID_SCORE: float = -1000
REWARD_WIN = 50
REWARD_LOOSE = -50
UPDATE_STEP = 1

boardsize = 3

use_invalid = False
players = [RandomPlayer(1, boardsize), AITrainer(2, boardsize, HIDDEN, REWARD_NO_SCORE, REWARD_SCORE,
                                                 REWARD_OPPONENT_SCORE, REWARD_INVALID_SCORE, use_invalid, SAMPLE_SIZE,
                                                 CAPACITY, GAMMA, limited_batch=False)]
game = GameTraining(players, boardsize)

network_experience(game, 1_000, get_wins=True)

for i in range(20):
    t = time()
    training_cycle(game, 1_000)
    # random_experience(game, 1_000, get_wins=True)

    print("iteration ", i, "   time: ", str(int((time() - t) / 60)) + ": " + str(int(((time() - t) % 60) * 60)))

players[1].network.save_weights()

AI = AIPlayer(3, 3, HIDDEN)
test_players = [players[0], AI]
test_match = Game(test_players, 3)
test_match.play()
print(test_players[0].score, test_players[1].score)

