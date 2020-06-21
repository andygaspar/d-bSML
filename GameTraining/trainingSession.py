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

#ReplayMemory Params
SAMPLE_SIZE = 1_000
CAPACITY = 30_000
UPDATE_STEP = 1
HIDDEN = 100
GAMMA = 0.5

REWARD_NO_SCORE: float = 0.5
REWARD_SCORE: float = 10
REWARD_OPPONENT_SCORE: float = -10
REWARD_INVALID_SCORE: float = -1000
REWARD_SCORES_IN_ROW: float = 0
REWARD_WIN = 50
REWARD_LOSE = -50
FIXED_BATCH= False
EPS_GREEDY_VALUE = 1.
SOFTMAX = False


boardsize = 3
only_valid_moves = True

trainer = AITrainer(2, boardsize, HIDDEN, REWARD_NO_SCORE, REWARD_SCORE, REWARD_OPPONENT_SCORE,
                    REWARD_INVALID_SCORE, REWARD_SCORES_IN_ROW, REWARD_WIN, REWARD_LOSE,
                    only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA,
                    fixed_batch=FIXED_BATCH, eps_greedy_value=EPS_GREEDY_VALUE, softmax=SOFTMAX)

players = [RandomPlayer(1, boardsize), trainer]
game = GameTraining(players, boardsize)

network_experience(game, 100, get_wins=True)

for i in range(1):
    t = time()
    training_cycle(game, 100)
    #network_experience(game, 1_000, get_wins=True)

    print("iteration ", i, "   time: ", str(int((time() - t) / 60)) + ": " + str(int(((time() - t) % 60) * 60)))

AI = players[1].get_trained_player(1)

test_players = [players[0], AI]
test_match = Game(test_players, boardsize)
num_games = 1
wins = 0

start = time()
for j in range(10):
    for i in range(num_games):
        test_match.play()
        wins += int(test_players[1].score >= test_players[0].score)
        test_match.reset()

    print("win rate: ", wins / (num_games * (j + 1)))
    print("end: ", time() - start)