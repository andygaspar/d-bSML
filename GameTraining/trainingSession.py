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


boardsize = 3
only_valid_moves = True

trainer = AITrainer(2, boardsize, HIDDEN, REWARD_NO_SCORE, REWARD_SCORE, REWARD_OPPONENT_SCORE,
                    REWARD_INVALID_SCORE, REWARD_SCORES_IN_ROW, REWARD_WIN, REWARD_LOSE,
                    only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, UPDATE_STEP,
                    fixed_batch=False, eps_greedy_value=1, softmax=False)

players = [RandomPlayer(1, boardsize), trainer]
#game = GameTraining(players, boardsize)

#network_experience(game, 1_000, get_wins=True)

#for i in range(20):
#    t = time()
#    training_cycle(game, 1_000)
#     random_experience(game, 1_000, get_wins=True)

#    print("iteration ", i, "   time: ", str(int((time() - t) / 60)) + ": " + str(int(((time() - t) % 60) * 60)))

# players[1].network.save_weights()

AI = AIPlayer(3, boardsize, HIDDEN)
test_players = [players[0], RandomPlayer(2, boardsize)]
test_match = Game(test_players, boardsize)
num_games = 10_000
wins = 0

start = time()
for j in range(10):
    for i in range(num_games):
        test_match.play()
        wins += int(test_players[1].score >= test_players[0].score)
        test_match.reset()

    print("win rate: ", wins / (num_games * (j + 1)))
    #print("end: ", time() - start)