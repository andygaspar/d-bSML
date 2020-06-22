from GameTraining.Gym.AItrainer import AITrainer
from Players.randomPlayer import RandomPlayer
from GameTraining.gameTraining import GameTraining
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
    if get_wins : print("win rate  ", sum(wins) / len(wins))


def training_cycle(game: GameTraining, num_games: int, double_q_learning: bool):
    losses = []
    interval: int = 100
    t = time()
    for i in range(num_games):
        game.play(train=True)
        losses.append(game.players[1].model_network.loss)
        game.reset()
        if i % interval == 0:
            print("time: ", str(int((time() - t) / 60)) + "min " + str(int((time() - t) % 60)) + "s"
                  , "    ieter: ", i)
            t=time()
            print("Mean loss in previous " + str(interval) + " games ", np.mean(losses))
            losses = []
            if double_q_learning:
                game.players[1].update_target_network()
        game.players[1].update_eps(i)

#ReplayMemory Params
SAMPLE_SIZE = 300
CAPACITY = 1_000

#
HIDDEN = 50
GAMMA = 0.9

REWARD_NO_SCORE: float = 0
REWARD_SCORE: float = 10
REWARD_OPPONENT_SCORE: float = -5
REWARD_INVALID_SCORE: float = -1000
REWARD_SCORES_IN_ROW: float = 0
REWARD_WIN = 50
REWARD_LOSE = -50
FIXED_BATCH = False
only_valid_moves = True
EPS_GREEDY_VALUE = 0.1
SOFTMAX = True
NUM_GAMES = 1_000
EPS_MIN: float = 0.01
DECAY: float = 0.0001
DOUBLE_Q_LEARNING: bool = True

boardsize = 3

trainer = AITrainer(2, boardsize, HIDDEN, REWARD_NO_SCORE, REWARD_SCORE, REWARD_OPPONENT_SCORE,
                    REWARD_INVALID_SCORE, REWARD_SCORES_IN_ROW, REWARD_WIN, REWARD_LOSE,
                    only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, NUM_GAMES, EPS_MIN, DECAY,
                    fixed_batch=FIXED_BATCH, eps_greedy_value=EPS_GREEDY_VALUE, softmax=SOFTMAX,
                    double_Q_learning=DOUBLE_Q_LEARNING)

players = [RandomPlayer(1, boardsize), trainer]
trainer.replayMemory.import_memory("Gym/")
game = GameTraining(players, boardsize)



tt = time()
training_cycle(game, NUM_GAMES, DOUBLE_Q_LEARNING)
print("global time: ", str(int((time() - tt) / 60)) + "min " + str(int((time() - tt) % 60)) + "s")

AI = players[1].get_trained_player(1)

test_players = [players[0], AI]
test_match = Game(test_players, boardsize)
wins = 0

test_plays = 1_000
start = time()

for i in range(test_plays):
   test_match.play()
   wins += int(test_players[1].score >= test_players[0].score)
   test_match.reset()

print("win rate: ", wins / test_plays)
print("playing time: ", str(int((time() - start) / 60)) + "min " + str(int((time() - start) % 60)) + "s")

#network_experience(game, 3_000)