from GameTraining.Gym.AItrainer import AITrainer
from Players.AIPlayer import AIPlayer
from Players.randomPlayer import RandomPlayer
from GameTraining.gameTraining import GameTraining
from Game.game import Game
from time import time
import numpy as np


def play_test(random_player, trained, num_games):
    AI = AIPlayer(2, 3, trained.model_network)
    AI.otherPlayer = random_player
    test_players = [random_player, AI]
    test_match = Game(test_players, boardsize)
    wins = 0

    start = time()
    matches = num_games

    for i in range(matches):
        test_match.play()
        wins += int(AI.score >= random_player.score)
        test_match.reset()

    print("win rate: ", wins / matches)
    print("playing time: ", time() - start)

    return wins


def network_experience(game: GameTraining, num_games: int, get_wins: bool = False):
    wins = []
    for i in range(num_games):
        game.play(train=False)
        if get_wins:
            wins.append(players[1].score >= players[0].score)
        game.reset()
    if get_wins: print("win rate  ", sum(wins) / len(wins))


def training_cycle(game: GameTraining, update_target_every, num_games: int):
    losses = []
    win_rate = []
    losses_means = []
    interval: int = 100
    t = time()
    trainer: AITrainer
    trainer = game.players[1]
    random_player = game.players[0]
    for i in range(num_games):
        game.play(train=True)
        losses.append(trainer.model_network.loss)
        game.reset()
        #if i % 10 == 0:
        #    print(i, trainer.model_network.loss)
        if i % update_target_every == 0:
            trainer.target_network.take_weights(trainer.model_network)
        if i % interval == 0:
            #print("time: ", str(int((time() - t) // 60)) + "min " + str(int(((time() - t) % 60))) + "s"
            #      , "eps: ", trainer.eps_greedy_value, "    iter: ", i)
            t = time()
            l_mean = np.mean(losses)
            print(str(i), "  eps: ", trainer.eps_greedy_value, " Mean loss in previous ", "   loss ", l_mean)
            losses_means.append(l_mean)
            losses = []
            trainer.model_network.save_weights("network")
        if i % 500 == 0:
            win_rate.append(play_test(random_player, trainer, 500))

        trainer.update_eps(i)

    from csv import writer
    with open(str(num_games)+"_loss.csv", 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(losses_means)

    with open(str(num_games)+"_win_rate.csv", 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(win_rate)


SAMPLE_SIZE = 1024 * 5
CAPACITY = 1_000_000

HIDDEN = 50
GAMMA = 1.

REWARD_SCORE: float = 0
REWARD_INVALID_SCORE: float = 0
REWARD_WIN = 1
REWARD_LOSE = -1
FIXED_BATCH = False
only_valid_moves = True
EPS_GREEDY_VALUE = 1.
EPS_MIN: float = 0.01
SOFTMAX = False
NUM_GAMES = 50_000
DECAY: float = 0.001
DOUBLE_Q_LEARNING: bool = True
UPDATE_TARGET_EVERY = 20

boardsize = 3

trainer = AITrainer(2, boardsize, HIDDEN, REWARD_SCORE, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE,
                    only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, NUM_GAMES, EPS_MIN, DECAY,
                    fixed_batch=FIXED_BATCH, softmax=SOFTMAX, double_Q_learning=DOUBLE_Q_LEARNING)

players = [RandomPlayer(1, boardsize), trainer]
trainer.otherPlayer = players[0]
#trainer.replayMemory.import_memory("Gym/")
game = GameTraining(players, boardsize)

tt = time()
training_cycle(game, UPDATE_TARGET_EVERY, NUM_GAMES)
print("global time: ", str(int((time() - tt) // 60)) + "min " + str(int(((time() - tt) % 60))) + "s")

play_test(players[0], trainer, 1_000)

trainer.model_network.save_weights("pesiSimoneVuoleRewardBinaria")