from GameTraining.Gym.AItrainer import AITrainer
from Players.AIPlayer import AIPlayer
from Players.randomPlayer import RandomPlayer
from Players.greedyPlayer import GreedyPlayer
from GameTraining.gameTraining import GameTraining
from Game.game import Game
from time import time
import numpy as np


def play_test(player_tester, trained, num_games):
    AI = AIPlayer(2, 3, trained.model_network)
    AI.otherPlayer = player_tester
    test_players = [player_tester, AI]
    test_match = Game(test_players, boardsize)
    wins = 0

    start = time()
    matches = num_games

    for i in range(matches):
        test_match.play()
        wins += int(AI.score >= player_tester.score)
        test_match.reset()

    print("win rate: ", wins / matches)
    print("playing time: ", time() - start)

    return wins

def training_cycle(game: GameTraining, num_games: int):
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
        if i % interval == 0:
            #print("time: ", str(int((time() - t) // 60)) + "min " + str(int(((time() - t) % 60))) + "s"
            #      , "eps: ", trainer.eps_greedy_value, "    iter: ", i)
            t = time()
            l_mean = np.mean(losses)
            print(str(i), "  eps: ", trainer.eps_greedy_value, " Mean loss in previous ", "   loss ", l_mean)
            losses_means.append(l_mean)
            losses = []
            trainer.model_network.save_weights("network")
        if i % 200 == 0:
            win_rate.append(play_test(random_player, trainer, 500))

        trainer.update_eps(i)

    from csv import writer
    with open(str(num_games)+"_loss.csv", 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(losses_means)

    with open(str(num_games)+"_win_rate.csv", 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(win_rate)


SAMPLE_SIZE = 100 #1024 * 5
CAPACITY = 1_000_000

HIDDEN = 30
GAMMA = 1.

REWARD_SCORE: float = 0
REWARD_INVALID_SCORE: float = 0
REWARD_WIN = 1
REWARD_LOSE = -1
FIXED_BATCH = False
only_valid_moves = True
EPS_MIN: float = 0.01
SOFTMAX = False
NUM_GAMES = 5_000 #50_000
EPS_DECAY: float = 0.001
UPDATE_TARGET_EVERY = 20

boardsize = 3

trainer = AITrainer(2, boardsize, HIDDEN, REWARD_SCORE, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE,
                    only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, NUM_GAMES, EPS_MIN, EPS_DECAY,
                    fixed_batch=FIXED_BATCH, softmax=SOFTMAX, double_q_interval=UPDATE_TARGET_EVERY)

players = [GreedyPlayer(1, boardsize), trainer]
trainer.otherPlayer = players[0]
#trainer.replayMemory.import_memory("Gym/")
game = GameTraining(players, boardsize)

tt = time()
training_cycle(game, NUM_GAMES)
trained_player = trainer.get_trained_player(1)

print("global time: ", str(int((time() - tt) // 60)) + "min " + str(int(((time() - tt) % 60))) + "s")

play_test(players[0], trainer, 1_000)

trainer.model_network.save_weights("pesi_rete_allenata")