from GameTraining.Gym.AItrainer import AITrainer
from Players.AIPlayer import AIPlayer
from Players.randomPlayer import RandomPlayer
from Players.greedyPlayer import GreedyPlayer
from Players.stupidPlayer import StupidPlayer
from GameTraining.gameTraining import GameTraining
from Game.game import Game
from time import time
import numpy as np


def play_test(tester_player, trained, num_games):
    AI = AIPlayer(2, 3, trained.model_network)
    AI.otherPlayer = tester_player
    test_players = [tester_player, AI]
    test_match = Game(test_players, boardsize)
    wins = 0

    start = time()
    matches = num_games

    for i in range(matches):
        test_match.play()
        wins += int(AI.score >= tester_player.score)
        test_match.reset()

    print("win rate: ", wins / matches)
    print("playing time: ", time() - start)

    return wins

def training_cycle(game: GameTraining, num_games: int):
    max_win_rate = 0
    losses = []
    win_rate = []
    losses_means = []
    interval: int = 100
    t = time()
    trainer: AITrainer
    trainer = game.players[1]
    tester_player = game.players[0]
    for i in range(num_games):
        game.play(train=True)
        losses.append(trainer.model_network.loss)
        game.reset()
        if i % interval == 0:
            l_mean = np.mean(losses)
            print(i, "  eps: ", str(trainer.eps_greedy_value)[:4], "  time", int((time() - t) // 60),
                  ":", str(int(((time() - t) % 60)))[:2], "   loss ", l_mean)
            t = time()
            losses_means.append(l_mean)
            losses = []
            trainer.model_network.save_weights("network")
        if i % 500 == 0 and i > 0:
            win_rate_current = play_test(tester_player, trainer, 500)
            win_rate.append(win_rate_current)
            if win_rate_current > max_win_rate:
                trainer.model_network.save_weights("pesi_rete_allenata")
                max_win_rate = win_rate_current
            t = time()
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

REWARD_SCORE: float = 0.5
REWARD_INVALID_SCORE: float = 0
REWARD_WIN = 0
REWARD_LOSE = 0
FIXED_BATCH = False
only_valid_moves = True
EPS_MIN: float = 0.01
SOFTMAX = False
NUM_GAMES = 50_000 #50_000
EPS_DECAY: float = 0.001
UPDATE_TARGET_EVERY = 20
STUPID_PLAYER_RANDOMNESS = 0.3

boardsize = 3

trainer = AITrainer(2, boardsize, HIDDEN, REWARD_SCORE, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE,
                    only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, NUM_GAMES, EPS_MIN, EPS_DECAY,
                    fixed_batch=FIXED_BATCH, softmax=SOFTMAX, double_q_interval=UPDATE_TARGET_EVERY)

trainer.model_network.load_weights("pesi_rete_allenata_8.pt")
trainer.target_network.load_weights("pesi_rete_allenata_8.pt")
play_test(StupidPlayer(1, boardsize, 0), trainer, 1)
play_test(StupidPlayer(1, boardsize, 0.1), trainer, 500)
play_test(StupidPlayer(1, boardsize, 0.2), trainer, 500)
players = [StupidPlayer(1, boardsize, STUPID_PLAYER_RANDOMNESS), trainer]
trainer.otherPlayer = players[0]
#trainer.replayMemory.import_memory("Gym/")
game = GameTraining(players, boardsize)

tt = time()
play_test(players[0], players[1], 500)
training_cycle(game, NUM_GAMES)
trained_player = trainer.get_trained_player(1)

print("global time: ", str(int((time() - tt) // 60)) + "min " + str(int(((time() - tt) % 60))) + "s")

play_test(players[0], trainer, 1_000)

# trainer.model_network.save_weights("pesi_rete_allenata")