from GameTraining.Gym.AItrainer import AITrainer
from Players.AIPlayer import AIPlayer
from Players.randomPlayer import RandomPlayer
from Players.greedyPlayer import GreedyPlayer
from Players.stupidPlayer import StupidPlayer
from GameTraining.gameTraining import GameTraining
from Game.game import Game
from time import time
import numpy as np
from Players.greedyPlayerWithMemory import GreedyPlayerWithMemory

def play_test(tester_player, trained, num_games):
    AI = AIPlayer(2, 3, trained.model_network)
    AI.otherPlayer = tester_player
    test_players = [tester_player, AI]
    test_match = Game(test_players, boardsize)
    wins = 0

    start = time()
    matches = num_games
    invalid = 0

    for i in range(matches):
        result = test_match.play()
        if result == 1:
            invalid += 1
        else:
            wins += int(AI.score >= tester_player.score)
        test_match.reset()
    if matches == 500:
        print("invalids: ", invalid)
        print("win rate: ", wins / matches)
        print("playing time: ", time() - start)

    return wins, invalid

def training_cycle(game: GameTraining,game_memory: GameTraining,  num_games: int):
    max_win_rate = 0
    invalid_min = 1_000_000
    losses = []
    invalids = []
    win_rate = []
    invalid = 0
    losses_means = []
    interval: int = 100
    t = time()
    trainer: AITrainer
    trainer = game.players[1]
    tester_player = game.players[0]

    for i in range(num_games):
        game_memory.play(train=True)
        game_memory.reset()
        invalid += game.play(train=True)
        losses.append(trainer.model_network.loss)
        game.reset()
        if i % interval == 0:
            l_mean = np.mean(losses)
            print(i, "  eps: ", str(trainer.eps_greedy_value)[:4], "  time", int((time() - t) // 60),
                  ":", str(int(((time() - t) % 60)))[:2], "   loss ", l_mean,"  win:",
                  play_test(tester_player, trainer, 50)[0]/50,
                  play_test(GreedyPlayer(0, game.boardsize), trainer, 50)[0]/50, "invalid: ", invalid)
            invalid = 0
            t = time()
            losses_means.append(l_mean)
            losses = []
            trainer.model_network.save_weights("network")
        if i % 500 == 0 and i > 0:
            win_rate_current, niente = play_test(tester_player, trainer, 500)
            win_rate.append(win_rate_current)
            invalids.append(invalid)
            if win_rate_current > max_win_rate and invalid < invalid_min:
                trainer.model_network.save_weights("pesi_rete_allenata")
                max_win_rate = win_rate_current
                invalid_min = invalid
            t = time()
        trainer.update_eps(i)


    from csv import writer
    with open(str(num_games)+"_loss.csv", 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(losses_means)

    with open(str(num_games)+"_win_rate.csv", 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(win_rate)

    with open(str(num_games)+"_invalid.csv", 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(invalids)


SAMPLE_SIZE = 1500 #1024 * 5
CAPACITY = 1_000_000

HIDDEN = 30
GAMMA = 0.5

REWARD_SCORE: float = 0.5
REWARD_INVALID_SCORE: float = -5
REWARD_WIN = 1
REWARD_LOSE = -1
FIXED_BATCH = False
only_valid_moves = False
EPS_MIN: float = 0.1
SOFTMAX = False
NUM_GAMES = 50_000 #50_000
EPS_DECAY: float = 1000
UPDATE_TARGET_EVERY = 20
STUPID_PLAYER_RANDOMNESS = 1

boardsize = 3

trainer = AITrainer(2, boardsize, HIDDEN, REWARD_SCORE, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE,
                    only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, NUM_GAMES, EPS_MIN, EPS_DECAY,
                    fixed_batch=FIXED_BATCH, softmax=SOFTMAX, double_q_interval=UPDATE_TARGET_EVERY)
greedy_memory = GreedyPlayerWithMemory(1, boardsize, HIDDEN, REWARD_SCORE, REWARD_INVALID_SCORE, REWARD_WIN, REWARD_LOSE,
                    only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, NUM_GAMES, EPS_MIN, EPS_DECAY,
                    fixed_batch=FIXED_BATCH, softmax=SOFTMAX, double_q_interval=UPDATE_TARGET_EVERY)

greedy_memory.replayMemory = trainer.replayMemory
random = RandomPlayer(1, boardsize)
# trainer.model_network.load_weights("pesi_rete_allenata_7.pt")
# trainer.target_network.load_weights("pesi_rete_allenata_7.pt")
play_test(greedy_memory, trainer, 100)
game_memory = GameTraining([random, greedy_memory], boardsize)
# play_test(StupidPlayer(1, boardsize, 0.1), trainer, 500)
# play_test(StupidPlayer(1, boardsize, 0.2), trainer, 500)
players = [random, trainer]
#trainer.otherPlayer = players[0]
#trainer.replayMemory.import_memory("Gym/")
game = GameTraining(players, boardsize)

tt = time()
play_test(players[0], players[1], 500)
training_cycle(game, game_memory, NUM_GAMES)
trained_player = trainer.get_trained_player(1)

print("global time: ", str(int((time() - tt) // 60)) + "min " + str(int(((time() - tt) % 60))) + "s")

play_test(players[0], trainer, 1_000)

# trainer.model_network.save_weights("pesi_rete_allenata")