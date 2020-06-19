from GameTraining.Gym.AItrainer import AITrainer
from GameTraining.gameTraining import GameTraining
from Players.randomPlayer import RandomPlayer
from time import time


def random_experience(game: GameTraining, num_games: int, get_wins: bool = False):
    wins = []
    for i in range(num_games):
        game.play(train=False)
        if get_wins:
            wins.append(players[1].score >= players[0].score)
        game.reset()
    print(sum(wins) / len (wins))


def training_cycle(game: GameTraining, num_games: int):
    for i in range(num_games):
        game.play(train=True)
        if i%10 == 0:
            print(game.players[1].network.loss)
        game.reset()


CAPACITY = 30_000
SAMPLE_SIZE = 500
HIDDEN = 100
EPOCHS = 1000
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

t = time()
for i in range(20):
    random_experience(game, 1_000, get_wins=True)
    training_cycle(game, 1_000)
    #random_experience(game, 1_000, get_wins=True)

    print("iteration ", i, "   time: ", (time() - t) / 60)

