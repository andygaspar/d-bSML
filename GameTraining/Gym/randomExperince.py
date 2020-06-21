from GameTraining.gameTraining import GameTraining
from GameTraining.Gym.initalTrainer import InitialTrainer

SAMPLE_SIZE = 20
CAPACITY = 3_000
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
FIXED_BATCH = False
EPS_GREEDY_VALUE = 1.
SOFTMAX = False
NUM_GAMES = 1_000
EPS_MIN: float = 0.01
DECAY: float = 0.001
only_valid_moves = True





def inital_memory_genereator(size: int):
    SAMPLE_SIZE = 300_000
    CAPACITY = 300_000
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
    FIXED_BATCH = False
    EPS_GREEDY_VALUE = 1.
    SOFTMAX = False
    NUM_GAMES = 1_000
    EPS_MIN: float = 0.01
    DECAY: float = 0.001
    only_valid_moves = True
    boardsize = 3

    player1 = InitialTrainer(2, boardsize, HIDDEN, REWARD_NO_SCORE, REWARD_SCORE, REWARD_OPPONENT_SCORE,
                             REWARD_INVALID_SCORE, REWARD_SCORES_IN_ROW, REWARD_WIN, REWARD_LOSE,
                             only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, NUM_GAMES, EPS_MIN, DECAY,
                             fixed_batch=FIXED_BATCH, eps_greedy_value=EPS_GREEDY_VALUE, softmax=SOFTMAX)

    player2 = InitialTrainer(2, boardsize, HIDDEN, REWARD_NO_SCORE, REWARD_SCORE, REWARD_OPPONENT_SCORE,
                             REWARD_INVALID_SCORE, REWARD_SCORES_IN_ROW, REWARD_WIN, REWARD_LOSE,
                             only_valid_moves, SAMPLE_SIZE, CAPACITY, GAMMA, NUM_GAMES, EPS_MIN, DECAY,
                             fixed_batch=FIXED_BATCH, eps_greedy_value=EPS_GREEDY_VALUE, softmax=SOFTMAX)

    players = [player1, player2]

    # game = GameTraining(players, boardsize)
    # for i in range(size):
    #     game.play(train=False)
    #     game.reset()
    #     if i%1000 == 0: print("partite giocate ", i)
    # print(player1.replayMemory.actions)
    player1.replayMemory.import_memory()
    print("ciccio")


    # print(player2.replayMemory.nextStates)
    # player1.replayMemory.export_memory()
    # player2.replayMemory.export_memory()


import os

inital_memory_genereator(1)
print(os.getcwd())

