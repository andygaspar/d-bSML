from Players.randomPlayer import RandomPlayer
from Players.greedyPlayer import GreedyPlayer
from Players.humanPlayer import HumanPlayer
from Players.AIPlayer import AIPlayer
from Game.game import Game

HIDDEN = 100

boardsize = 3
#players = [RandomPlayer(1, boardsize), AIPlayer(3, 3, HIDDEN)]
players = [RandomPlayer(1, boardsize), GreedyPlayer(3, boardsize)]
g = Game(players, boardsize)
g.play()
