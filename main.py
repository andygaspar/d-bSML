from Players.AIPlayer import AIPlayer
from Players.randomPlayer import RandomPlayer
from Game.game import Game

HIDDEN = 100

boardsize = 3
players = [RandomPlayer(1, boardsize), AIPlayer(3, 3, HIDDEN)]
g = Game(players, boardsize)
g.play()
