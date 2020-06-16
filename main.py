from Players.AIPlayer import AIPlayer
from Players.randomPlayer import RandomPlayer
from Game.game import Game

boardsize = 3
players = [RandomPlayer(1, boardsize), AIPlayer(2, boardsize)]
g = Game(players, boardsize)
g.play()
