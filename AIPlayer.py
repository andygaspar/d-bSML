from player import Player

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import torch
from torch import nn, optim
from IPython import display

class RandomPlayer(Player):

	def __init__(self, id: int, boardsize: int):
		super().__init__(id, boardsize)
		np.random.seed(2)

	def get_move(self, board):
		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		# print('Device: {}'.format(device))
		torch.manual_seed(41)
		n = self.boardsize
		N = (2 * n + 2) * n  # num samples = classes
		D = 1    # dimensions
		H = 100  # num hidden units
		epochs = 1000

		X = torch.zeros(N, D)
		y = torch.zeros(N, dtype=torch.long)

		model = nn.Sequential(
			nn.Linear(D, H),
			nn.ReLU(),
			nn.Linear(H, N))
		#model.to(device)
		criterion = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
		optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

		for e in range(epochs):
			y_pred = model(X)
			loss = criterion
			print("[EPOCH]: {}, [LOSS]: {}".format(e, loss.item()))
			display.clear_output(wait=True)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		validMoves = np.flatnonzero(board.vectorBoard == 0)
		return np.random.choice(validMoves)

	def scored(self):
		self.score += 1
		print("bravo, hai fatto punto")
