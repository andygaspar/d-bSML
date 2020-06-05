import numpy as np

class Board:

    def __init__(self, N):
        self.rows = np.zeros((N+1,N))
        self.cols = np.zeros((N,N+1))
        self.size = N

    def set_board(self, is_row, r, c):
        if is_row == True:
            self.rows[r,c]=1
        else:
            self.cols[r,c]=1

    def print_board(self):
        for i in range(self.size):
            row=""
            col=""
            for j in range(self.size):
                if self.rows[i,j] == 0:
                    row+="  _ "
                else:
                    row+="  * "
            for j in range(self.size+1):
                if self.cols[i,j] == 0:
                    col+="|   "
                else:
                    col+="*   "
            print(row)
            print(col)

        row=""
        for j in range(self.size):
            if self.rows[self.size,j] == 0:
                row+="  _ "
            else:
                row+="  * "
        print(row,"\n\n")