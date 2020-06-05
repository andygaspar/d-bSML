import numpy as np

class Board:

    def __init__(self, N):
        self.vectorBoard=np.zeros((N+1)*N*2)
        self.size = N


        # Human representation of the board, just for printing, in case of human player
        self.rows = np.zeros((N+1,N))
        self.cols = np.zeros((N,N+1))
        

   
    def set_board(self, num):
        self.vectorBoard[num]  = 1

    def count_boxes(self):
        new_num_boxes=0
        N = self.size
        k = 0

        while k < N*(N+1)*2:
            for j in range(N):
                if self.vectorBoard[k] == 1 and self.vectorBoard[k+2*N+1] == 1:
                    if self.vectorBoard[k+N] == 1 and self.vectorBoard[k+N+1] == 1:
                        new_num_boxes+=1
            k += N+2
            
        return new_num_boxes




    # methods just for printing (human conversion)
    def convertToHumanBoard(self):
        k = 0
        N = self.size
        for i in range(N):
            for j in range(N):
                if self.vectorBoard[k] == 0:
                    self.rows[i,j] = 0
                else:
                    self.rows[i,j] = 1
                k += 1

            for j in range(N+1):
                if self.vectorBoard[k] == 0:
                    self.cols[i,j] = 0
                else:
                    self.cols[i,j] = 1
                k += 1
        for j in range(N):
            if self.vectorBoard[k] == 0:
                self.rows[-1,j] = 0
            else:
                self.rows[-1,j] = 1
            k += 1



    def print_board(self):
        self.convertToHumanBoard()
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