import numpy as np

class Board:

    def __init__(self, N):
        self.vectorBoard=np.zeros((2*N+1)*N)
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

        while k < len(self.vectorBoard)-2*N-1:
            for j in range(N):
                print(k,k+2*N+1)
                if self.vectorBoard[k] == 1 and self.vectorBoard[k+2*N+1] == 1:
                    if self.vectorBoard[k+N] == 1 and self.vectorBoard[k+N+1] == 1:
                        new_num_boxes+=1
                k += 1
            k += N+2
        print(len(self.vectorBoard))
        return new_num_boxes




    def print_board(self):

        k = 0
        N = self.size
        for i in range(N):
            orizontal=""
            vertical=""
            for j in range(N):
                
                if self.vectorBoard[k] == 0:
                    orizontal+="  _ "
                else:
                    orizontal+="  * "
                k += 1

            for j in range(N+1):
                if self.vectorBoard[k] == 0:
                    vertical+="|   "
                else:
                    vertical+="*   "
                k += 1
            print(orizontal)
            print(vertical)

        for j in range(N):
            if self.vectorBoard[k] == 0:
                orizontal+="  _ "
            else:
                orizontal+="  * "
            k += 1
