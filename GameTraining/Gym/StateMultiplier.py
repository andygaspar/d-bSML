import numpy as np


class StateMultiplier:
    @staticmethod
    def rotate(state: np.array, next_state: np.array, action: int):
        left_rotation = [6, 13, 20,
                         2, 9, 16, 23,
                         5, 12, 19,
                         1, 8, 15, 22,
                         4, 11, 18,
                         0, 7, 14, 21,
                         3, 10, 17]
        rotated_state = np.array([state[i] for i in left_rotation])
        # self.print_board(state)
        # self.print_board(rotated_state)
        rotated_action = StateMultiplier.find_index(left_rotation, action)
        # print(action, rotated_action)
        rotated_next_state = np.array([next_state[i] for i in left_rotation])
        # self.print_board(next_state)
        # self.print_board(rotated_next_state)
        return rotated_state, rotated_next_state, rotated_action

    @staticmethod
    def reflect(state: np.array, next_state: np.array, action: int):
        reflection = [21, 22, 23,
                      17, 18, 19, 20,
                      14, 15, 16,
                      10, 11, 12, 13,
                      7, 8, 9,
                      3, 4, 5, 6,
                      0, 1, 2]
        reflected_state = np.array([state[i] for i in reflection])
        # self.print_board(state)
        # self.print_board(reflected_state)
        reflected_action = StateMultiplier.find_index(reflection, action)
        # print(action, reflected_action)
        reflected_next_state = np.array([next_state[i] for i in reflection])
        # self.print_board(next_state)
        # self.print_board(reflected_next_state)
        return reflected_state, reflected_next_state, reflected_action

    @staticmethod
    def find_index(vect: np.array, action: int) -> int:
        ciccio = np.flatnonzero(np.array(vect) == int(action))[0]
        return ciccio
        # find value -> np.flatnonzero(array == value)
        # for i in range(len(vect)):
        #    if vect[i] == action:
        #        return i
