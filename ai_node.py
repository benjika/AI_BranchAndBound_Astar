import numpy as np

initial_state = None
final_state = None
heuristic = ''


class Node:
    initial_state = None
    final_state = None
    heuristic = ''

    def __init__(self, mat, parent, g):
        self.mat = mat
        self.parent = parent
        self.children = []
        self.g = g
        self.h = calculate_h(self.mat)
        self.f = self.g + self.h
        self.status = 'live'
        # live = node that has been generated but whose children have not yet been generated
        # dead = a generated node that is not to be expanded or explored any further

    def __lt__(self, other):

        if self.status == 'live' and other.status == 'dead':
            return True
        if self.status == 'dead' and other.status == 'live':
            return False
        if self.f == other.f:
            return self.mat.tolist() < other.mat.tolist()
        return self.f < other.f

    def __eq__(self, other):
        return (self.mat.flatten() == other.mat.flatten()).all()

    def print_mat(self):
        print('f: {}'.format(self.f))
        print('g: {}'.format(self.g))
        print('h: {}'.format(self.h))
        for j in np.arange(self.mat.shape[0]):
            print(str(self.mat[j]).replace(' ', ' | ').replace('0', ' ').strip('[]'))
            if j < self.mat.shape[0] - 1:
                print('---------')
        print()


def calculate_misplaced_tile(mat1, mat2):
    misplaced_tiles_counter = 0

    for i in np.arange(1, 9):
        if np.where(mat1 == i) != np.where(mat2 == i):
            misplaced_tiles_counter += 1

    return misplaced_tiles_counter


def calculate_manhattan_distance(mat1, mat2):
    xs_mat1 = [np.where(mat1 == i)[1][0] for i in np.arange(1, 9)]
    ys_mat1 = [np.where(mat1 == i)[0][0] for i in np.arange(1, 9)]
    xs_mat2 = [np.where(mat2 == i)[1][0] for i in np.arange(1, 9)]
    ys_mat2 = [np.where(mat2 == i)[0][0] for i in np.arange(1, 9)]

    diffs = [np.abs(x1 - x2) + np.abs(y1 - y2) for x1, x2, y1, y2 in zip(xs_mat1, xs_mat2, ys_mat1, ys_mat2)]
    diffs_sum = sum(diffs)

    return diffs_sum


# find all adjacent neighbours which are reversal
def calculate_reversal_penalty(mat1, mat2):
    reversal_penalty = 0
    nums_list = list(np.arange(1, 9))
    borders = np.arange(3)
    # for each number grater than 0
    for i in nums_list:
        i_y, i_x = np.where(mat1 == i)
        i_y, i_x = i_y[0], i_x[0]
        # find all adjacent
        indices = list(zip([i_x - 1, i_x + 1], [i_y, i_y])) + list(zip([i_x, i_x], [i_y - 1, i_y + 1]))
        for j_x, j_y in indices:
            if j_x in borders and j_y in borders:
                j = mat1[j_y, j_x]
                if np.where(mat1 == i) == np.where(mat2 == j) and np.where(mat1 == j) == np.where(mat2 == i) and j != 0:
                    reversal_penalty += 2

    return reversal_penalty


def calculate_manhattan_distance_plus_reversal_penalty(mat1, mat2):
    manhattan_distance = calculate_manhattan_distance(mat1, mat2)
    reversal_penalty = calculate_reversal_penalty(mat1, mat2)
    return manhattan_distance + reversal_penalty


def calculate_h(mat):
    global final_state

    if Node.final_state is None:
        return 0

    if Node.heuristic == 'misplaced_tile':
        return calculate_misplaced_tile(mat, Node.final_state.mat)
    elif Node.heuristic == 'manhattan_distance':
        return calculate_manhattan_distance(mat, Node.final_state.mat)
    elif Node.heuristic == 'manhattan_distance_plus_reversal_penalty':
        return calculate_manhattan_distance_plus_reversal_penalty(mat, Node.final_state.mat)


def assign_heuristic(heuristic1):
    Node.heuristic = heuristic1


def assign_final_state(node):
    Node.final_state = node


def assign_initial_state(node):
    Node.initial_state = node
