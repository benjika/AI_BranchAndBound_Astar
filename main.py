import numpy as np
import timeit
import csv
import random
import pandas as pd

np.random.RandomState(0)
final_state_n = None
initial_state_n = None
final_state = None
heuristic = 'misplaced_tile'
is_found_solution = False


def calculate_misplaced_tile(mat1, mat2):
    list1, list2 = mat1.flatten(), mat2.flatten()
    list1, list2 = np.delete(list1, np.where(list1 == 0)), np.delete(list2, np.where(list2 == 0))
    diffs = np.absolute(list1 - list2)
    count_diffs = (diffs > 0).sum()
    return count_diffs


def calculate_manhattan_distance(mat1, mat2):
    xs_mat1 = [np.where(mat1 == i)[1][0] for i in np.arange(1, 9)]
    ys_mat1 = [np.where(mat1 == i)[0][0] for i in np.arange(1, 9)]
    xs_mat2 = [np.where(mat2 == i)[1][0] for i in np.arange(1, 9)]
    ys_mat2 = [np.where(mat2 == i)[0][0] for i in np.arange(1, 9)]
    difs = [np.abs(x1 - x2) + np.abs(y1 - y2) for x1, x2, y1, y2 in zip(xs_mat1, xs_mat2, ys_mat1, ys_mat2)]
    difs_sum = sum(difs)
    return difs_sum


class Graph:
    def __init__(self, initial_node):
        self.initial_node = initial_node
        self.live_list = [initial_node]
        self.dead_list = []
        self.upper_bound = np.inf
        self.best_path = []

    def create_best_path(self):
        global final_state_n
        curr_node = final_state_n
        if final_state_n.c == np.inf:
            final_state_n.c = 0
            final_state_n.f = 0

        self.best_path = []
        while curr_node is not None:
            self.best_path.insert(0, curr_node)
            curr_node = curr_node.parent

    def print_best_path(self, max_iteration, max_depth):
        print('****** B&B *****')
        print('num of iterations: {}'.format(max_iteration))
        print('num of levels: {}'.format(max_depth))
        for node in self.best_path:
            node.print_mat()


class Node:
    def __init__(self, mat, parent, prev_direction, f):
        self.mat = mat
        self.parent = parent
        self.children = []
        self.prev_direction = prev_direction
        self.f = f
        self.h = self.calculate_h()
        self.c = self.f + self.h
        self.status = 'live'
        # live = node that has been generated but whose children have not yet been generated
        # dead = a generated node that is not to be expanded or explored any further

    def print_mat(self):
        print('c: {}'.format(self.c))
        print('f: {}'.format(self.f))
        print('h: {}'.format(self.h))
        for j in np.arange(self.mat.shape[0]):
            print(str(self.mat[j]).replace(' ', ' | ').replace('0', ' ').strip('[]'))
            if j < self.mat.shape[0] - 1:
                print('---------')
        print()

    def calculate_h(self):
        global final_state
        global heuristic

        if heuristic == 'misplaced_tile':
            return calculate_misplaced_tile(self.mat, final_state)
        elif heuristic == 'manhattan_distance':
            return calculate_manhattan_distance(self.mat, final_state)


def a_star_print_path(max_iteration, max_depth):
    global final_state_n

    curr_node = final_state_n
    path = []
    print('****** A* *****')
    print('num of iterations: {}'.format(max_iteration))
    print('num of levels: {}'.format(max_depth))

    while curr_node is not None:
        path.insert(0, curr_node)
        curr_node = curr_node.parent

    for node in path:
        node.print_mat()


def get_empty_location(mat):
    empty_loc = np.where(mat == 0)
    return empty_loc[0][0], empty_loc[1][0]


# take parent's matrix, swap position and create a new node with this position
def create_potential_child(x, y, del_x, del_y, direction, present_node):
    pot_mat = np.copy(present_node.mat)
    pot_mat[y + del_y, x + del_x], pot_mat[y, x] = pot_mat[y, x], pot_mat[y + del_y, x + del_x]
    return Node(pot_mat, present_node, direction, present_node.f + 1)


# get possible directions for current node
# we exclude position out of bounds and parent's position
def get_potential_directions(prev_direction, mat_h, mat_w, empty_x, empty_y):
    direction_limits = [True if not direction == prev_direction else False for direction in
                        ['DOWN', 'UP', 'RIGHT', 'LEFT']]
    out_of_bounds_limits = [empty_y > 0, empty_y < mat_h - 1, empty_x > 0, empty_x < mat_w - 1]
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    delta_xs = [0, 0, -1, 1]
    delta_ys = [-1, 1, 0, 0]
    possible_directions = []
    for direction_limit, out_of_bounds_limit, i in zip(direction_limits, out_of_bounds_limits, np.arange(4)):
        # if the current position is not parent's and not out of bounds
        if direction_limit and out_of_bounds_limit:
            possible_directions.append((directions[i], delta_xs[i], delta_ys[i]))
    return possible_directions


# get the index of the lowest cost in the list
def get_least_cost_index(list):
    min_cost = min([node.c for node in list])  # get the lowest cost in the list
    min_index = [node.c for node in list].index(min_cost)  # get the node index with the lowest cost in the open list
    return min_index


# get the index of the lowest cost in the list
def get_least_cost_live_index(list):
    min_cost = min([node.c for node in list if node.status == 'live'])  # get the live lowest cost in the list
    for index in np.arange(len(list)):
        if list[index].status == 'live' and list[index].c == min_cost:
            return index
    return 0


def branch_and_bound(iterations_limit, depth_limit, i):
    global initial_state_n
    global final_state_n

    graph = Graph(initial_state_n)
    current_node = initial_state_n
    is_solved = False
    dead_list = []

    if initial_state_n.h == 0:
        current_node = None
        graph.create_best_path()
        is_solved = True

    curr_iteration, max_depth = 0, 0
    mat_h, mat_w = initial_state_n.mat.shape

    while current_node is not None and curr_iteration <= iterations_limit:
        curr_iteration = curr_iteration + 1
        max_depth = max([max_depth, current_node.f])  # increase depth if necessary
        print('bnb {} curr_iteration: {},max_depth: {}'.format(i, curr_iteration, max_depth))

        # if we found a new goal node
        if current_node.h == 0:
            # if we found a better path to goal node
            if graph.upper_bound > current_node.c:
                is_solved = True
                graph.upper_bound = current_node.c
                final_state_n = current_node
                graph.create_best_path()
            current_node.status = 'dead'
            #dead_list.append(current_node.mat.tolist())
            current_node = current_node.parent

        # if the current node haven't been explored yet
        elif len(current_node.children) == 0:

            # here is no better solution than what found in this node or it's children
            if current_node.c > graph.upper_bound or current_node.f == depth_limit:
                current_node.status = 'dead'
                #dead_list.append(current_node.mat.tolist())
                current_node = current_node.parent

            # create children node and choose the cheapest for the next iteration
            else:
                empty_y, empty_x = get_empty_location(current_node.mat)  # get the coordinates of the empty tile
                prev_direction = current_node.prev_direction  # get the direction of the parent
                potential_directions = get_potential_directions(prev_direction, mat_h, mat_w, empty_x, empty_y)
                for pot_direction, delta_x, delta_y in potential_directions:
                    child = create_potential_child(empty_x, empty_y, delta_x, delta_y, pot_direction, current_node)
                    current_node.children.append(child)
                    #if child.mat.tolist() in dead_list:
                    #    child.status = 'dead'
                if len([node for node in current_node.children if node.status == 'live']) > 0:
                    min_cost_children = get_least_cost_live_index(current_node.children)
                    current_node = current_node.children[min_cost_children]
                else:
                    current_node.status = 'dead'
                    #dead_list.append(current_node.mat.tolist())
                    current_node = current_node.parent

        # if there are live children choose the best alive
        elif len([node for node in current_node.children if node.status == 'live']) > 0:
            min_cost_children = get_least_cost_live_index(current_node.children)
            current_node = current_node.children[min_cost_children]

        # if all children are dead no reason to explore farther, back to parent
        else:
            current_node.status = 'dead'
            #dead_list.append(current_node.mat.tolist())
            current_node = current_node.parent

    """if curr_iteration > iterations_limit and not is_solved:
        # return 'failed iteration limit'
        print('Branch And Bound passed iterations limit. No solution found in {} iteration.'.format(iterations_limit))
    elif max_depth > depth_limit and not is_solved:
        # return 'failed depth limit'
        print('A* passed depth limit. No solution found in {} levels.'.format(depth_limit))
    else:
        # graph.print_best_path(curr_iteration, max_depth)"""
    return curr_iteration, max_depth, final_state_n.c


def a_star(iterations_limit, depth_limit, i):
    global initial_state_n
    global final_state_n

    open_list, close_list = [initial_state_n], []
    # if the initial state is also the final state
    if initial_state_n.h == 0:
        open_list = []

    curr_iteration, max_depth = 0, 0
    mat_h, mat_w = initial_state_n.mat.shape

    while len(open_list) > 0:  # and curr_iteration <= iterations_limit and max_depth <= depth_limit:
        curr_iteration = curr_iteration + 1
        min_index = get_least_cost_index(open_list)  # get the index of the lowest cost in the open list
        current_node = open_list.pop(min_index)  # pop the node with the lowest cost from the open list
        max_depth = max([max_depth, current_node.f + 1])  # increase depth if necessary
        print('a* {} curr_iteration: {},max_depth: {}'.format(i, curr_iteration, max_depth))
        empty_y, empty_x = get_empty_location(current_node.mat)  # get the coordinates of the empty tile
        prev_direction = current_node.prev_direction  # get the direction of the parent
        potential_directions = get_potential_directions(prev_direction, mat_h, mat_w, empty_x, empty_y)
        for pot_direction, delta_x, delta_y in potential_directions:
            potential_child = create_potential_child(empty_x, empty_y, delta_x, delta_y, pot_direction,
                                                     current_node)
            # found goal
            if potential_child.h == 0:
                final_state_n = potential_child
                open_list = []
                max_depth = max([max_depth, potential_child.f])
                break
            # there a node with the same matrix but better cost in the open list
            elif potential_child.mat.tolist() in [node.mat.tolist() for node in open_list]:
                # (np.array_equal(node.mat, potential_child.mat) and node.c <= potential_child.c)]) > 0:
                continue
            # there a node with the same matrix but better cost in the closed list
            elif potential_child.mat.tolist() in [node.mat.tolist() for node in close_list]:
                # (np.array_equal(node.mat, potential_child.mat) and node.c <= potential_child.c)]) > 0:
                continue
            else:
                # open_list = [node for node in open_list if not np.array_equal(node.mat, potential_child.mat)]
                # close_list = [node for node in close_list if not np.array_equal(node.mat, potential_child.mat)]
                open_list.append(potential_child)
        close_list.append(current_node)
    """
    if curr_iteration > iterations_limit:
        return 'failed iteration limit'
        # print('A* passed iterations limit. No solution found in {} iteration.'.format(iterations_limit))
    elif max_depth > depth_limit:
        # print('A* passed depth limit. No solution found in {} levels.'.format(depth_limit))
        return 'failed depth limit'
    else:"""
    # a_star_print_path(curr_iteration, max_depth)
    return curr_iteration, max_depth


def is_solvable(list):
    # index = [list.index(i) for i in np.arange(9)]
    list1 = list.copy()
    list1.remove(0)
    inversions_counter = 0
    for i in np.arange(1, 8):
        for j in np.arange(i):
            if list1[i] < list1[j]:
                inversions_counter += 1

    return inversions_counter % 2 == 0


def is_legit_list(list):
    for i in np.arange(9):
        if not list.count(i) == 1:
            return False
    return True


def test_runtime():
    global initial_state_n
    global heuristic
    global is_found_solution
    global final_state
    global final_state_n

    iterations_limit = 150000
    depth_limit = 100
    mats = []
    template = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    final_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]).reshape((3, 3))
    final_state_n = Node(final_state, None, None, np.inf)
    while len(mats) < 20:
        random.shuffle(template)
        if is_solvable(template) and template not in mats:
            mats.append(template.copy())

    df = pd.DataFrame()
    df['Matrix'] = mats
    df.to_csv(r'mats.csv', index=False, header=True)

    results = []
    for mat, i in zip(mats, np.arange(len(mats))):
        initial_mat = np.array(mat).reshape((3, 3))
        initial_state_n = Node(initial_mat, None, None, 0)
        heuristic = 'misplaced_tile'
        print('a_star_misplaced_tile')
        a_star_misplaced_tile = a_star(iterations_limit, depth_limit, i)
        initial_state_n = Node(initial_mat, None, None, 0)
        print('bnb_misplaced_tile')
        bnb_misplaced_tile = branch_and_bound(iterations_limit, depth_limit, i)
        heuristic = 'manhattan_distance'
        initial_state_n = Node(initial_mat, None, None, 0)
        print('a_star_manhattan_distance')
        a_star_manhattan_distance = a_star(iterations_limit, depth_limit, i)
        initial_state_n = Node(initial_mat, None, None, 0)
        print('bnb_manhattan_distance')
        bnb_manhattan_distance = branch_and_bound(iterations_limit, depth_limit, i)
        results.append(
            (mat, a_star_misplaced_tile, bnb_misplaced_tile, a_star_manhattan_distance, bnb_manhattan_distance))
        print(i)

    with open('results.csv', 'w') as f:
        write = csv.writer(f)

        # write.writerow(['matrix', 'bnb_misplaced_tile', 'bnb_manhattan_distance'])
        write.writerow(['matrix', 'a_star_misplaced_tile', 'bnb_misplaced_tile', 'a_star_manhattan_distance',
                        'bnb_manhattan_distance'])
        write.writerows(results)


def assign_heuristic():
    global heuristic
    print('Choose heuristic:')
    print('For misplaced tile press 1')
    print('For Manhattan press 2')
    input1 = input("Enter heuristic")
    print()
    if input1 == '1':
        heuristic = 'misplaced_tile'
    elif input1 == '2':
        heuristic = 'manhattan_distance'
    else:
        print('Wrong input was typed')
        heuristic = 'misplaced_tile'
    print("Heuristic is: " + heuristic)
    print()


def main():
    pass
    """
    global final_state_n
    global final_state
    global initial_state_n

    iterations_limit = 150000
    depth_limit = 100
    final_list = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    initial_list = [4, 8, 2, 3, 0, 5, 1, 6, 7]
    # initial_list = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    if is_legit_list(initial_list):
        if is_solvable(initial_list):
            final_state = np.array(final_list).reshape((3, 3))
            final_state_n = Node(final_state, None, None, np.inf)
            initial_state = np.array(initial_list).reshape((3, 3))
            initial_state_n = Node(initial_state, None, None, 0)
            assign_heuristic()
            input1 = input("For Branch&Bound press 1. For A* press 2")
            print()
            if input1 == '1':
                branch_and_bound(iterations_limit, depth_limit, "")
            else:
                a_star(iterations_limit, depth_limit, '')
        else:
            print('The matrix is not solvable ')
    else:
        print('The matrix is not legit')
    """


test_runtime()

if __name__ == '__main__':
    main()
