import numpy as np
import bisect
import ai_node

initial_state = None
final_state = None


def assign_heuristic(heuristic1):
    ai_node.assign_heuristic(heuristic1)


def init_initial_state(list1):
    global initial_state
    mat = np.array(list1).reshape((3, 3))
    initial_state = ai_node.Node(mat, None, 0)
    ai_node.assign_initial_state(initial_state)
    return initial_state


def get_initial_state():
    global initial_state
    return initial_state


def assign_final_state(list1):
    global final_state
    mat = np.array(list1).reshape((3, 3))
    final_state = ai_node.Node(mat, None, np.inf)
    ai_node.assign_final_state(final_state)


def assign_final_state_node(node):
    global final_state
    final_state = node
    ai_node.assign_final_state(final_state)


def get_final_state():
    global final_state
    return final_state


def create_path():
    global final_state
    curr_node = final_state

    if final_state.f == np.inf:
        final_state.f = 0
        final_state.g = 0

    best_path = []
    while curr_node is not None:
        best_path.insert(0, curr_node)
        curr_node = curr_node.parent

    return best_path


def print_path(max_iteration, max_depth, algorithm, path, **kwargs):
    global final_state

    curr_node = final_state
    if algorithm == 'a*':
        print('****** A* *****')
        print('num of iterations: {}'.format(max_iteration))
        print('num of levels: {}'.format(max_depth))
    else:
        print('****** B&B *****')
        print('num of iterations: {}'.format(max_iteration))
        print('num of max level: {}'.format(max_depth))
        print('num of iterations until solution: {}'.format(kwargs['solution_iterations']))
        print('num of levels until solution: {}'.format(kwargs['solution_depth']))

    for node in path:
        node.print_mat()


def get_empty_location(mat):
    empty_loc = np.where(mat == 0)
    return empty_loc[0][0], empty_loc[1][0]


# take parent's matrix, swap position and create a new node with this position
def create_potential_child(x_parent, y_parent, x_child, y_child, present_node):
    pot_mat = np.copy(present_node.mat)
    pot_mat[y_parent, x_parent], pot_mat[y_child, x_child] = pot_mat[y_child, x_child], pot_mat[y_parent, x_parent]
    return ai_node.Node(pot_mat, present_node, present_node.g + 1)


def bisect_index(node, list1):
    i = bisect.bisect_left(list1, node)
    if i != len(list1) and list1[i] == node:
        return i
    return -1


def get_children(node):
    empty_y, empty_x = get_empty_location(node.mat)  # get the coordinates of the empty tile
    potential_next_empty_list = list(zip([empty_y + 1, empty_y - 1], [empty_x, empty_x])) + \
                                list(zip([empty_y, empty_y], [empty_x + 1, empty_x - 1]))
    borders = np.arange(3)

    if node.parent is not None:
        parent_empty = get_empty_location(node.parent.mat)
        potential_next_empty_list = [pot_next for pot_next in potential_next_empty_list if pot_next != parent_empty]

    children = []
    for pot_next in potential_next_empty_list:
        if pot_next[0] in borders and pot_next[1] in borders:
            children.append(create_potential_child(empty_x, empty_y, pot_next[1], pot_next[0], node))

    return children


def is_solvable(list_input):
    list1 = list_input.copy()
    list1.remove(0)
    inversions_counter = 0
    for i in np.arange(1, 8):
        for j in np.arange(i):
            if list1[i] < list1[j]:
                inversions_counter += 1

    return inversions_counter % 2 == 0


def is_legit_list(list1):
    for i in np.arange(9):
        if not list1.count(i) == 1:
            return False
    return True
