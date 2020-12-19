import numpy as np
import csv
import random
import pandas as pd
import ai_utils as utils
import bisect


def branch_and_bound(iterations_limit, depth_limit, i):
    current_node = utils.get_initial_state()
    upper_bound, solution_iteration = np.inf, 0
    is_solved = False
    dead_list = []
    path = []

    if current_node.h == 0:
        utils.assign_final_state_node(current_node)
        path = utils.create_path()
        current_node = None
        is_solved = True

    curr_iteration, max_depth, iterations_until_solution = 0, 0, 0

    while current_node is not None:
        max_depth = max([max_depth, current_node.g])  # increase depth if necessary
        print('bnb {} curr_iteration: {},max_depth: {}, curr_depth: {}'.format(i, curr_iteration, max_depth,
                                                                               current_node.g))

        # if we found a new goal node
        if current_node.h == 0:
            # if we found a better path to goal node
            if upper_bound > current_node.f:
                is_solved = True
                upper_bound = current_node.f
                solution_iteration = curr_iteration
                utils.assign_final_state_node(current_node)
                path = utils.create_path()
            current_node.status = 'dead'
            bisect.insort(dead_list, current_node)
            current_node = current_node.parent
            continue

        # if the current node haven't been explored yet
        elif len(current_node.children) == 0:

            # in this node or it's children there is no better solution than what was found
            if current_node.f >= upper_bound or current_node.f > depth_limit:
                current_node.status = 'dead'
                bisect.insort(dead_list, current_node)
                current_node = current_node.parent

            # create children node and choose the cheapest for the next iteration
            else:
                curr_iteration = curr_iteration + 1
                potential_children = utils.get_children(current_node)
                for child in potential_children:
                    if utils.bisect_index(child, dead_list) > -1:
                        child.status = 'dead'
                    bisect.insort(current_node.children, child)

                if current_node.children[0].status == 'dead':
                    current_node.status = 'dead'
                    bisect.insort(dead_list, current_node)
                    current_node = current_node.parent
                else:
                    current_node = current_node.children[0]
            continue
        current_node.children.sort()
        # if there are live children choose the best alive
        if current_node.children[0].status == 'live':
            current_node = current_node.children[0]
        # if all children are dead no reason to explore farther, back to parent
        else:
            current_node.status = 'dead'
            bisect.insort(dead_list, current_node)
            current_node = current_node.parent

    if curr_iteration > iterations_limit and not is_solved:
        return 'failed iteration limit. iterations:{} , max_depth: {}'.format(curr_iteration, max_depth)
    elif curr_iteration <= iterations_limit and not is_solved:
        return 'failed to find solution. iterations:{} , max_depth: {}'.format(curr_iteration, max_depth)
    else:
        # print_path(curr_iteration, max_depth, 'b&b', path, solution_iterations=solution_iteration,
        #           solution_depth=best_solution)
        return {'best_solution': upper_bound, 'solution_iteration': solution_iteration,
                'final_iteration': curr_iteration, 'max_depth': max_depth}


def a_star(iterations_limit, depth_limit, i):
    current_node = utils.get_initial_state()
    open_list, close_list = [current_node], []

    # if the initial state is also the final state
    if current_node.h == 0:
        utils.assign_final_state_node(current_node)
        open_list = []

    curr_iteration, max_depth = 0, 0

    while len(open_list) > 0:
        curr_iteration = curr_iteration + 1
        current_node = open_list.pop(0)  # pop the node with the lowest cost from the open list
        max_depth = max([max_depth, current_node.g + 1])  # increase depth if necessary
        print('a* {} curr_iteration: {},max_depth: {}, curr_depth: {}'.format(i, curr_iteration, max_depth,
                                                                              current_node.g))

        potential_children = utils.get_children(current_node)
        for potential_child in potential_children:

            # found goal
            if potential_child.h == 0:
                utils.assign_final_state_node(potential_child)
                open_list = []
                max_depth = max([max_depth, potential_child.g])
                break
            # there a node with the same matrix the closed list
            elif utils.bisect_index(current_node, close_list) > -1:
                continue
            # there a node with the same matrix in the open list
            elif utils.bisect_index(current_node, open_list) > -1:
                index_in_open_list = utils.bisect_index(current_node, open_list)
                if potential_child.f >= open_list[index_in_open_list].f:
                    continue
                else:
                    old_node = open_list.pop(index_in_open_list)
                    bisect.insort(open_list, potential_child)
                    if utils.bisect_index(old_node, close_list) == -1:
                        bisect.insort(close_list, old_node)
            else:
                bisect.insort(open_list, potential_child)
        bisect.insort(close_list, current_node)

    # if curr_iteration > iterations_limit:
    #    return 'failed iteration limit. iterations:{} , max_depth: {}'.format(curr_iteration, max_depth)
    # else:
    #    # print_path(curr_iteration, max_depth, 'a*', create_path())
    #    return {'curr_iteration': curr_iteration, 'max_depth': max_depth}
    return {'final_iteration': curr_iteration, 'max_depth': max_depth}


def test_runtime():
    iterations_limit = 150000
    depth_limit = 31
    mats = [[2, 3, 6, 7, 5, 4, 0, 8, 1], [7, 6, 3, 5, 0, 8, 2, 1, 4]]

    template = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    utils.assign_final_state([1, 2, 3, 4, 5, 6, 7, 8, 0])

    while len(mats) < 2:
        random.shuffle(template)
        if utils.is_solvable(template) and template not in mats:
            mats.append(template.copy())

    df = pd.DataFrame()
    df['Matrix'] = mats
    df.to_csv(r'mats.csv', index=False, header=True)

    results = []
    for mat, i in zip(mats, np.arange(len(mats))):
        result = {'matrix': mat}
        for curr_heuristic in ['misplaced_tile', 'manhattan_distance', 'manhattan_distance_plus_reversal_penalty']:
            utils.assign_heuristic(curr_heuristic)

            initial_state = utils.init_initial_state(mat)
            print('a_star {} {}'.format(i, curr_heuristic))
            a_star_result = a_star(iterations_limit, depth_limit, '{} {}'.format(i, curr_heuristic))

            initial_state = utils.init_initial_state(mat)
            print('bnb {} {}'.format(i, curr_heuristic))
            bnb_result = branch_and_bound(iterations_limit, depth_limit, '{} {}'.format(i, curr_heuristic))

            result['a_star_{}'.format(curr_heuristic)] = a_star_result
            result['bnb_{}'.format(curr_heuristic)] = bnb_result

        results.append(
            (mat, result['a_star_misplaced_tile'], result['bnb_misplaced_tile'], result['a_star_manhattan_distance'],
             result['bnb_manhattan_distance'], result['a_star_manhattan_distance_plus_reversal_penalty'],
             result['bnb_manhattan_distance_plus_reversal_penalty']))
        print(i)

    with open('results.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(['matrix', 'a_star_misplaced_tile', 'bnb_misplaced_tile', 'a_star_manhattan_distance',
                        'bnb_manhattan_distance', 'a_star_manhattan_distance_plus_reversal_penalty',
                        'bnb_manhattan_distance_plus_reversal_penalty'])
        write.writerows(results)


def assign_heuristic_ui():
    heuristic = ''

    print('Choose heuristic:')
    print('For misplaced tile press 1')
    print('For Manhattan distance press 2')
    print('For Manhattan distance plus reversal penalty press 3')
    input1 = input("Enter heuristic")
    print()
    if input1 == '1':
        heuristic = 'misplaced_tile'
    elif input1 == '2':
        heuristic = 'manhattan_distance'
    elif input1 == '3':
        heuristic = 'manhattan_distance_plus_reversal_penalty'
    else:
        heuristic = 'manhattan_distance_plus_reversal_penalty'
        print('Wrong input was typed')

    utils.assign_heuristic(heuristic)
    print("Heuristic is: " + heuristic)
    print()


def main():
    iterations_limit = 150000
    depth_limit = 31
    final_list = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    initial_list = [4, 8, 2, 3, 0, 5, 1, 6, 7]
    if utils.is_legit_list(initial_list):
        if utils.is_solvable(initial_list):

            initial_state = utils.init_initial_state(initial_list)
            utils.assign_final_state(final_list)

            assign_heuristic_ui()

            input1 = input("For Branch&Bound press 1. For A* press 2")
            print()
            if input1 == '1':
                print(branch_and_bound(initial_state, iterations_limit, depth_limit, ""))
            else:
                print(a_star(initial_state, iterations_limit, depth_limit, ''))
        else:
            print('The matrix is not solvable ')
    else:
        print('The matrix is not legit')


if __name__ == '__main__':
    # main()
    test_runtime()
