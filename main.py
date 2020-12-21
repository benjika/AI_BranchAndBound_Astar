import numpy as np
import csv
import pandas as pd
import random
import ai_utils as utils
import bisect

phase = 'Assignment'  # 'Test' 'Assignment'


def branch_and_bound(iterations_limit, depth_limit, i):
    current_node = utils.get_initial_state()
    upper_bound, solution_iteration = np.inf, 0
    dead_list = []
    path = []

    curr_iteration, max_depth, iterations_until_solution = 0, 0, 0

    while current_node is not None:
        max_depth = max([max_depth, current_node.g])  # increase depth if necessary
        if phase == 'Test':
            print('bnb {} curr_iteration: {},max_depth: {}, curr_depth: {}'.format(i, curr_iteration, max_depth,
                                                                                   current_node.g))

        # if we found a new goal node
        if current_node.h == 0:
            # if we found a better path to goal node
            if upper_bound > current_node.f:
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

            # create children node and choose the most promising for the next iteration
            else:
                curr_iteration = curr_iteration + 1
                potential_children = utils.get_children(current_node)
                for child in potential_children:
                    # if the new born child already been found and died
                    if utils.node_in_list(child, dead_list):
                        child.status = 'dead'
                    # in sort the new born to children list
                    # sorted by : 1) live before dead 2) lower f
                    bisect.insort(current_node.children, child)

                # if your most promising child is dead, kill your self and go back
                # else choose him
                if current_node.children[0].status == 'dead':
                    current_node.status = 'dead'
                    bisect.insort(dead_list, current_node)
                    current_node = current_node.parent
                else:
                    current_node = current_node.children[0]
            continue

        # this part happens on the way back:
        # sort your children
        # sorted by : 1) live before dead 2) lower f
        current_node.children.sort()

        # if the most promising child is a live choose him
        if current_node.children[0].status == 'live':
            current_node = current_node.children[0]

        # if the most promising child is dead no reason to go farther
        # kill yourself and go back
        else:
            current_node.status = 'dead'
            if not utils.node_in_list(current_node, dead_list):
                bisect.insort(dead_list, current_node)
            current_node = current_node.parent

    if len(path) == 0:
        return "failed to find solution. iterations:{} , max_depth: {}".format(curr_iteration, max_depth)
    else:
        if phase == 'Assignment':
            utils.print_path(curr_iteration, max_depth, 'b&b', path, solution_iterations=solution_iteration,
                             solution_depth=upper_bound)
        return {"solution_depth": upper_bound, "solution_iteration": solution_iteration,
                "final_iteration": curr_iteration, "max_depth": max_depth}


def a_star(iterations_limit, depth_limit, i):
    current_node = utils.get_initial_state()
    open_list, close_list = [current_node], []
    path = []

    curr_iteration, max_depth = 0, 0

    while len(open_list) > 0:
        current_node = open_list.pop(0)  # pop the node with the lowest cost from the open list
        if phase == 'Test':
            print('a* {} curr_iteration: {},max_depth: {}, curr_depth: {}'.format(i, curr_iteration, max_depth,
                                                                                  current_node.g))
        # found goal
        if current_node.h == 0:
            utils.assign_final_state_node(current_node)
            open_list = []
            path = utils.create_path()

        else:
            potential_children = utils.get_children(current_node)
            curr_iteration += 1
            for potential_child in potential_children:

                # there a node with the same matrix the closed list or in the open list
                if utils.node_in_list(current_node, close_list) or utils.node_in_list(current_node, open_list):
                    continue

                # if the potential solution is worse than the longest possible path (in other words - no solution)
                elif potential_child.f > depth_limit:
                    bisect.insort(close_list, potential_child)

                # there is a chance this child is in the path solution
                else:
                    bisect.insort(open_list, potential_child)

        max_depth = max([max_depth, current_node.g])  # increase depth if necessary
        bisect.insort(close_list, current_node)

    if len(path) == 0:
        return "failed to find solution. iterations:{} , max_depth: {}".format(curr_iteration, max_depth)
    else:
        if phase == 'Assignment':
            utils.print_path(curr_iteration, max_depth, 'a*', path)
        return {"final_iteration": curr_iteration, "max_depth": max_depth, "solution_depth": len(path) - 1}


def test_runtime():
    iterations_limit = 150000
    depth_limit = 31
    mats = []

    template = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    utils.assign_final_state([1, 2, 3, 4, 5, 6, 7, 8, 0])

    while len(mats) < 2:
        random.shuffle(template)
        if utils.is_solvable(template) and template not in mats:
            mats.append(template.copy())

    df = pd.DataFrame()
    df['Matrix'] = mats
    df.to_csv(r'mats.csv', index=False, header=True)

    final_list = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    results = []
    for mat, i in zip(mats, np.arange(len(mats))):
        result = {'matrix': mat}
        for curr_heuristic in ['misplaced_tile', 'manhattan_distance', 'manhattan_distance_plus_reversal_penalty']:
            utils.assign_heuristic(curr_heuristic)

            utils.assign_final_state(final_list)
            utils.assign_initial_state(mat)
            print('a_star {} {}'.format(i, curr_heuristic))
            a_star_result = a_star(iterations_limit, depth_limit, '{} {}'.format(i, curr_heuristic))

            utils.assign_final_state(final_list)
            utils.assign_initial_state(mat)
            print('bnb {} {}'.format(i, curr_heuristic))
            bnb_result = branch_and_bound(iterations_limit, depth_limit, '{} {}'.format(i, curr_heuristic))

            result['a_star_{}'.format(curr_heuristic)] = a_star_result
            result['bnb_{}'.format(curr_heuristic)] = bnb_result

        avg_all, avg_a_star, avg_bnb, avg_misplaced_tile, avg_manhattan_distance, \
        avg_manhattan_distance_plus_reversal_penalty = utils.get_avgs(result)

        results.append(
            (mat, result['a_star_misplaced_tile'], result['bnb_misplaced_tile'], result['a_star_manhattan_distance'],
             result['bnb_manhattan_distance'], result['a_star_manhattan_distance_plus_reversal_penalty'],
             result['bnb_manhattan_distance_plus_reversal_penalty'], avg_all, avg_a_star, avg_bnb, avg_misplaced_tile,
             avg_manhattan_distance, avg_manhattan_distance_plus_reversal_penalty))
        print(i)

    with open('results.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(['matrix', 'a_star_misplaced_tile', 'bnb_misplaced_tile', 'a_star_manhattan_distance',
                        'bnb_manhattan_distance', 'a_star_manhattan_distance_plus_reversal_penalty',
                        'bnb_manhattan_distance_plus_reversal_penalty', 'avg_all', 'avg_a_star', 'avg_bnb',
                        'avg_misplaced_tile', 'avg_manhattan_distance', 'avg_manhattan_distance_plus_reversal_penalty'])
        write.writerows(results)


def assign_heuristic_ui():
    heuristic = ''

    print('Choose heuristic:')
    print('For Misplaced Tile press 1')
    print('For Manhattan Distance press 2')
    print('For Manhattan Distance + Reversal Penalty press 3')
    input1 = input("Enter Heuristic")
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
    utils.assign_final_state(final_list)

    ###################################################
    ###################################################
    ## to change initial state please change this line
    initial_list = [1, 3, 6, 4, 8, 2, 0, 5, 7]
    ####################################################
    ####################################################
    if utils.is_legit_list(initial_list):
        if utils.is_solvable(initial_list):
            assign_heuristic_ui()

            utils.assign_initial_state(initial_list)
            print("For Branch&Bound press 1")
            print("For A* press 2")
            input1 = input("")
            print()
            if input1 == '1':
                print(branch_and_bound(iterations_limit, depth_limit, ""))
            else:
                print(a_star(iterations_limit, depth_limit, ''))
        else:
            print('The matrix is not solvable ')
    else:
        print('The matrix is not legit')


if __name__ == '__main__':
    if phase == 'Assignment':
        main()
    elif phase == 'Test':
        test_runtime()
