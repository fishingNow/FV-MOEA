# %%
import numpy as np
from matplotlib import pyplot as plt

# import theoretical_experiments.start
def nondominated_worse(anchor, siblings, ref_point, debug=False):
    # Consider solutions in relation to the reference_point.
    # Else, border points won't have contibutions.
    ref_points = np.array([anchor, ref_point])
    ref_points[:, 1] = ref_points[:, 1][::-1]

    siblings = np.concatenate([siblings, ref_points])

    worse_solutions = np.maximum(anchor, siblings)
    worse_hv_mask = pareto_front(
        worse_solutions, np.ones(len(worse_solutions), dtype=bool)
    )
    worse_solutions = worse_solutions[worse_hv_mask]
    if debug:
        plt.scatter(*anchor)
        plt.scatter(*siblings.T)
        plt.scatter(*worse_solutions.T)
        plt.show()
        plt.close()
    return worse_solutions

# %%
def abline(slope, point):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = point[1] + slope * (x_vals - point[0])
    plt.plot(x_vals, y_vals, "--")

def fv_moea(fitnesses, front_mask, k_selected, metric_weights=np.array([1, 1])):
    # Create the reference point
    ref = fitnesses.max(axis=0) * (1 + metric_weights)

    # Calculate contributions for each point.
    player_idxs = np.arange(len(fitnesses))[front_mask]
    contribs = np.empty(len(player_idxs))
    for contrib_idx, s_idx in enumerate(player_idxs):
        anchor_score = fitnesses[s_idx]
        siblings_mask = front_mask.copy()
        siblings_mask[s_idx] = False
        siblings = fitnesses[siblings_mask]
        # get the element-wise worst fitness.
        worse_solutions = nondominated_worse(anchor_score, siblings, ref)
        contrib = np.prod(worse_solutions.max(axis=0) - anchor_score)
        contribs[contrib_idx] = contrib

    winners_idxs = np.argpartition(contribs, -k_selected)[-k_selected:]
    winners_idxs = player_idxs[winners_idxs]
    winners_mask = np.zeros(len(fitnesses), dtype=bool)
    winners_mask[winners_idxs] = True
    return winners_mask

# %%
def pareto_front(scores, remaining):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A boolean array of indices of pareto-efficient points.
    """

    player_idxs = np.arange(len(scores))[remaining]
    costs = scores[player_idxs].copy()

    nondominated = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the nondominated array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        nondominated = nondominated[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    nondominated_mask = np.zeros(n_points, dtype=bool)
    nondominated_mask[nondominated] = True

    winnerss = np.zeros(len(remaining), dtype=bool)
    winnerss[player_idxs[nondominated_mask]] = True

    return remaining & winnerss
# %%
fitnesses = np.array(
    [
        # non-dominated
        [0.1, 0.5],  # c
        [0.2, 0.4],  # b
        [0.35, 0.25],  # a
        [0.7, 0.1],  # d
        [0.8, 0.05],  # e
        # # dominated
        [0.35, 0.5],  # w2
        [0.35, 0.25],  # w1
        [0.7, 0.25],  # w3
        [0.5, 0.4],
        [0.6, 0.3],
        [0.8, 0.25],  # w4
    ]
)

# %%
# plotting solutions
plt.scatter(*fitnesses.T)
for i in range(len(fitnesses)):
    plt.text(*fitnesses[i], str(i))
# %%
# get R, anchor.
percentage_increase = 0.5
ref = fitnesses.max(axis=0) * (1 + percentage_increase)

plt.scatter(*fitnesses.T)
plt.scatter(*ref)



# basic test works
remaining = np.ones(len(fitnesses), dtype=bool)
winners = pareto_front(fitnesses, remaining)
plt.scatter(*fitnesses[winners].T)
for i, exists in enumerate(winners):
    if exists:
        plt.text(*fitnesses[i], str(i))
# %%
test_unordered = np.array(
    [
        # non-dominated
        [0.8, 0.05],  # e
        # # dominated
        [0.35, 0.5],  # w2
        [0.2, 0.4],  # b
        [0.35, 0.4],  # w1
        [0.1, 0.5],  # c
        [0.35, 0.25],  # a
        [0.7, 0.1],  # d
        [0.7, 0.25],  # w3
        [0.8, 0.25],  # w4
        [0.25, 0.25],  # w4
        [0.27, 0.24],  # w4
    ]
)
plt.scatter(*test_unordered.T)
for i in range(len(test_unordered)):
    plt.text(*test_unordered[i], str(i))
# %%
remaining = np.ones(len(test_unordered), dtype=bool)
winners = pareto_front(test_unordered, remaining)
plt.scatter(*test_unordered[winners].T)
for i, exists in enumerate(winners):
    if exists:
        plt.text(*test_unordered[i], str(i))




# %%
remaining = np.ones(len(fitnesses), dtype=bool)
remaining
# %%
plt.close()
# remaining, winners
plt.scatter(*fitnesses[remaining].T)
for i, exists in enumerate(remaining):
    if exists:
        plt.text(*fitnesses[i], str(i))
        abline(-1, fitnesses[i])
winners = pareto_front(fitnesses, remaining)
plt.scatter(*fitnesses[winners].T)
for i, exists in enumerate(remaining & winners):
    if exists:
        plt.text(*fitnesses[i], str(i))
remaining &= ~winners
# %%



# %%
fitnesses = test_unordered
debug = True
population_size = 8
next_gen = np.zeros(len(fitnesses), dtype=bool)
remaining = np.ones(len(fitnesses), dtype=bool)
while next_gen.sum() < population_size and remaining.sum() > 0:
    winners = pareto_front(fitnesses, remaining)
    remaining &= ~winners

    if next_gen.sum() + winners.sum() <= population_size:
        print(f"tenemos: {next_gen.sum() + winners.sum()}/{population_size}")
        next_gen |= winners
    elif next_gen.sum() + winners.sum() > population_size:
        # fv-moea para los que faltan
        fv_winners = fv_moea(fitnesses, winners, population_size - next_gen.sum())
        next_gen |= fv_winners
        if debug:
            print(next_gen)
            plt.scatter(*fitnesses.T)
            plt.scatter(*fitnesses[next_gen].T)
            plt.scatter(*fitnesses[fv_winners].T)
            plt.show()
            plt.close()
        break

    if debug:
        print(next_gen)
        plt.scatter(*fitnesses.T)
        plt.scatter(*fitnesses[next_gen].T)
        plt.scatter(*fitnesses[winners].T)
        plt.show()
        plt.close()

    if next_gen.sum() == population_size:
        break

# %%
plt.scatter(*fitnesses.T)
# %%