import numpy as np
from random import randrange
import random
import math
import operator
from pathlib import Path
import csv
from collections import Counter
from glob import glob
import os
import operator as op
from functools import reduce
import matplotlib as mpl
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
from random import sample
import cProfile
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tikzplotlib
import matplotlib
import copy
import sys
import statistics
from scipy.stats import kendalltau, spearmanr
import mallows_sample
from functools import cmp_to_key
from os import listdir
from os.path import isfile, join
import glob

k_approv = 1
try:
    from gurobipy import *
except:
    pass
import itertools

random_tie_breaking = True


def better(vote, a, b):
	return vote.index(a) < vote.index(b)


def kemeny(profile):
	C = profile[0]
	w = {}
	for a, b in itertools.permutations(C, 2):
		w[a, b] = 0
		for vote in profile:
			if better(vote, a, b):
				w[a, b] += 1
			else:
				w[a, b] -= 1
	m = Model()
	m.params.OutputFlag = 0
	x = {}
	for a, b in itertools.combinations(C, 2):
		x[a, b] = m.addVar(vtype=GRB.BINARY)
		x[b, a] = m.addVar(vtype=GRB.BINARY)
		m.addConstr(x[a, b] + x[b, a] == 1)
	m.addConstrs(x[a, b] + x[b, c] + x[c, a] <= 2 for a, b, c in itertools.permutations(C, 3))
	m.setObjective(quicksum(x[a, b] * w[a, b] for a, b in itertools.permutations(C, 2)), GRB.MAXIMIZE)
	m.optimize()
	return tuple(sorted(C, key=lambda a: -sum(x[a, b].x for b in C if b != a)))


def set_k_approv(k):
	global k_approv
	k_approv = k


def kapproval(vot, m):
	global k_approv
	points = [0] * m
	for v in vot:
		for t in range(k_approv):
			points[v[t]] = points[v[t]] + 1
	return points


def generate_ordinal_euclidean_votes(n, m, num_elections, dim):
	elections = []
	for i in range(num_elections):
		voters = np.random.rand(n, dim)
		candidates = np.random.rand(m, dim)
		election = []
		for v in range(n):
			distance_to_candidates = []
			for c in range(m):
				distance_to_candidates.append(np.linalg.norm(voters[v] - candidates[c]))

			election.append([x for _, x in sorted(zip(distance_to_candidates, list(range(m))))])
		elections.append(election)

	return elections


def plurality(vot, m):
	points = [0] * m
	for v in vot:
		if len(v) > 0:
			points[v[0]] = points[v[0]] + 1
	return points


def borda(vot, m):
	points = [0] * m
	for v in vot:
		for i in range(len(v)):
			points[v[i]] = points[v[i]] + (len(v) - i - 1)
	return points


def veto(vot, m):
	points = [0] * m
	for v in vot:
		for i in range(len(v) - 1):
			points[v[i]] = points[v[i]] + 1
	return points


def half(vot, m):
	points = [0] * m
	for v in vot:
		for i in range(int(len(v) / 2)):
			points[v[i]] = points[v[i]] + 1
	return points


def eliminate_bottom(vot, m, rule, tiebreaking):
	tie = 0
	votes = []
	for v in vot:
		vvv = []
		for c in v:
			vvv.append(c)
		votes.append(vvv)

	not_deleted = list(range(m))
	order = [0] * m
	points = rule(vot, m)
	for i in range(m - 1):
		min_relevant = min([points[i] for i in not_deleted])
		cand_to_be_del = [i for i in not_deleted if points[i] == min_relevant]
		if len(cand_to_be_del) > 1:
			tie = tie + 1
		for t in tiebreaking:
			if t in cand_to_be_del:
				delete = t
				break
		order[m - i - 1] = delete
		not_deleted.remove(delete)
		for i in range(len(votes)):
			if delete in votes[i]:
				votes[i].remove(delete)
		points = rule(votes, m)
	order[0] = not_deleted[0]
	return order, tie


def eliminate_top(vot, m, rule, tiebreaking):
	tie = 0
	tiebreaking = list(reversed(tiebreaking))
	votes = []
	for v in vot:
		vvv = []
		for c in v:
			vvv.append(c)
		votes.append(vvv)
	not_deleted = list(range(m))
	order = [0] * m
	points = rule(vot, m)
	for i in range(m - 1):
		max_relevant = max([points[i] for i in not_deleted])
		cand_to_be_del = [i for i in not_deleted if points[i] == max_relevant]
		if len(cand_to_be_del) > 1:
			tie = tie + 1
		for t in tiebreaking:
			if t in cand_to_be_del:
				delete = t
				break
		order[i] = delete
		not_deleted.remove(delete)
		for i in range(len(votes)):
			if delete in votes[i]:
				votes[i].remove(delete)
		points = rule(votes, m)
	order[m - 1] = not_deleted[0]
	return order, tie


tie_breaking_order = None
tie = None


def compare(item1, item2):
	if item1[0] > item2[0]:
		return 1
	elif item1[0] < item2[0]:
		return -1
	elif tie_breaking_order.index(item1[1]) < tie_breaking_order.index(item2[1]):
		global tie
		tie = tie + 1
		return 1
	else:
		return -1


def score_ordering(vot, m, rule, tiebreaking):
	global tie
	tie = 0
	global tie_breaking_order
	tie_breaking_order = tiebreaking
	points = rule(vot, m)
	inversed_points = [-x for x in points]
	to_be_sorted = list(zip(inversed_points, list(range(m))))
	return [x for _, x in sorted(to_be_sorted, key=cmp_to_key(compare))], tie


def swap_distance(order1, order2, m):
	distance = 0
	for i in range(m):
		for j in range(0, i):
			if not (order1.index(i) < order1.index(j)) == (order2.index(i) < order2.index(j)):
				distance += 1
	return distance / (m * (m - 1) / 2)


def spear_by_position(order1, order2, m):
	distance = [0] * m
	for i in range(m):
		dis_1 = abs(i - order2.index(order1[i]))
		dis_2 = abs(i - order1.index(order2[i]))
		distance[i] = (dis_1 + dis_2) / 2
	return distance


def run_experiment(elections, m, rule, rt, natural=False, visuals=False, name=""):
	list_disSTVEli = []
	list_disSTVPoints = []
	list_disEliPoints = []

	list_disSTVnatural = []
	list_disPointsnatural = []
	list_disElinatural = []

	list_spearSTVEli = []
	list_spearSTVPoints = []
	list_spearEliPoints = []

	list_spearSTVnatural = []
	list_spearPointsnatural = []
	list_spearElinatural = []

	list_tieSTV = []
	list_tieElimination = []
	list_tiePoints = []

	lists_spear_by_position = [[], [], [], [], [], []]
	result_names = ["Seq.-Lo. vs Seq.-Wi.", "Seq.-Lo. vs Score", "Seq.-Wi. vs Score", "Kemeny vs Seq.-Lo.",
					"Kemeny vs Score", "Kemeny vs Seq.-Wi."]

	for index, election in enumerate(elections):
		print(index)
		if rt:
			tie_breaking = list(np.random.permutation(m))
		else:
			tie_breaking = list(reversed(range(m)))
		orderingSTV, tieSTV = eliminate_bottom(election, m, rule, tie_breaking)
		orderingElimination, tieElimination = eliminate_top(election, m, rule, tie_breaking)
		orderingPoints, tiePoints = score_ordering(election, m, rule, tie_breaking)

		list_tieSTV.append(tieSTV)
		list_tieElimination.append(tieElimination)
		list_tiePoints.append(tiePoints)

		disSTVEli = swap_distance(orderingSTV, orderingElimination, m)
		disSTVPoints = swap_distance(orderingSTV, orderingPoints, m)
		disEliPoints = swap_distance(orderingElimination, orderingPoints, m)

		spearSTVEli, _ = spearmanr(orderingSTV, orderingElimination)
		spearSTVPoints, _ = spearmanr(orderingSTV, orderingPoints)
		spearEliPoints, _ = spearmanr(orderingElimination, orderingPoints)

		list_spearSTVEli.append(spearSTVEli)
		list_spearSTVPoints.append(spearSTVPoints)
		list_spearEliPoints.append(spearEliPoints)

		list_disSTVEli.append(disSTVEli)
		list_disSTVPoints.append(disSTVPoints)
		list_disEliPoints.append(disEliPoints)

		if natural:
			central = kemeny(election)
		else:
			central = list(range(m))
		list_disSTVnatural.append(swap_distance(orderingSTV, central, m))
		list_disPointsnatural.append(swap_distance(orderingPoints, central, m))
		list_disElinatural.append(swap_distance(orderingElimination, central, m))

		list_spearSTVnatural.append(spearmanr(orderingSTV, central)[0])
		list_spearPointsnatural.append(spearmanr(orderingPoints, central)[0])
		list_spearElinatural.append(spearmanr(orderingElimination, central)[0])

		lists_spear_by_position[0].append(spear_by_position(orderingSTV, orderingElimination, m))
		lists_spear_by_position[1].append(spear_by_position(orderingSTV, orderingPoints, m))
		lists_spear_by_position[2].append(spear_by_position(orderingElimination, orderingPoints, m))

		lists_spear_by_position[3].append(spear_by_position(orderingSTV, central, m))
		lists_spear_by_position[4].append(spear_by_position(orderingPoints, central, m))
		lists_spear_by_position[5].append(spear_by_position(orderingElimination, central, m))

	if visuals:
		averages_lists_spear_by_position = [[0] * m, [0] * m, [0] * m, [0] * m, [0] * m, [0] * m]
		for i in range(6):
			for j in range(m):
				sum = 0
				for t in range(len(elections)):
					sum += lists_spear_by_position[i][t][j]
				averages_lists_spear_by_position[i][j] = sum / len(elections)
		colors = ['darkred', 'darkorange', 'forestgreen', 'dodgerblue', 'darkviolet', "brown"]
		for i in range(3):
			plt.plot(list(range(m)), averages_lists_spear_by_position[i], label=result_names[i], color=colors[i])

		for i in range(3, 6):
			plt.plot(list(range(m)), averages_lists_spear_by_position[i], label=result_names[i], color=colors[i],
					 linestyle="dashed")

		directory = "./plots/"
		if not os.path.exists(directory):
			os.makedirs(directory)
		plt.legend(loc="upper left")
		plt.ylabel('Average Spearman distance')
		plt.xlabel('rank')
		plt.savefig(directory + name + "_positionchange.png")
		tikzplotlib.save(directory + name + "_positionchange.tex", encoding='utf-8')
		plt.close()

	return statistics.mean(list_disSTVEli), statistics.mean(list_disSTVPoints), statistics.mean(list_disEliPoints), \
		   statistics.mean(list_spearSTVEli), statistics.mean(list_spearSTVPoints), \
		   statistics.mean(list_spearEliPoints), statistics.mean(list_disSTVnatural), statistics.mean(
		list_disPointsnatural), \
		   statistics.mean(list_disElinatural), statistics.mean(list_spearSTVnatural), statistics.mean(
		list_spearPointsnatural), statistics.mean(list_spearElinatural), statistics.mean(list_tieSTV), statistics.mean(
		list_tieElimination), statistics.mean(list_tiePoints)


#By setting the mallows parameter to false, this also produces results for Euclidean elections
def mallows_experiment(nn, mm, sample, rule, rt, rule_name, mallows, natural=False):
	if mallows:
		string = "mallows"
	else:
		string = "euclidean"
	colors = ['darkred', 'darkorange', 'forestgreen', 'dodgerblue', 'darkviolet', "brown"]
	for n in nn:
		for m in mm:
			if mallows:
				norm_phis = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
			else:
				norm_phis = [1, 2, 3, 4, 5, 10, 15, 20]
			list_results = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
			result_names = ["Seq.-Lo. vs Seq.-Wi.", "Seq.-Lo. vs Score", "Seq.-Wi. vs Score", "Seq.-Lo. vs Seq.-Wi.",
							"Seq.-Lo. vs Score", "Seq.-Wi. vs Score", "Kemeny vs Seq.-Lo.", "Kemeny vs Score",
							"Kemeny vs Seq.-Wi.", "Kemeny vs Seq.-Lo.", "Kemeny vs Score", "Kemeny vs Seq.-Wi.",
							"Seq.-Lo.", "Seq.-Wi.", "Score"]
			for phi in norm_phis:
				if mallows:
					elections = mallows_sample.mallowsElection(n, m, sample, phi)
				else:
					elections = generate_ordinal_euclidean_votes(n, m, sample, phi)
				results = run_experiment(elections, m, rule, rt, natural)
				for i in range(15):
					list_results[i].append(results[i])
			for i in range(3):
				plt.plot(norm_phis, list_results[i], label=result_names[i], color=colors[i])
			if natural:
				for i in range(6, 9):
					plt.plot(norm_phis, list_results[i], linestyle='dashed',
							 label=result_names[i], color=colors[i - 3])

	directory = "./plots/"
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.legend(loc="upper left")
	plt.ylabel('normalized swap distance')
	if mallows:
		plt.xlabel('dispersion parameter')
	else:
		plt.xlabel('dimension')
	plt.savefig(directory + rule_name + "_" + str(n) + "_" + str(m) + "_" + string + ".png")
	tikzplotlib.save(directory + rule_name + "_" + str(n) + "_" + str(m) + "_" + string + ".tex", encoding='utf-8')
	plt.close()
	for i in range(12, 15):
		plt.plot(norm_phis, list_results[i],
				 label=result_names[i], color=colors[i - 12])
	plt.ylabel('average number of rounds with tie')
	plt.xlabel('dispersion parameter')
	plt.legend(loc="upper left")
	plt.savefig(directory + "tie_freq_"+rule_name + "_" + str(n) + "_" + str(m) + "_" + string + ".png")
	tikzplotlib.save(directory +"tie_freq_"+rule_name + "_" + str(n) + "_" + str(m) + "_" + string + ".tex",
					 encoding='utf-8')
	plt.close()


#By setting the mallows parameter to false, this also produces results for Euclidean elections
def mallows_experiment_changingsize(nn, m, sample, rule, rule_name, mallows, param, natural=False):
	if mallows:
		string = "mallows"
	else:
		string = "euclidean"
	colors = ['darkred', 'darkorange', 'forestgreen', 'dodgerblue', 'darkviolet', "brown"]
	list_results = [[], [], [], [], [], [], [], [], [], [], [], []]
	result_names = ["Seq.-Lo. vs Seq.-Wi.", "Seq.-Lo. vs Score", "Seq.-Wi. vs Score", "Seq.-Lo. vs Seq.-Wi.",
					"Seq.-Lo. vs Score", "Seq.-Wi. vs Score", "Kemeny vs Seq.-Lo.", "Kemeny vs Score",
					"Kemeny vs Seq.-Wi.", "Kemeny vs Seq.-Lo.", "Kemeny vs Score", "Kemeny vs Seq.-Wi."]
	for n in nn:
		if mallows:
			elections = mallows_sample.mallowsElection(n, m, sample, param)
		else:
			elections = generate_ordinal_euclidean_votes(n, m, sample, param)
		results = run_experiment(elections, m, rule, True, natural)
		for i in range(12):
			list_results[i].append(results[i])
	for i in range(3):
		plt.plot(nn, list_results[i], label=result_names[i], color=colors[i])
	if natural:
		for i in range(6, 9):
			plt.plot(nn, list_results[i], linestyle='dashed',
					 label=result_names[i], color=colors[i - 3])

	directory = "./plots/"
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.legend(loc="upper left")
	plt.ylabel('normalized swap distance')
	if mallows:
		plt.xlabel('number of voters')
	else:
		plt.xlabel('number of voters')
	plt.savefig(directory + "varying_n_" + rule_name + "_" + string + ".png")
	tikzplotlib.save(directory + "varying_n_" + rule_name + "_" + string + ".tex",
					 encoding='utf-8')
	plt.close()

#By setting the mallows parameter to false, this also produces results for Euclidean elections
def mallows_experiment_changingsize_m(n, mm, sample, rule, rule_name, mallows, param, natural=False):
	if mallows:
		string = "mallows"
	else:
		string = "euclidean"
	colors = ['darkred', 'darkorange', 'forestgreen', 'dodgerblue', 'darkviolet', "brown"]
	list_results = [[], [], [], [], [], [], [], [], [], [], [], []]
	result_names = ["Seq.-Lo. vs Seq.-Wi.", "Seq.-Lo. vs Score", "Seq.-Wi. vs Score", "Seq.-Lo. vs Seq.-Wi.",
					"Seq.-Lo. vs Score", "Seq.-Wi. vs Score", "Kemeny vs Seq.-Lo.", "Kemeny vs Score",
					"Kemeny vs Seq.-Wi.", "Kemeny vs Seq.-Lo.", "Kemeny vs Score", "Kemeny vs Seq.-Wi."]
	for m in mm:
		if mallows:
			elections = mallows_sample.mallowsElection(n, m, sample, param)
		else:
			elections = generate_ordinal_euclidean_votes(n, m, sample, param)
		results = run_experiment(elections, m, rule, True, natural)
		for i in range(12):
			list_results[i].append(results[i])
	for i in range(3):
		plt.plot(mm, list_results[i], label=result_names[i], color=colors[i])
	if natural:
		for i in range(6, 9):
			plt.plot(mm, list_results[i], linestyle='dashed',
					 label=result_names[i], color=colors[i - 3])

	directory = "./plots/"
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.legend(loc="upper left")
	plt.ylabel('normalized swap distance')
	plt.xlabel('number of candidates')
	plt.savefig(directory + "varying_m_" + rule_name + "_" + string + ".png")
	tikzplotlib.save(directory + "varying_m_" + rule_name + "_" + string + ".tex",
					 encoding='utf-8')
	plt.close()



os.makedirs("./plots", exist_ok=True)

it = 10000
l_it = 100

random.seed(0)
# Figure 1 and 4 (plots/borda_100_10_mallows, plots/tie_freq_borda_100_10_mallows, plots/plurality_100_10_mallows, plots/tie_freq_plurality_100_10_mallows)
mallows_experiment([100], [10], it, plurality, True, "plurality", mallows=True, natural=True)
mallows_experiment([100], [10], it, borda, True, "borda", mallows=True, natural=True)


#Figure 2 (plots/borda_100_10_euclidean, plots/plurality_100_10_euclidean)
random.seed(0)
mallows_experiment([100], [10], it, plurality, True, "plurality", mallows=False, natural=True)
mallows_experiment([100], [10], it, borda, True, "borda", mallows=False, natural=True)

# Figure 3  (plots/Plurality_mal_positionchange, plots/Plurality_euc_positionchange, plots/Borda_mal_positionchange, plots/Borda_euc_positionchange)
run_experiment(mallows_sample.mallowsElection(100, 10, it, 0.8), 10, borda, True, natural=True, visuals=True,
			   name="Borda_mal")
run_experiment(mallows_sample.mallowsElection(100, 10, it, 0.8), 10, plurality, True, natural=True, visuals=True,
			   name="Plurality_mal")

run_experiment(generate_ordinal_euclidean_votes(100, 10, it, 10), 10, borda, True, natural=True, visuals=True,
			   name="Borda_euc")
run_experiment(generate_ordinal_euclidean_votes(100, 10, it, 10), 10, plurality, True, natural=True, visuals=True,
			   name="Plurality_euc")


# Figure 5 (plots/varying_n_plurality_mallows, plots/varying_n_plurality_euclidean, plots/varying_n_borda_mallows, plots/varying_n_borda_euclidean)
mallows_experiment_changingsize([25, 50, 100, 200, 300, 400, 500], 10, l_it, plurality, 'plurality', True, 0.8,
								natural=True)
mallows_experiment_changingsize([25, 50, 100, 200, 300, 400, 500], 10, l_it, plurality, 'plurality', False, 10,
								natural=True)

mallows_experiment_changingsize([25, 50, 100, 200, 300, 400, 500], 10, l_it, borda, 'borda', True, 0.8, natural=True)
mallows_experiment_changingsize([25, 50, 100, 200, 300, 400, 500], 10, l_it, borda, 'borda', False, 10, natural=True)


# Figure 6 (plots/varying_m_plurality_mallows, plots/varying_m_plurality_euclidean, plots/varying_m_borda_mallows, plots/varying_m_borda_euclidean)
mallows_experiment_changingsize_m(100, [5, 10, 25, 50, 75], l_it, plurality, 'plurality', True, 0.8,
								  natural=True)
mallows_experiment_changingsize_m(100, [5, 10, 25, 50, 75], l_it, plurality, 'plurality', False, 10,
								  natural=True)

mallows_experiment_changingsize_m(100, [5, 10, 25, 50, 75], l_it, borda, 'borda', True, 0.8,
								  natural=True)
mallows_experiment_changingsize_m(100, [5, 10, 25, 50, 75], l_it, borda, 'borda', False, 10,
								  natural=True)

# Figure 7 (plots/veto_100_10_mallows, plots/half_100_10_mallows)
mallows_experiment([100], [10], it, veto, True, "veto", mallows=True, natural=True)
mallows_experiment([100], [10], it, half, True, "half", mallows=True, natural=True)
