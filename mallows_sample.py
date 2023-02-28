import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from random import choices
from random import randint
import math
import random
from random import randrange
import code as relph
import multiprocessing
from functools import partial
import time

lookup={}
summed_iv=0
#Given the number m of candidates and a phi\in [0,1] function computes the expected number of swaps in a vote sampled from Mallows model
def calculateExpectedNumberSwaps(m,phi):
    res= phi*m/(1-phi)
    for j in range(1,m+1):
        res = res + (j*(phi**j))/((phi**j)-1)
    return res

#Given the number m of candidates and a absolute number of expected swaps exp_abs, this function returns a value of phi such that in a vote sampled from Mallows model with this parameter the expected number of swaps is exp_abs
def binary_search_phi(m,exp_abs):
    if exp_abs==m*(m-1)/2:
        return 1
    if exp_abs==0:
        return 0
    low=0
    high=1
    while low <= high:
        mid = (high + low) / 2
        cur=calculateExpectedNumberSwaps(m, mid)
        #print('mid',mid)
        if abs(cur-exp_abs)<1e-5:
            return mid
        if mid>0.999999:
            return 1
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1

def computeInsertionProbas(i,phi):
    probas = (i+1)*[0]
    for j in range(i+1):
        probas[j]=pow(phi,(i+1)-(j+1))
    return probas

def weighted_choice(choices):
    total=0
    for w in choices:
        total=total + w
    r = random.uniform(0, total)
    upto = 0.0
    for i,w in enumerate(choices):
        if upto + w >= r:
            return i
        upto =upto + w
    assert False, "Shouldn't get here"

def mallowsVote(m,insertion_probas):
    vote=[0]
    for i in range(1,m):
        we=insertion_probas[:i+1]
        #rounding issue
        if we[-1]==0:
            we[-1]=1
        index = weighted_choice(we)
        vote.insert(index,i)
    return vote

#Number n of voters, number m of candidadtes, number num_elections of elections to be returned
#relphi=True means that we use normalized phi, relphi=False means we use classical Mallows
#lphi\in [0,1] that gets normalized depending on the value of relphi
def mallowsElection(n,m,num_elections,lphi, reverse=0):
    elections=[]
    phi = binary_search_phi(m,lphi*(m*(m-1))/4)
    #print(phi)
    insertion_probas = computeInsertionProbas(m, phi)
    #print(insertion_probas)
    for i in range(num_elections):
        election = []
        for j in range(n):
            vote = mallowsVote(m,insertion_probas)
            if reverse>0:
                probability= random.random()
                if probability>=reverse:
                    vote.reverse()
            election.append(vote)
        elections.append(election)
    return elections


def getInvCount(arr, n):
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (arr[i] > arr[j]):
                inv_count += 1

    return inv_count

def calcMallosElectionIncomplete(election, n, n_cand_list, n_mods):
    #print(n_mods)
    global lookup
    #global summed_iv
    elec = []
    iv=0
    for i in range(n):
        m=n_cand_list[i]
        if m in lookup:
            if n_mods in lookup[m]:
                insertion = lookup[m][n_mods]
            else:
                phi = binary_search_phi(m, n_mods * (m * (m - 1)) / 4)
                insertion = computeInsertionProbas(m, phi)
                lookup[m][n_mods] = insertion
        else:
            lookup[m] = {}
            phi = binary_search_phi(m, n_mods * (m * (m - 1)) / 4)
            insertion = computeInsertionProbas(m, phi)
            lookup[m][n_mods] = insertion
            #print('CREATE')
            #print(lookup)
        #print(insertion)
        perm = mallowsVote(m,insertion)
        #iv+=getInvCount(perm, m)


        new_vote = [0] * m
        old_vote = election[i]
        for t in range(m):
            new_vote[t] = old_vote[perm[t]]
        elec.append(new_vote)
    #print(iv)
    #summed_iv+=iv
    #print(summed_iv)
    return elec

def calcMallosElection(election, n,m, n_mods):
    #print('Elec')
    global lookup
    if m in lookup:
        if n_mods in lookup[m]:
            insertion=lookup[m][n_mods]
        else:
            phi = binary_search_phi(m, n_mods * (m * (m - 1)) / 4)
            insertion= computeInsertionProbas(m, phi)
            lookup[m][n_mods] = insertion
    else:
        lookup[m]={}
        phi = binary_search_phi(m, n_mods * (m * (m - 1)) / 4)
        insertion = computeInsertionProbas(m, phi)
        lookup[m][n_mods] = insertion
    elec = []
    for i in range(n):
        perm=mallowsVote(m,insertion)
        new_vote=[0]*m
        old_vote=election[i]
        for t in range(m):
            new_vote[t]=old_vote[perm[t]]
        elec.append(new_vote)
    return elec
