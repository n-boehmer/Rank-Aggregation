#Given the number m of candidates and a phi\in [0,1] function computes the expected number of swaps in a vote sampled from Mallows model
def calculateExpectedNumberSwaps(m,phi):
    res= phi*m/(1-phi)
    for j in range(1,m+1):
        res = res + (j*(phi**j))/((phi**j)-1)
    return res

#Given the number m of candidates and a absolute number of expected swaps exp_abs, this function returns a value of phi such that in a vote sampled from Mallows model with this parameter the expected number of swaps is exp_abs
def binary_search_phi(m,exp_abs):
    low=0
    high=1
    while low <= high:
        #print(low)
        #print(high)
        mid = (high + low) / 2
        cur=calculateExpectedNumberSwaps(m, mid)
        if abs(cur-exp_abs)<1e-5:
            return mid
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1


m=10000
relphi=0.1
exp_abs=relphi*(m*(m-1))/2
phi=binary_search_phi(m,exp_abs)
print(phi)
print(exp_abs)
print(calculateExpectedNumberSwaps(m,phi))
