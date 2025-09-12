# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd

# boarding a plane — Karen vs. Drunk Passengers

def is_karen_upset(N: int, rng: random.Random = None) -> bool:
    """
    Scenario: passenger 1 is intoxicated (random seat); others take assigned seats
    unless occupied, then choose a random vacant seat.
    Returns True if the last passenger's (seat N) is taken when she boards.
    """
    r = rng or random
    seat_taken = [False] * N
    first = r.randrange(N)           
    seat_taken[first] = True
    if first == N - 1:
        return True  # Karen's seat instantly taken

    for i in range(1, N - 1):         # passengers 2 - N-1 (1-based) -> indices 1 - N-2
        if not seat_taken[i]:
            seat_taken[i] = True
        else:
            empties = [idx for idx, taken in enumerate(seat_taken) if not taken]
            s = r.choice(empties)
            seat_taken[s] = True
            if s == N - 1:
                return True
    return False


def is_karen_upset_odd_drunk(N: int, rng: random.Random = None) -> bool:
    """
    Modified scenario: every odd passenger (1st, 3rd, 5th, ...) is intoxicated and
    picks a random vacant seat; others try assigned seat else random vacant seat.
    Returns True if Karen's seat (N) is taken when the last passenger boards.
    """
    r = rng or random
    seat_taken = [False] * N

    # Passengers 1 - N-1 take seats
    for i in range(1, N): 
        if i % 2 == 1:
            empties = [idx for idx, taken in enumerate(seat_taken) if not taken]
            s = r.choice(empties)
            seat_taken[s] = True
        else:
            idx = i - 1 
            if not seat_taken[idx]:
                seat_taken[idx] = True
            else:
                empties = [k for k, taken in enumerate(seat_taken) if not taken]
                s = r.choice(empties)
                seat_taken[s] = True

    return seat_taken[N - 1]


def task1():
    rng = random.Random(222)  

    # try: N=20, simulate 5000 times
    N = 20
    num_sim = 5000
    results_b = [is_karen_upset(N, rng) for _ in range(num_sim)]
    num_upset = sum(results_b)
    prob_upset = num_upset / num_sim
    print("task1:")
    print("num_upset =", num_upset)
    print("prob_upset =", prob_upset)

    # generalize: N in {20,40,...,200}, each with 5000 sims
    N_values = list(range(20, 201, 20))
    probs = []
    for n in N_values:
        res = [is_karen_upset(n, rng) for _ in range(num_sim)]
        probs.append(sum(res) / num_sim)
    print("\ntask1: estimated probabilities by N")
    print(pd.DataFrame({"N": N_values, "prob_upset": probs}))
    # typically ~0.5 and not vary much with N 

    # modified scenario: every odd passenger is intoxicated. N=20, 50,000 sims
    num_sim_new = 50000
    N_mod = 20
    results_d = [is_karen_upset_odd_drunk(N_mod, rng) for _ in range(num_sim_new)]
    prob_upset_new = sum(results_d) / num_sim_new
    print("\ntask1:")
    print("prob_upset (odd passengers intoxicated, N=20) =", prob_upset_new)


# die-rolling experiment

def roll_dice(N: int, rng: random.Random = None) -> int:
    """
    Roll two fair dice N times.
    +1 point if both dice show even numbers in a trial; else +0.
    Return total score X after N trials.
    """
    r = rng or random
    score = 0
    for _ in range(N):
        d1 = r.randint(1, 6)
        d2 = r.randint(1, 6)
        if d1 % 2 == 0 and d2 % 2 == 0:
            score += 1
    return score


def task2():
    rng = random.Random(222212)
    num_sim_2 = 5000
    num_repeat = 100

    # mean
    total_scores = [roll_dice(num_repeat, rng) for _ in range(num_sim_2)]
    expected_value = float(np.mean(total_scores))
    print("\ntask2: E[X] ≈", expected_value)

    # variance 
    variance = float(np.var(total_scores, ddof=1))
    print("task2: Var(X) ≈", variance)

    # P(X > 25)
    prob_greater_25 = np.mean(np.array(total_scores) > 25)
    print("task2: P(X > 25) ≈", prob_greater_25)

    # You vs Friend (independent games)
    def simulation_once(N):
        x = roll_dice(N, rng)
        y = roll_dice(N, rng)
        if x > y:
            return "I win"
        elif y > x:
            return "Friend win"
        else:
            return "Draw"

    outcomes = [simulation_once(num_repeat) for _ in range(num_sim_2)]
    prob_win = outcomes.count("I win") / num_sim_2
    prob_friend_win = outcomes.count("Friend win") / num_sim_2
    prob_draw = outcomes.count("Draw") / num_sim_2

    print("task2:")
    print("P(I win)       ≈", prob_win)
    print("P(Friend wins) ≈", prob_friend_win)
    print("P(Draw)        ≈", prob_draw)
    # symmetry implies fairness: P(I win) ≈ P(Friend wins), P(Draw) small.


# matching number

def sim3(N: int, rng: random.Random = None) -> bool:
    """
    N individuals each pick an integer 1..100 with replacement.
    Return True if at least one duplicate exists; else False.
    """
    r = rng or random
    picks = [r.randint(1, 100) for _ in range(N)]
    return len(set(picks)) < N


def task3():
    rng = random.Random(222111)
    num_sims = 5000
    num = 1
    probability = 0.0
    # find smallest N such that P(duplicate) > 1/2
    while probability <= 0.5:
        probability = np.mean([sim3(num, rng) for _ in range(num_sims)])
        num += 1
    print("task3:")
    print("Smallest N with P(duplicate) > 1/2 ≈", num)


# main 
if __name__ == "__main__":
    task1()
    task2()
    task3()
