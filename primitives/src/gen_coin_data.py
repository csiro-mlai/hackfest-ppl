"""
Synthetic data for coin toss model
"""

import csv
import random


def gen_coin_tosses(n=1000, fairness=0.6, seed=11):
    """
    Generate coin tosses with a given fairness.
    """
    rstate = random.getstate()
    if seed is not None:
        random.seed(seed)
    tosses = random.choices([0, 1], k=n, weights=[1-fairness, fairness])
    random.setstate(rstate)
    return tosses

def write_coin_tosses(tosses):
    """
    Write coin tosses to a csv file.
    """
    with open('coin_tosses.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['toss'])
        for toss in tosses:
            writer.writerow([toss])

if __name__ == '__main__':
    write_coin_tosses(gen_coin_tosses())