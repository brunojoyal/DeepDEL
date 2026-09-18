import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import norm
import argparse, os, json, random, csv
from deepdelgfn.rewards import reward







def main():
    ap = argparse.ArgumentParser("Simulate sub-library sampling by random sampling")
    ap.add_argument("--csv", required=True, help="Dataset of samples")
    ap.add_argument("--k", required=False, type=int, default=None, help="top-k")
    ap.add_argument("--b", required=True, type=int, help="Block size")
    ap.add_argument("--M", required=False, type=int, default=1000, help="Num of samples")
    ap.add_argument("--value", required=False, type=float, default=None)
    ap.add_argument("--threshold", required=False, default=None, type=float, help="threshold")
    ap.add_argument("--with-replacement", action='store_true')
    
    
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    results = []
    for _ in range(args.M):
        sampled = df.sample(n=args.b**3, random_state=None, replace=args.with_replacement)
        if args.k is not None:
            r = reward(np.array(sampled["docking_score"], dtype=float), "topk_mean", args.k, threshold=None)
        if args.threshold is not None:
            v = np.array(sampled["docking_score"], dtype=float)
            r = reward(v, "threshold", k=None, threshold = args.threshold)
        results.append(r)

    # Plot histogram
    mu, sigma = np.mean(np.array(results)), np.std(np.array(results))
    print(mu, sigma)
    if args.value is not None:
        print(f"Library with score {args.value} is one-in-exp({int(1-norm.sf(args.value, mu, sigma))**(-1)})")
    plt.hist(results, bins=50, edgecolor="black")
    if args.k is not None:
        plt.xlabel("Reward")
    if args.threshold is not None: 
        plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of pseudo-library scores, {args.b}^3={args.b**3} trimers, \n({args.M} samples)")
    plt.tight_layout()
    plt.savefig("outputs/fake_library_scores.png")


if __name__ == "__main__":
    main()
