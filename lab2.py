import numpy as np
import random
import matplotlib.pyplot as plt

open('load_points_k.txt', 'w').close() 

class BallBinExperimentWithKQueries:
    def __init__(self, m, T, k_values, beta_values, d_values):
        self.m = m                      
        self.T = T                      
        self.k_values = k_values        
        self.beta_values = beta_values  
        self.d_values = d_values

    def choose_bin_with_k_queries(self, bins, i, j, k):
        median_load = np.median(bins)
        load_25th_percentile = np.percentile(bins, 25)

        # First question: is the bin in the top 50% most loaded?
        is_i_above_median = bins[i] > median_load
        is_j_above_median = bins[j] > median_load

        if k == 1:
            # If answers differ, choose the one below the median load
            if is_i_above_median != is_j_above_median:
                return i if not is_i_above_median else j
            else:
                # If answers are the same, choose randomly
                return random.choice([i, j])

        elif k == 2:
            # If answers differ, choose the one below the median load
            if is_i_above_median != is_j_above_median:
                return i if not is_i_above_median else j
            else:
                # Second question: is the bin in the top 25% most loaded?
                is_i_in_25_percent = bins[i] > load_25th_percentile
                is_j_in_25_percent = bins[j] > load_25th_percentile

                if is_i_in_25_percent != is_j_in_25_percent:
                    # Choose the one that is not in the 25% most loaded
                    return i if not is_i_in_25_percent else j
                else:
                    # If answers are still the same, choose randomly
                    return random.choice([i, j])

    def two_choice_allocation_with_queries(self, n, k):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n):
            # Choose two bins randomly and use k-query strategy to pick one
            i, j = np.random.choice(self.m, 2, replace=False)
            chosen_bin = self.choose_bin_with_k_queries(bins, i, j, k)
            bins[chosen_bin] += 1
        return bins

    def beta_choice_allocation_with_queries(self, n, k, beta):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n):
            if np.random.rand() < beta:
                # With probability β, perform a one-choice allocation
                chosen_bin = np.random.randint(0, self.m)
            else:
                # With probability 1 - β, perform a two-choice allocation
                i, j = np.random.choice(self.m, 2, replace=False)
                chosen_bin = self.choose_bin_with_k_queries(bins, i, j, k)

            bins[chosen_bin] += 1
        return bins

    def d_choice_allocation_with_queries(self, n, k, d):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n):
            # Select `d` random bins and use k-query strategy on two of them
            choices = np.random.choice(self.m, d, replace=False)
            i, j = np.random.choice(choices, 2, replace=False)
            chosen_bin = self.choose_bin_with_k_queries(bins, i, j, k)
            bins[chosen_bin] += 1
        return bins

    def compute_gap(self, bins):
        """Calculate the gap G_n as max load - mean load."""
        return np.max(bins) - np.mean(bins)

    def run_experiments(self, n_values):
        results = {f'k={k}_two_choice': [] for k in self.k_values}
        for beta in self.beta_values:
            results.update({f'k={k}_beta_{beta}_choice': [] for k in self.k_values})
        for d in self.d_values:
            results.update({f'k={k}_d_{d}_choice': [] for k in self.k_values})

        for n in n_values:
            for k in self.k_values:
                # Run the two-choice strategy with k queries
                two_choice_gaps = []
                for _ in range(self.T):
                    bins = self.two_choice_allocation_with_queries(n, k)
                    two_choice_gaps.append(self.compute_gap(bins))
                results[f'k={k}_two_choice'].append((n, np.mean(two_choice_gaps), np.std(two_choice_gaps)))

                # Run the (1 + β)-choice strategy with k queries for each β
                for beta in self.beta_values:
                    beta_choice_gaps = []
                    for _ in range(self.T):
                        bins = self.beta_choice_allocation_with_queries(n, k, beta)
                        beta_choice_gaps.append(self.compute_gap(bins))
                    results[f'k={k}_beta_{beta}_choice'].append((n, np.mean(beta_choice_gaps), np.std(beta_choice_gaps)))

                # Run the d-choice strategy with k queries for each d
                for d in self.d_values:
                    d_choice_gaps = []
                    for _ in range(self.T):
                        bins = self.d_choice_allocation_with_queries(n, k, d)
                        d_choice_gaps.append(self.compute_gap(bins))
                    results[f'k={k}_d_{d}_choice'].append((n, np.mean(d_choice_gaps), np.std(d_choice_gaps)))

        return results
    
    
    def plot_results(self, results, n_values):
        # Separate the results for k=1 and k=2
        k1_results = {key: data for key, data in results.items() if 'k=1' in key}
        k2_results = {key: data for key, data in results.items() if 'k=2' in key}
        m = self.m  # Number of bins

        # Plot for k=1
        fig, ax = plt.subplots(figsize=(8, 6))
        for strategy, data in k1_results.items():
            n_vals = np.atleast_1d([entry[0] for entry in data])  # Ensure n_vals is 1D
            mean_gaps = [entry[1] for entry in data]
            std_gaps = [entry[2] for entry in data]
            ax.errorbar(n_vals, mean_gaps, yerr=std_gaps, label=strategy.replace('_', ' '), fmt='o-')

            # Highlight light load (n = m) and heavy load (n = m^2) points
            idx_m = np.where(n_vals == m)[0]
            if idx_m.size > 0:
                idx_m = idx_m[0]
                ax.scatter(n_vals[idx_m], mean_gaps[idx_m], color='green', s=100, zorder=5)
                ax.annotate(f'{strategy}: {mean_gaps[idx_m]:.2f}', 
                            (n_vals[idx_m], mean_gaps[idx_m]), 
                            xytext=(10, 10), textcoords='offset points', color='green')
                with open('load_points_k.txt', 'a') as f:
                    f.write(f'Light load (k=1) - {strategy}: {mean_gaps[idx_m]:.2f}\n')

            idx_m2 = np.where(n_vals == m**2)[0]
            if idx_m2.size > 0:
                idx_m2 = idx_m2[0]
                ax.scatter(n_vals[idx_m2], mean_gaps[idx_m2], color='red', s=100, zorder=5)
                ax.annotate(f'{strategy}: {mean_gaps[idx_m2]:.2f}', 
                            (n_vals[idx_m2], mean_gaps[idx_m2]), 
                            xytext=(10, -10), textcoords='offset points', color='red')
                with open('load_points_k.txt', 'a') as f:
                    f.write(f'Heavy load (k=1) - {strategy}: {mean_gaps[idx_m2]:.2f}\n')

        ax.set_xlabel('Number of Balls (n)')
        ax.set_ylabel('Gap (G_n)')
        ax.set_title("Ball Allocation with k=1 Queries")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig('ball_allocation_k1_queries.png')  
        plt.close(fig)

        # Plot for k=2
        fig, ax = plt.subplots(figsize=(8, 6))
        for strategy, data in k2_results.items():
            n_vals = np.atleast_1d([entry[0] for entry in data])  # Ensure n_vals is 1D
            mean_gaps = [entry[1] for entry in data]
            std_gaps = [entry[2] for entry in data]
            ax.errorbar(n_vals, mean_gaps, yerr=std_gaps, label=strategy.replace('_', ' '), fmt='o-')

            idx_m = np.where(n_vals == m)[0]
            if idx_m.size > 0:
                idx_m = idx_m[0]
                ax.scatter(n_vals[idx_m], mean_gaps[idx_m], color='green', s=100, zorder=5)
                ax.annotate(f'{strategy}: {mean_gaps[idx_m]:.2f}', 
                            (n_vals[idx_m], mean_gaps[idx_m]), 
                            xytext=(10, 10), textcoords='offset points', color='green')
                with open('load_points_k.txt', 'a') as f:
                    f.write(f'Light load (k=2) - {strategy}: {mean_gaps[idx_m]:.2f}\n')

            idx_m2 = np.where(n_vals == m**2)[0]
            if idx_m2.size > 0:
                idx_m2 = idx_m2[0]
                ax.scatter(n_vals[idx_m2], mean_gaps[idx_m2], color='red', s=100, zorder=5)
                ax.annotate(f'{strategy}: {mean_gaps[idx_m2]:.2f}', 
                            (n_vals[idx_m2], mean_gaps[idx_m2]), 
                            xytext=(10, -10), textcoords='offset points', color='red')
                with open('load_points_k.txt', 'a') as f:
                    f.write(f'Heavy load (k=2) - {strategy}: {mean_gaps[idx_m2]:.2f}\n')

        ax.set_xlabel('Number of Balls (n)')
        ax.set_ylabel('Gap (G_n)')
        ax.set_title("Ball Allocation with k=2 Queries")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig('ball_allocation_k2_queries.png')  
        plt.close(fig)


# Experiment configuration
m = 100  
T = 10  
k_values = [1, 2]
beta_values = [0.1, 0.5, 0.7]
d_values = [3, 4, 5]
n_values = np.arange(m, m**2, m * 10)
n_values = np.unique(np.append(n_values, [m, m**2])) 

# Run experiment and plot results
experiment = BallBinExperimentWithKQueries(m, T, k_values, beta_values, d_values)
results = experiment.run_experiments(n_values)
experiment.plot_results(results, n_values)