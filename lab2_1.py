import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

open('load_points.txt', 'w').close() 

class BallBinExperiment:
    def __init__(self, m, T, beta_values, d_values):
        self.m = m                  # Number of bins
        self.T = T                  # Number of trials
        self.beta_values = beta_values  # List of beta values for (1 + β)-choice strategies
        self.d_values = d_values        # List of d values for d-choice strategies

    def one_choice_allocation(self, n):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n):
            bin_index = np.random.randint(0, self.m)
            bins[bin_index] += 1
        return bins

    def two_choice_allocation(self, n):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n):
            choices = np.random.choice(self.m, 2, replace=False)
            min_bin = choices[np.argmin(bins[choices])]
            bins[min_bin] += 1
        return bins

    def beta_choice_allocation(self, n, beta):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n):
            if np.random.rand() < beta:
                # One-choice allocation with probability β
                bin_index = np.random.randint(0, self.m)
                bins[bin_index] += 1
            else:
                # Two-choice allocation with probability 1 - β
                choices = np.random.choice(self.m, 2, replace=False)
                min_bin = choices[np.argmin(bins[choices])]
                bins[min_bin] += 1
        return bins

    def d_choice_allocation(self, n, d):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n):
            choices = np.random.choice(self.m, d, replace=False)
            bin_index = choices[np.argmin(bins[choices])]
            bins[bin_index] += 1
        return bins

    def compute_gap(self, bins): #G_ap
        return np.max(bins) - np.mean(bins)

    def run_experiments(self, n_values):
        results = {'one_choice': [], 'two_choice': []}
        results.update({f'beta_{beta}': [] for beta in self.beta_values})
        results.update({f'{d}-choice': [] for d in self.d_values})

        for n in n_values:
            one_choice_gaps = []
            two_choice_gaps = []
            beta_choice_gaps = {beta: [] for beta in self.beta_values}
            d_choice_gaps = {d: [] for d in self.d_values}

            for _ in range(self.T):
                # One-choice
                one_choice_bins = self.one_choice_allocation(n)
                one_choice_gaps.append(self.compute_gap(one_choice_bins))

                # Two-choice
                two_choice_bins = self.two_choice_allocation(n)
                two_choice_gaps.append(self.compute_gap(two_choice_bins))

                # (1 + β)-choice for each beta value
                for beta in self.beta_values:
                    beta_bins = self.beta_choice_allocation(n, beta)
                    beta_choice_gaps[beta].append(self.compute_gap(beta_bins))

                # d-choice for each d value
                for d in self.d_values:
                    d_bins = self.d_choice_allocation(n, d)
                    d_choice_gaps[d].append(self.compute_gap(d_bins))

            results['one_choice'].append((np.mean(one_choice_gaps), np.std(one_choice_gaps)))
            results['two_choice'].append((np.mean(two_choice_gaps), np.std(two_choice_gaps)))

            for beta in self.beta_values:
                mean_gap = np.mean(beta_choice_gaps[beta])
                std_gap = np.std(beta_choice_gaps[beta])
                results[f'beta_{beta}'].append((mean_gap, std_gap))

            for d in self.d_values:
                mean_gap = np.mean(d_choice_gaps[d])
                std_gap = np.std(d_choice_gaps[d])
                results[f'{d}-choice'].append((mean_gap, std_gap))

        return results

    def plot_results(self, results, n_values):
        plt.figure(figsize=(12, 8))

        def plot_with_error_bars(data, label):
            mean_values = [entry[0] for entry in data]
            std_values = [entry[1] for entry in data]
            plt.errorbar(n_values, mean_values, yerr=std_values, label=label, fmt='o-')

            # Highlight light and heavy load points
            m = self.m
            idx_m = np.where(n_values == m)[0]
            if idx_m.size > 0:  # Check if m exists in n_values
                idx_m = idx_m[0]
                plt.scatter(n_values[idx_m], mean_values[idx_m], color='green', s=100, zorder=5)
                plt.annotate(f'{label}: {mean_values[idx_m]:.2f}', 
                             (n_values[idx_m], mean_values[idx_m]), 
                             xytext=(10, 10), textcoords='offset points')
                # Save light load point to file
                with open('load_points.txt', 'a') as f:
                    f.write(f'Light load - {label}: {mean_values[idx_m]:.2f}\n')

            idx_m2 = np.where(n_values == m**2)[0]
            if idx_m2.size > 0:  # Check if m**2 exists in n_values
                idx_m2 = idx_m2[0]
                plt.scatter(n_values[idx_m2], mean_values[idx_m2], color='red', s=100, zorder=5)
                plt.annotate(f'{label}: {mean_values[idx_m2]:.2f}', 
                             (n_values[idx_m2], mean_values[idx_m2]), 
                             xytext=(10, -10), textcoords='offset points')
                # Save heavy load point to file  
                with open('load_points.txt', 'a') as f:
                    f.write(f'Heavy load - {label}: {mean_values[idx_m2]:.2f}\n')

        # Plot one-choice and two-choice results
        plot_with_error_bars(results['one_choice'], 'One-Choice')
        plot_with_error_bars(results['two_choice'], 'Two-Choice')

        # Plot (1 + β)-choice results
        for beta in self.beta_values:
            plot_with_error_bars(results[f'beta_{beta}'], f'(1 + {beta})-Choice')

        # Plot d-choice results
        for d in self.d_values:
            plot_with_error_bars(results[f'{d}-choice'], f'{d}-Choice')

        # Add vertical lines for light and heavy load
        m = self.m
        plt.axvline(x=m, color='green', linestyle=':', label='Light-load (n = m)')
        plt.axvline(x=m**2, color='red', linestyle=':', label='Heavy-load (n = m²)')

        plt.xlabel('Number of Balls (n)')
        plt.ylabel('Gap (G_n)')
        plt.title('Ball Allocation Gap Analysis for Various Choice Strategies')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig('ball_allocation_gap_analysis.png')
        plt.show()


# Experiment Configuration
m = 100      
T = 20       
beta_values = [0.1, 0.5, 0.7]
d_values = [3, 4, 5]  # Different d values to explore
experiment = BallBinExperiment(m, T, beta_values, d_values)
n_values = np.arange(m, m**2 + m + 1, m * 10) 
n_values = np.append(n_values, m**2)   

# Run Experiments
results = experiment.run_experiments(n_values)
experiment.plot_results(results, n_values)
