import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class BallBinExperimentBatched:
    def __init__(self, m, T, beta_values, d_values):
        self.m = m                      # Number of bins
        self.T = T                      # Number of trials
        self.beta_values = beta_values  # List of beta values for (1 + β)-choice strategies
        self.d_values = d_values        # List of d values for d-choice strategies

    def one_choice_allocation_batched(self, n, batch_size):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n // batch_size):
            current_batch = np.zeros(self.m, dtype=int)
            for _ in range(batch_size):
                bin_index = np.random.randint(0, self.m)
                current_batch[bin_index] += 1
            bins += current_batch
        return bins

    def two_choice_allocation_batched(self, n, batch_size):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n // batch_size):
            current_batch = np.zeros(self.m, dtype=int)
            for _ in range(batch_size):
                choices = np.random.choice(self.m, 2, replace=False)
                min_bin = choices[np.argmin(bins[choices])]
                current_batch[min_bin] += 1
            bins += current_batch
        return bins

    def beta_choice_allocation_batched(self, n, batch_size, beta):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n // batch_size):
            current_batch = np.zeros(self.m, dtype=int)
            for _ in range(batch_size):
                if np.random.rand() < beta:
                    # One-choice allocation with probability β
                    bin_index = np.random.randint(0, self.m)
                    current_batch[bin_index] += 1
                else:
                    # Two-choice allocation with probability 1 - β
                    choices = np.random.choice(self.m, 2, replace=False)
                    min_bin = choices[np.argmin(bins[choices])]
                    current_batch[min_bin] += 1
            bins += current_batch
        return bins

    def d_choice_allocation_batched(self, n, batch_size, d):
        bins = np.zeros(self.m, dtype=int)
        for _ in range(n // batch_size):
            current_batch = np.zeros(self.m, dtype=int)
            for _ in range(batch_size):
                choices = np.random.choice(self.m, d, replace=False)
                bin_index = choices[np.argmin(bins[choices])]
                current_batch[bin_index] += 1
            bins += current_batch
        return bins

    def compute_gap(self, bins):
        return np.max(bins) - np.mean(bins)

    def run_experiments_batched(self, batch_sizes):
        results = {}
        
        for batch_size in batch_sizes:
            batch_results = {
                'one_choice': [],
                'two_choice': [],
            }
            batch_results.update({f'beta_{beta}': [] for beta in self.beta_values})
            batch_results.update({f'd_{d}': [] for d in self.d_values})

            # Generate n_values and ensure m and m^2 are included
            n_values = np.arange(self.m, self.m**2 + 1, batch_size)
            if self.m not in n_values:
                n_values = np.insert(n_values, 0, self.m)
            if self.m**2 not in n_values:
                n_values = np.append(n_values, self.m**2)

            for n in n_values:
                one_choice_gaps = []
                two_choice_gaps = []
                beta_choice_gaps = {beta: [] for beta in self.beta_values}
                d_choice_gaps = {d: [] for d in self.d_values}

                for _ in range(self.T):
                    # One-choice batched
                    one_choice_bins = self.one_choice_allocation_batched(n, batch_size)
                    one_choice_gaps.append(self.compute_gap(one_choice_bins))

                    # Two-choice batched
                    two_choice_bins = self.two_choice_allocation_batched(n, batch_size)
                    two_choice_gaps.append(self.compute_gap(two_choice_bins))

                    # (1 + β)-choice batched
                    for beta in self.beta_values:
                        beta_bins = self.beta_choice_allocation_batched(n, batch_size, beta)
                        beta_choice_gaps[beta].append(self.compute_gap(beta_bins))

                    # d-choice batched
                    for d in self.d_values:
                        d_bins = self.d_choice_allocation_batched(n, batch_size, d)
                        d_choice_gaps[d].append(self.compute_gap(d_bins))

                # Store the mean and std deviation of gaps for each allocation strategy
                batch_results['one_choice'].append((n, np.mean(one_choice_gaps), np.std(one_choice_gaps)))
                batch_results['two_choice'].append((n, np.mean(two_choice_gaps), np.std(two_choice_gaps)))

                for beta in self.beta_values:
                    mean_gap = np.mean(beta_choice_gaps[beta])
                    std_gap = np.std(beta_choice_gaps[beta])
                    batch_results[f'beta_{beta}'].append((n, mean_gap, std_gap))

                for d in self.d_values:
                    mean_gap = np.mean(d_choice_gaps[d])
                    std_gap = np.std(d_choice_gaps[d])
                    batch_results[f'd_{d}'].append((n, mean_gap, std_gap))
                    
            results[batch_size] = batch_results
            
        return results

    def plot_results(self, results):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        batch_sizes = list(results.keys())
        plot_titles = [f"Batch Size = {batch_size}" for batch_size in batch_sizes]

        colors = {
            'one_choice': 'blue',
            'two_choice': 'orange',
            'beta_0.1': 'green',
            'beta_0.5': 'red',
            'beta_0.7': 'purple',
            'd_3': 'cyan',
            'd_4': 'magenta',
            'd_5': 'brown'
        }

        for i, batch_size in enumerate(batch_sizes):
            ax = axs[i // 2, i % 2]
            batch_results = results[batch_size]
            n_values = [entry[0] for entry in batch_results['one_choice']]

            for key, data in batch_results.items():
                mean_values = [entry[1] for entry in data]
                std_values = [entry[2] for entry in data]
                ax.errorbar(n_values, mean_values, yerr=std_values, label=key.replace('_', ' ').title(), color=colors[key])

                # Highlight light load point (n = m) if it exists in n_values
                if self.m in n_values:
                    idx = n_values.index(self.m)
                    ax.scatter(n_values[idx], mean_values[idx], color='green', s=100)
                    ax.annotate(f"{key}:{mean_values[idx]:.2f}", 
                                (n_values[idx], mean_values[idx]), 
                                textcoords="offset points", xytext=(5, 5), ha='left', color='green')
                    # Save light load point to file
                    with open('load_points_b.txt', 'a') as f:
                        f.write(f'Light load - Batch size {batch_size} - {key}: {mean_values[idx]:.2f}\n')

                # Highlight heavy load point (n = m^2) if it exists in n_values
                if self.m**2 in n_values:
                    idx = n_values.index(self.m**2)
                    ax.scatter(n_values[idx], mean_values[idx], color='red', s=100)
                    ax.annotate(f"{key}:{mean_values[idx]:.2f}", 
                                (n_values[idx], mean_values[idx]), 
                                textcoords="offset points", xytext=(5, -10), ha='left', color='red')
                    # Save heavy load point to file
                    with open('load_points_b.txt', 'a') as f:
                        f.write(f'Heavy load - Batch size {batch_size} - {key}: {mean_values[idx]:.2f}\n')

            ax.set_title(plot_titles[i])
            ax.set_xlabel('Number of Balls (n)')
            ax.set_ylabel('Gap (G_n)')
            ax.legend()
            ax.grid(True, which="both", ls="-", alpha=0.2)

        plt.tight_layout()
        plt.savefig('batched_ball_allocation_gap_analysis.png')
        plt.show()


# Experiment Configuration
m = 100               
T = 10                
beta_values = [0.1, 0.5, 0.7]  
d_values = [3, 4, 5]            
batch_sizes = [m, 5 * m, 50 * m, m**2]  

experiment = BallBinExperimentBatched(m, T, beta_values, d_values)
results = experiment.run_experiments_batched(batch_sizes)
experiment.plot_results(results)
