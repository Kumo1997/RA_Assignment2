import numpy as np
import matplotlib.pyplot as plt

class BallBinExperiment:
    def __init__(self, m, n, T):
        self.m = m  # Number of bins
        self.n = n  # Number of balls
        self.T = T  # Number of trials
        self.gaps = []

    def standard_allocation(self):
        bins = np.zeros(self.m)
        for _ in range(self.n):
            bin_index = np.random.randint(0, self.m)
            bins[bin_index] += 1
        return bins

    def two_choice_allocation(self):
        bins = np.zeros(self.m)
        for _ in range(self.n):
            choices = np.random.choice(self.m, 2, replace=False)
            bin_index = choices[np.argmax(bins[choices])]
            bins[bin_index] += 1
        return bins

    def beta_choice_allocation(self, beta):
        bins = np.zeros(self.m)
        for _ in range(self.n):
            choices = np.random.choice(self.m, 1 + int(beta), replace=False)
            bin_index = choices[np.argmax(bins[choices])]
            bins[bin_index] += 1
        return bins
    
    # def d_choice_allocation(self, d):
    #     bins = np.zeros(self.m)


    def run_experiment(self, allocation_strategy, *args):
        for _ in range(self.T):
            if allocation_strategy == 'standard':
                bins = self.standard_allocation()
            elif allocation_strategy == 'two_choice':
                bins = self.two_choice_allocation()
            elif allocation_strategy == 'beta_choice':
                bins = self.beta_choice_allocation(*args)
            else:
                raise ValueError("Unknown allocation strategy")
            self.gaps.append(np.max(bins) - np.min(bins))

    def plot_results(self):
        plt.figure()
        plt.plot(self.gaps)
        plt.title('Evolution of Gap Gn')
        plt.xlabel('Trial')
        plt.ylabel('Gap Gn')
        plt.savefig('gap_evolution.png')
        plt.show()

def main():
    m = 20  # Number of bins
    n_light = m  # Light-loaded scenario
    n_heavy = m ** 2  # Heavy-loaded scenario
    T = 100  # Number of trials

    # Light-loaded scenario
    experiment_light = BallBinExperiment(m, n_light, T)
    experiment_light.run_experiment('standard')
    experiment_light.run_experiment('two_choice')
    for i in range(m):
        experiment_light.run_experiment('beta_choice', i) 
    experiment_light.plot_results()

    # Heavy-loaded scenario
    experiment_heavy = BallBinExperiment(m, n_heavy, T)
    experiment_heavy.run_experiment('standard')
    experiment_heavy.run_experiment('two_choice')
    for j in range(m):
        experiment_heavy.run_experiment('beta_choice', j)  # different value of β loop
    experiment_heavy.plot_results()


if __name__ == "__main__":
    main()
