import re
import importlib
import numpy as np
import pandas as pd
import GraphGeneration
import ARLFunc
import GridEvaluation
import seaborn as sns
import matplotlib.pyplot as plt

# Reload the module
importlib.reload(GraphGeneration)
importlib.reload(ARLFunc)
importlib.reload(GridEvaluation)

# reimport functions from the reloaded module
from GraphGeneration import generate_cusum_chart, generate_ewma_chart
from ARLFunc import arl_cusum, arl_ewma
from GridEvaluation import GridDataEvaluate, simulate_stream_data, stream_data_plot

# For displaying the full pd.df
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# ------------------Testing function for arl and graph-------------------

# Initial values
SEED = 666
BURNIN = 50
CUSUM_K = 1.50
CUSUM_H = 1.61
EWMA_RHO = 0.10
EWMA_K = 2.814
np.random.seed(SEED)

# Data with difference mean (increase)
data_1 = np.append(np.random.normal(size=400), 
                       np.random.normal(size=400,loc=3.))
generate_cusum_chart(data_1, BURNIN, CUSUM_K, CUSUM_H, true_cp=400)
print(arl_cusum(data_1, BURNIN, CUSUM_K, CUSUM_H, true_cp=400))
generate_ewma_chart(data_1, BURNIN, EWMA_RHO, EWMA_K, true_cp=400)
print(arl_ewma(data_1, BURNIN, EWMA_RHO, EWMA_K, true_cp=400))

# Data with difference mean (decrease)
data_2 = np.append(np.random.normal(size=400), 
                       np.random.normal(size=400,loc=-3.))
generate_cusum_chart(data_2, BURNIN, CUSUM_K, CUSUM_H, true_cp=400)
print(arl_cusum(data_2, BURNIN, CUSUM_K, CUSUM_H, true_cp=400))
generate_ewma_chart(data_2, BURNIN, EWMA_RHO, EWMA_K, true_cp=400)
print(arl_ewma(data_2, BURNIN, EWMA_RHO, EWMA_K, true_cp=400))

# Data with same distribution
data_3 = np.random.normal(size=600)
generate_cusum_chart(data_3, BURNIN, CUSUM_K, CUSUM_H, true_cp=None)
print(arl_cusum(data_3, BURNIN, CUSUM_K, CUSUM_H, true_cp=None))
generate_ewma_chart(data_3, BURNIN, EWMA_RHO, EWMA_K, true_cp=None)
print(arl_ewma(data_3, BURNIN, EWMA_RHO, EWMA_K, true_cp=None))

# Data with same mean but different variance, no change point
data_4 = np.append(np.random.normal(size=300), np.random.normal(scale=np.sqrt(9), size=300))
generate_cusum_chart(data_4, BURNIN, CUSUM_K, CUSUM_H, true_cp=None)
print(arl_cusum(data_4, BURNIN, CUSUM_K, CUSUM_H, true_cp=None))
generate_ewma_chart(data_4, BURNIN, EWMA_RHO, EWMA_K, true_cp=None)
print(arl_ewma(data_4, BURNIN, EWMA_RHO, EWMA_K, true_cp=None))

# Data with different mean and different variance
data_5 = np.append(np.random.normal(size=300), np.random.normal(loc=3., scale=np.sqrt(9), size=300))
generate_cusum_chart(data_5, BURNIN, CUSUM_K, CUSUM_H, true_cp=300)
print(arl_cusum(data_5, BURNIN, CUSUM_K, CUSUM_H, true_cp=300))
generate_ewma_chart(data_5, BURNIN, EWMA_RHO, EWMA_K, true_cp=300)
print(arl_ewma(data_5, BURNIN, EWMA_RHO, EWMA_K, true_cp=300))

# ------------------End-------------------



# ------------------Testing function for grid_params_eval function-------------------
# Setup initial values
n_sam_bef_cp = 400
n_sam_aft_cp = 500
gap_sizes = [1, 5, 10]
variances = [1, 4, 9]
seeds = [111, 666, 999]
BURNIN = 50
cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
# simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
grideval = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, 
                                      seeds, BURNIN, cusum_params_list, ewma_params_list)

per_table, per_summary = grideval.grid_params_eval()
per_summary
grideval.plot_ARL0_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
grideval.plot_ARL1_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
grideval.plot_best_models()


from scipy.stats import norm

class OutlierInjector:
    def __init__(self, data, n_sam_bef_cp:int, n_sam_aft_cp:int, burnin:int, in_control_var, out_control_mean, out_control_var, alpha, 
                 in_control_mean=0, outlier_ratio=0.01, outlier_positions=['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']):
        self.data = data
        self.n_sam_bef_cp = n_sam_bef_cp
        self.n_sam_aft_cp = n_sam_aft_cp
        self.burnin = burnin
        self.in_control_mean = in_control_mean
        self.in_control_var = in_control_var
        self.in_control_std = np.sqrt(in_control_var)
        self.out_control_mean = out_control_mean
        self.out_control_var = out_control_var
        self.out_control_std = np.sqrt(out_control_var)
        self.alpha = alpha
        self.outlier_ratio = outlier_ratio
        self.outlier_positions = outlier_positions
        self.outlier_indices = []


    def calculate_thresholds(self):
        in_control_lower_threshold = norm.ppf(self.alpha/2, loc=self.in_control_mean, scale=self.in_control_std)
        in_control_upper_threshold = norm.ppf(1 - self.alpha/2, loc=self.in_control_mean, scale=self.in_control_std)
        out_control_lower_threshold = norm.ppf(self.alpha/2, loc=self.out_control_mean, scale=self.out_control_std)
        out_control_upper_threshold = norm.ppf(1 - self.alpha/2, loc=self.out_control_mean, scale=self.out_control_std)
        return in_control_lower_threshold, in_control_upper_threshold, out_control_lower_threshold, out_control_upper_threshold

    def insert_outliers(self):
        """
        Inserts outliers into the data at specified positions.
        """
        num_outliers = int(self.outlier_ratio * (self.n_sam_bef_cp + self.n_sam_aft_cp))
        print(num_outliers)
        in_control_lower_threshold, in_control_upper_threshold, out_control_lower_threshold, out_control_upper_threshold = self.calculate_thresholds()
        print(in_control_lower_threshold)
        print(in_control_upper_threshold)
        if self.outlier_positions[0] == 'in-control':
            # specify indices for 'in-control' period
            in_control_indices = np.arange(self.burnin, self.n_sam_bef_cp)
            self.add_outliers(num_outliers, in_control_indices, in_control_lower_threshold, in_control_upper_threshold)
            np.arange(50)
        
        elif self.outlier_positions[0] == 'out-of-control':
            # specify indices for 'out-of-control' period
            out_of_control_indices = np.arange(self.n_sam_bef_cp, self.n_sam_aft_cp)
            self.add_outliers(num_outliers, out_of_control_indices, out_control_lower_threshold, out_control_upper_threshold)

        elif self.outlier_positions[0] == 'burn-in':
            # specify indices for 'burn-in' period
            burnin_indices = np.arange(self.burnin)
            self.add_outliers(num_outliers, burnin_indices, in_control_lower_threshold, in_control_upper_threshold)
        
        elif self.outlier_positions[0] == 'both_in_and_out':
            # specify indices for 'both_in_and_out' period
            both_in_and_out_indices = np.arange(self.burnin,self.n_sam_bef_cp + self.n_sam_aft_cp)
            self.add_outliers_for_both(num_outliers, both_in_and_out_indices, in_control_lower_threshold, 
                                       in_control_upper_threshold, out_control_lower_threshold, out_control_upper_threshold)
        return self.data

    def add_outliers(self, num_outliers, indices, lower_threshold, upper_threshold):
        outlier_indices = np.random.choice(indices, num_outliers, replace=False)
        self.outlier_indices = np.sort(outlier_indices)
        for index in outlier_indices:
            if np.random.random() < 0.5:
                # Generate a lower outlier with prob=0.5
                outlier_value = lower_threshold * np.random.uniform(1, 1.2)
            else:
                # Generate an upper outlier
                outlier_value = upper_threshold * np.random.uniform(1, 1.2)
            self.data[index] = outlier_value

    def add_outliers_for_both(self, num_outliers, indices, ic_lower_threshold, ic_upper_threshold, oc_lower_threshold, oc_upper_threshold):
        outlier_indices = np.random.choice(indices, num_outliers, replace=False)
        self.outlier_indices = np.sort(outlier_indices)
        for index in outlier_indices:
            # seperate the case for in-control or out-control
            if index < self.n_sam_bef_cp:
                if np.random.random() < 0.5:
                    # Generate a lower outlier with prob=0.5
                    outlier_value = ic_lower_threshold * np.random.uniform(1, 1.2) # randomly generate more extreme value
                else:
                    # Generate an upper outlier
                    outlier_value = ic_upper_threshold * np.random.uniform(1, 1.2)
            else:
                if np.random.random() < 0.5:
                    # Generate a lower outlier with prob=0.5
                    outlier_value = oc_lower_threshold * np.random.uniform(1, 1.2)
                else:
                    # Generate an upper outlier
                    outlier_value = oc_upper_threshold * np.random.uniform(1, 1.2)
            self.data[index] = outlier_value

np.sort(np.random.choice(np.arange(10), 2, replace=False))

# ------------------End-------------------

# -------------------testing for NaN value--------------------

BURNIN = 50
seeds = [111, 666, 999]
np.random.seed(999)

# CUSUM (0.7,20) -10 1 Inf for one of ARL0
# CUSUM (0.7,20)	-5	4	Inf for one of ARL0
# CUSUM (0.7,20)	1	1  
# CUSUM (1.5,1.61)	MG:1 Var:9
data_1 = np.append(np.random.normal(size=400, scale=3), 
                       np.random.normal(size=500,loc=1, scale=3))
generate_cusum_chart(data_1, BURNIN, cusum_k=1.5, cusum_h=1.61, true_cp=400)
print(arl_cusum(data_1, BURNIN, cusum_k=1.5, cusum_h=1.61, true_cp=400))
generate_ewma_chart(data_1, BURNIN, EWMA_RHO, EWMA_K, true_cp=None)
print(arl_ewma(data_1, BURNIN, EWMA_RHO, EWMA_K, true_cp=None))


data_1 = np.append(np.random.normal(size=400, scale=3), 
                       np.random.normal(size=500,loc=1, scale=3))
outinj = OutlierInjector(data_1.copy(),n_sam_bef_cp, n_sam_aft_cp, BURNIN, 9, 1, 9, 0.00000001, outlier_positions=['in-control'])
out_data = outinj.insert_outliers()
plt.figure(figsize=(12, 6))

plt.plot(out_data, color='gold', label='Data with Outliers')
plt.plot(data_1, label='Original Data')

plt.scatter(outinj.outlier_indices, out_data[outinj.outlier_indices], color='red', zorder=5, label='Outliers')

plt.title('Comparison between Original Data and Data with Outliers')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.show()

# -------------------streaming data simulation--------------------

data_stream, tau_list, mu_list, size_list = simulate_stream_data(v=50, G=50, D=50, M=10, S=[0.25, 0.5, 1, 3], sigma=1, seed=666)
stream_data_plot(data_stream, tau_list)
tau_list.shape
size_list.sum()
data_stream.shape
# ------------------End-------------------


