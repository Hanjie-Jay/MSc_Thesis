import os
import importlib
import numpy as np
import pandas as pd
import GraphGeneration
import ARLFunc
import GridEvaluation
import Outliers
import ControlChartFunc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Reload the module and reimport functions from the reloaded module
importlib.reload(GraphGeneration)
from GraphGeneration import generate_cusum_chart, generate_ewma_chart
importlib.reload(ARLFunc)
from ARLFunc import arl_cusum, arl_ewma, arl_robust_mean, combine_alert_ind, compute_arl0, compute_arl1

importlib.reload(GridEvaluation)
from GridEvaluation import GridDataEvaluate, simulate_stream_data, stream_data_plot

importlib.reload(Outliers)
from Outliers import OutlierInjector

importlib.reload(ControlChartFunc)
from ControlChartFunc import RobustMethods, ControlChart


# ------------------Testing function for the grid_robust_params_eval function-------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 400)
# Setup initial values
n_sam_bef_cp = 500
n_sam_aft_cp = 400
gap_sizes = [1, 5, 10]
variances = [1, 4, 9]
# seeds = [111, 222, 333, 666, 999]
seeds = [111, 666, 999]
BURNIN = 100
cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
# z_list = [1.6449, 1.96, 2.5759]
# alpha_list = [1, 1.5, 2, 2.5, 3]
z_list = [1.64, 1.96]
alpha_list = [1.5, 2, 2.5]
tm_params_list = [(0.1, 10), (0.1, 15), (0.1, 20), (0.15, 10), (0.15, 15), (0.15, 20), (0.2, 10), (0.2, 15), (0.2, 20)]
wm_params_list = [(0.1, 10), (0.1, 15), (0.1, 20), (0.15, 10), (0.15, 15), (0.15, 20), (0.2, 10), (0.2, 15), (0.2, 20)]
swm_params_list = [10, 15, 20, 25, 30]
ctm_params_list = [(0.1, 10), (0.1, 15), (0.1, 20), (0.15, 10), (0.15, 15), (0.15, 20), (0.2, 10), (0.2, 15), (0.2, 20)]
valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
outlier_position = valid_positions[2]
beta = 1e-5
outlier_ratio = 0.05
asymmetric_ratio = 0.25
# simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
grideval = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, 
                            seeds, BURNIN, cusum_params_list, ewma_params_list, z_list, alpha_list,
                             tm_params_list, wm_params_list, swm_params_list, ctm_params_list, 
                             outlier_position, beta, outlier_ratio, asymmetric_ratio)
rob_per_table, rob_per_summary = grideval.grid_robust_params_eval()

grideval.plot_robust_ARL0_graphs(True)
grideval.plot_robust_ARL1_graphs(True)
grideval.plot_best_models(True)
tm_table = rob_per_table[rob_per_table['Model (Parameters)'].str.contains('TM')]
tm_params = tm_table['Model (Parameters)'].unique()

rob_per_table[['Model', 'Parameters']] = rob_per_table['Model (Parameters)'].str.split(" ", n = 1, expand = True)

# Get the tables for each method
tm_table = rob_per_table[rob_per_table['Model'] == 'TM'] # Select rows that have TM
wm_table = rob_per_table[rob_per_table['Model'] == 'WM'] # Select rows that have WM
swm_table = rob_per_table[rob_per_table['Model'] == 'SWM'] # Select rows that have SWM
ctm_table = rob_per_table[rob_per_table['Model'] == 'CTM'] # Select rows that have CTM
# unique model parameters for all methods
tm_params = tm_table['Model (Parameters)'].unique()
wm_params = wm_table['Model (Parameters)'].unique()
swm_params = swm_table['Model (Parameters)'].unique()
ctm_params = ctm_table['Model (Parameters)'].unique()
plt.figure(figsize=(14, 8))
ax = sns.boxplot(data=tm_table, x='Gap Size', y='ARL0', hue='Data Var')
ax.legend(fontsize=14)
ax.tick_params(labelsize=14)
plt.title(f'$ARL_0$ Values of TM Model in Streaming Data Without Outliers', fontsize=18)
plt.ylabel('$ARL_0$', fontsize=14)
plt.xlabel('Gap Size', fontsize=14)
plt.show()
# ------------------Testing function for the arl_robust_mean function-------------------
burnin = 50
window_length = 25
trimmed_ratio = 0.15
winsorized_ratio = 0.15
cosine_ratio = 0.15
trimmed_window_length = 20
winsorized_window_length = 25
cosine_window_length = 25
z_val = 2.575 # 99%
alpha_val = 2 # small h, sansitive
true_cp = 500
# data with no outliers
data_1 = np.append(np.random.normal(size=true_cp), 
                       np.random.normal(size=400,loc=3.))
robust_arl_results = arl_robust_mean(data_1, burnin, window_length, trimmed_ratio, winsorized_ratio,
                                     cosine_ratio, trimmed_window_length, winsorized_window_length,
                                     cosine_window_length, z_val, alpha_val, true_cp)
# data with outliers
n_sam_bef_cp = 500
n_sam_aft_cp = 400
variance = 9
burnin = 100
gap_size = 5
alpha = 0.001
valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
outlier_position = valid_positions[2]
outlier_ratio = 0.05
asymmetric_ratio = 0.25
data_1 = np.append(np.random.normal(size=n_sam_bef_cp, scale=np.sqrt(variance)), 
                       np.random.normal(size=n_sam_aft_cp,loc=gap_size, scale=np.sqrt(variance)))
outinj = OutlierInjector(data_1 ,n_sam_bef_cp, n_sam_aft_cp, burnin, variance, 
                         gap_size, variance, alpha, outlier_position, outlier_ratio, asymmetric_ratio=0.25)
out_data = outinj.insert_outliers()
robust_arl_results_out = arl_robust_mean(out_data, burnin, window_length, trimmed_ratio, winsorized_ratio,
                                     cosine_ratio, trimmed_window_length, winsorized_window_length,
                                     cosine_window_length, z_val, alpha_val, n_sam_bef_cp)


# ------------------Testing function for the new ControlChart class with robust method-------------------
data_mean = 0
std_dev = 1
alpha = 3
1 - (2 * (1 - norm.cdf(alpha * std_dev + data_mean, loc=data_mean, scale=std_dev))) # prob_a_sigma_normal

probability = 0.995
norm.ppf(probability, loc=data_mean, scale=std_dev)

# data with no outliers
burnin = 30
burnin_data = np.random.normal(size=burnin)
random_data = np.append(np.random.normal(size=30), np.random.normal(loc=3., size=30))
robust_method_control_chart = ControlChart(random_data)
robust_method_control_chart.compute_robust_methods_mean_seq(median_window_length=10, trimmed_ratio=0.15, 
                                                            winsorized_ratio=0.15, cosine_ratio=0.15, 
                                                            trimmed_window_length=15, winsorized_window_length=18,
                                                             cosine_window_length=20, burnin_data=burnin_data)
# data with outliers
burnin = 30
valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
outlier_position = valid_positions[1]
alpha = 1e-5
outlier_ratio = 0.2
asymmetric_ratio = 0.25
random_data = np.append(np.random.normal(size=60, scale=np.sqrt(1)), 
                       np.random.normal(size=40,loc=3., scale=np.sqrt(1)))
outinj = OutlierInjector(random_data ,60, 40, burnin, 1, 
                         3, 1, 1e-4, outlier_position, outlier_ratio, asymmetric_ratio=0.25)
out_data = outinj.insert_outliers()
robust_method_control_chart = ControlChart(out_data[burnin:])
robust_method_control_chart.compute_robust_methods_mean_seq(median_window_length=10, trimmed_ratio=0.15, 
                                                            winsorized_ratio=0.15, cosine_ratio=0.15, 
                                                            trimmed_window_length=15, winsorized_window_length=18,
                                                             cosine_window_length=20, burnin_data=out_data[:burnin])

z_val = 2.575 # 99%
h_val = 2 # small h, sansitive

swm_CI_s, swm_CI_t, swm_CI_au, swm_CI_al = robust_method_control_chart.sliding_window_median_CI_val(z_val=z_val, h_val=h_val, mu=0, sigma=1)
swm_CI_ind = robust_method_control_chart.sliding_window_median_CI_detect(z_val=z_val, h_val=h_val, mu=0, sigma=1)
tm_CI_s, tm_CI_t, tm_CI_au, tm_CI_al = robust_method_control_chart.trimmed_mean_CI_val(z_val=z_val, h_val=h_val, mu=0, sigma=1)
tm_CI_ind = robust_method_control_chart.trimmed_mean_CI_detect(z_val=z_val, h_val=h_val, mu=0, sigma=1)
wm_CI_s, wm_CI_t, wm_CI_au, wm_CI_al = robust_method_control_chart.winsorized_mean_CI_val(z_val=z_val, h_val=h_val, mu=0, sigma=1)
wm_CI_ind = robust_method_control_chart.winsorized_mean_CI_detect(z_val=z_val, h_val=h_val, mu=0, sigma=1)
cpm_CI_s, cpm_CI_t, cpm_CI_au, cpm_CI_al = robust_method_control_chart.cosine_tapered_mean_CI_val(z_val=z_val, h_val=h_val, mu=0, sigma=1)
cpm_CI_ind = robust_method_control_chart.cosine_tapered_mean_CI_detect(z_val=z_val, h_val=h_val, mu=0, sigma=1)


# ------------------Testing function for RobustMethods class-------------------
# Compute lower and upper cutoff indices
from scipy.stats import trim_mean
from scipy.stats import mstats
from scipy.signal import tukey
sorted_data = np.array([1,3,4,5,6,8,9,11,19])
ratio = 0.2
lower_ind = int(np.floor(len(sorted_data)) * ratio)
upper_ind = len(sorted_data) - lower_ind
# Trim the data and compute the mean
trimmed_data = sorted_data[lower_ind:upper_ind]
trimmed_mean = np.mean(trimmed_data)
trim_mean(sorted_data, ratio)
winsorized_data = mstats.winsorize(sorted_data,limits=[ratio, ratio])
winsorized_mean = np.mean(winsorized_data)
my_winsorized_data = sorted_data
my_winsorized_data[:lower_ind] = sorted_data[lower_ind]
my_winsorized_data[upper_ind:] = sorted_data[upper_ind-1]
# Compute the mean
winsorized_mean = np.mean(my_winsorized_data)
# Sliding window median
window_length = 4
medians = [] # for store a list of median values
for i in range(len(sorted_data) - window_length + 1):
    window_median = np.median(sorted_data[i : i + window_length])
    medians.append(window_median)

weights = tukey(len(sorted_data), ratio) 
# Apply the weight to the sorted data
cosine_tapered_data = weights * sorted_data
# Compute the mean
cosine_tapered_mean = np.mean(cosine_tapered_data)
burnin = 10
burnin_data = np.random.normal(size=burnin)
random_data = np.append(np.random.normal(size=20), np.random.normal(loc=-3., size=10))

robust_method = RobustMethods(random_data)
sliding_window_median = robust_method.sliding_window_median(window_length=12) # could not be None here
robust_mean_seq = robust_method.compute_mean_sequence(trimmed_ratio=0.20, winsorized_ratio=0.20, cosine_ratio=0.20, 
                                                      trimmed_window_length=12, winsorized_window_length=12, cosine_window_length=12)

trimmed_mean = robust_mean_seq['trimmed']
winsorized_mean = robust_mean_seq['winsorized']
cosine_tapered_mean = robust_mean_seq['cosine']
trimmed_mean.shape

# For the robust variance
robust_var_seq = robust_method.compute_variance_sequence(winsorized_ratio=0.2)
mad_sd_seq = robust_var_seq['mad'] * 1.4826
iqr_sd_seq = robust_var_seq['iqr'] * 0.7413
winsorized_var = robust_var_seq['winsorized']


# ------------------Testing function for grid_params_eval function without outlier-------------------
# For displaying the full pd.df
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
# Setup initial values
n_sam_bef_cp = 500
n_sam_aft_cp = 400
gap_sizes = [1, 5, 10]
variances = [1, 4, 9]
seeds = [111, 222, 333, 666, 999]
BURNIN = 100
cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
# simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
grideval = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, 
                            seeds, BURNIN, cusum_params_list, ewma_params_list, None)
per_table, per_summary = grideval.grid_C_E_params_eval()

per_summary
per_table

grideval.plot_ARL0_graphs(save=True)
grideval.plot_ARL1_graphs(save=True)
grideval.plot_best_models(save=True)
# ------------------End-------------------




# -------------------testing for outliers generation in GridEvaluation class--------------------
# Setup initial values
n_sam_bef_cp = 500
n_sam_aft_cp = 400
gap_sizes = [1, 5, 10]
variances = [1, 4, 9]
seeds = [111, 222, 333, 666, 999]
BURNIN = 100
cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
outlier_position = valid_positions[3]
alpha = 1e-5
outlier_ratio = 0.05
asymmetric_ratio = 0.25
# simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
grideval_outliers = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, seeds, BURNIN,
                             cusum_params_list, ewma_params_list, outlier_position, alpha, outlier_ratio, asymmetric_ratio)
per_table, per_summary = grideval_outliers.grid_C_E_params_eval()
grideval_outliers.plot_ARL0_graphs(save=True)
grideval_outliers.plot_ARL1_graphs(save=True)
grideval_outliers.plot_best_models(save=True)
outlier_grid_data = grideval_outliers.generate_with_outliers_grid_data(seeds[0])
outlier_grid_data[0][0].shape
per_table.iloc[50]
# ------------------End-------------------

# -------------------testing for outliers class--------------------
n_sam_bef_cp = 500
n_sam_aft_cp = 400
variance = 4
burnin = 100
gap_size = 5
alpha = 0.001
valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
outlier_position = valid_positions[3]
outlier_ratio = 0.05
asymmetric_ratio = 0.25
data_1 = np.append(np.random.normal(size=n_sam_bef_cp, scale=np.sqrt(variance)), 
                       np.random.normal(size=n_sam_aft_cp,loc=gap_size, scale=np.sqrt(variance)))
outinj = OutlierInjector(data_1 ,n_sam_bef_cp, n_sam_aft_cp, burnin, variance, 
                         gap_size, variance, alpha, outlier_position, outlier_ratio, asymmetric_ratio=0.25)
out_data = outinj.insert_outliers()
outinj.plot_data(save=True, dpi=600)
outinj.outlier_indices
outinj.num_outliers
# ------------------End-------------------


# -------------------streaming data simulation--------------------

data_stream, tau_list, mu_list, size_list = simulate_stream_data(v=50, G=50, D=50, M=10, S=[0.25, 0.5, 1, 3], sigma=1, seed=666)
stream_data_plot(data_stream, tau_list)
tau_list.shape
size_list.sum()
data_stream.shape
# ------------------End-------------------


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