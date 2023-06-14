import importlib
import numpy as np
import pandas as pd
import GraphGeneration
import ARLFunc
import GridEvaluation
import Outliers
import seaborn as sns
import matplotlib.pyplot as plt

# Reload the module
importlib.reload(GraphGeneration)
importlib.reload(ARLFunc)
importlib.reload(GridEvaluation)
importlib.reload(Outliers)

# reimport functions from the reloaded module
from GraphGeneration import generate_cusum_chart, generate_ewma_chart
from ARLFunc import arl_cusum, arl_ewma
from GridEvaluation import GridDataEvaluate, simulate_stream_data, stream_data_plot
from Outliers import OutlierInjector



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



# ------------------Testing function for grid_params_eval function without outlier-------------------
# Setup initial values
n_sam_bef_cp = 400
n_sam_aft_cp = 600
gap_sizes = [1, 5, 10]
variances = [1, 4, 9]
seeds = [111, 222, 666, 999]
BURNIN = 50
cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
# simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
grideval = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, 
                            seeds, BURNIN, cusum_params_list, ewma_params_list, None)

per_table, per_summary = grideval.grid_params_eval()
per_summary
grideval.plot_ARL0_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
grideval.plot_ARL1_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
grideval.plot_best_models()
# ------------------End-------------------


# -------------------testing for outliers generation in GridEvaluation class--------------------
# Setup initial values
n_sam_bef_cp = 400
n_sam_aft_cp = 400
gap_sizes = [1, 5, 10]
variances = [1, 4, 9]
seeds = [111, 222, 666, 999]
BURNIN = 50
cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
outlier_position = valid_positions[2]
alpha = 1e-5
outlier_ratio = 0.05
# simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
grideval = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, seeds, BURNIN,
                             cusum_params_list, ewma_params_list, outlier_position, alpha, outlier_ratio)
outlier_grid_data = grideval.generate_with_outliers_grid_data(seeds[0])
outlier_grid_data[0][0].shape
per_table, per_summary = grideval.grid_params_eval()
per_table.iloc[50]
grideval.plot_ARL0_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
grideval.plot_ARL1_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
grideval.plot_best_models()
# ------------------End-------------------

# -------------------testing for outliers function--------------------
n_sam_bef_cp = 500
n_sam_aft_cp = 400
variance = 4
burnin = 50
gap_size = 5
alpha = 0.001
valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
outlier_position = valid_positions[2]
outlier_ratio = 0.05
data_1 = np.append(np.random.normal(size=n_sam_bef_cp, scale=np.sqrt(variance)), 
                       np.random.normal(size=n_sam_aft_cp,loc=gap_size, scale=np.sqrt(variance)))
outinj = OutlierInjector(data_1.copy() ,n_sam_bef_cp, n_sam_aft_cp, burnin, variance, 
                         gap_size, variance, alpha, outlier_position, outlier_ratio)
out_data = outinj.insert_outliers()
outinj.outlier_indices
outinj.plot_data()
outinj.num_outliers

# ------------------End-------------------



# -------------------streaming data simulation--------------------

data_stream, tau_list, mu_list, size_list = simulate_stream_data(v=50, G=50, D=50, M=10, S=[0.25, 0.5, 1, 3], sigma=1, seed=666)
stream_data_plot(data_stream, tau_list)
tau_list.shape
size_list.sum()
data_stream.shape
# ------------------End-------------------

