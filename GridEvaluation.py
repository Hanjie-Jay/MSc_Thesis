import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ARLFunc import arl_cusum, arl_ewma, arl_robust_mean
from Outliers import OutlierInjector
from itertools import product


class GridDataEvaluate:
    """
    A class to evaluate the performance of all models in the context of change point detection.
    The class uses grid parameters to simulate data and calculate the Average Run Length (ARL).
    The functions to plot and compare their performance are also included.
    
    Attributes:
        n_sam_bef_cp (int): Number of samples before the change point.
        n_sam_fsgdahaft_cp (int): Number of samples after the change point.
        gap_sizes (list): List of mean gap sizes for data simulation.
        variances (list): List of variances for data simulation.
        seeds (list): List of seeds for random number generation in data simulation.
        burnin (int): Number of initial samples to discard in ARL calculation.
        cusum_params_list (list): List of parameter pairs (cusum_k, cusum_h) for the CUSUM model.
        ewma_params_list (list): List of parameter pairs (ewma_rho, ewma_k) for the EWMA model.
        outlier_position (None or str): The position to insert outliers ('in-control', 'out-of-control', 'both_in_and_out', 'burn-in').
        beta (float, optional): The threshold probability of occurrence for the outliers (default is None, should between (0,1)).
        outlier_ratio (float, optional): The ratio of data points of the given outlier position period to be considered outliers (default is None, should be between (0,1)).
        asymmetric_ratio (float, optimal): The ratio for adding a asymmetric outliers above the mean (default is 0.1, should be between [0,1]).
        
    Methods:
        generate_no_outlier_grid_data():
            Simulates grid data that have no outlier for the provided parameters.
        generate_with_outliers_grid_data():
            Simulates grid data that have outliers for the provided parameters.
        grid_C_E_params_eval():
            Evaluates the performance of CUSUM and EWMA control charts with different parameters by calculating the ARLs.
        grid_robust_params_eval():
            Evaluates the performance of trimmed mean, winsorized mean, sliding window median and cosine tapered mean control chart with different parameters by calculating the ARLs.
        plot_C_E_ARL0_graphs(each_G:bool, each_G_V:bool, all_CUSUM:bool, each_CUSUM:bool, all_EWMA:bool, each_EWMA:bool):
            Plots different types of box plots to visualize ARL0 values under various conditions.
        plot_C_E_ARL1_graphs():
            Plot function for ARL1. Specific parameters to be provided based on implementation.
        plot_best_models():
            Plot function for best models of CUSUM and EWMA w.r.t. ARL0 and ARL1 values.
    """
    def __init__(self, n_sam_bef_cp:int, n_sam_aft_cp:int, gap_sizes:list, variances:list,  seeds:list, burnin:int, 
                 cusum_params_list:list, ewma_params_list:list, z_list:list, alpha_list:list, tm_params_list:list, wm_params_list:list,
                 swm_params_list: list, ctm_params_list: list, outlier_position:str, beta:float=None, 
                 outlier_ratio:float=None, asymmetric_ratio:float=0.1):
        assert isinstance(n_sam_bef_cp, int) and n_sam_bef_cp > 0, f"n_sam_bef_cp:{n_sam_bef_cp} must be a positive integer"
        assert isinstance(n_sam_aft_cp, int) and n_sam_aft_cp > 0, f"n_sam_aft_cp:{n_sam_aft_cp} must be a positive integer"
        assert isinstance(gap_sizes, (list, int, float)) and (all(isinstance(i, (int, float)) and i >= 0 for i in gap_sizes) if isinstance(gap_sizes, list) else gap_sizes >= 0), f"gap_sizes:{gap_sizes} must be a number or list of non-negetive numbers"
        assert isinstance(variances, (list, int, float)) and (all(isinstance(i, (int, float)) and i >= 0 for i in variances) if isinstance(variances, list) else variances >= 0), f"variances:{variances} must be a number or list of non-negetive numbers"
        assert isinstance(seeds, (list, int)) and (all(isinstance(i, int) for i in seeds) if isinstance(seeds, list) else seeds>=0), f"seeds:{seeds} must be a non-negative integer or list of non-negative integers"
        assert isinstance(burnin, int) and burnin >= 0, f"burnin:{burnin} must be a non-negative integer"
        assert burnin < n_sam_bef_cp, f"Value of burnin:{burnin} should smaller than n_sam_bef_cp:{n_sam_bef_cp}"
        assert isinstance(cusum_params_list, list) and all(isinstance(i, tuple) and len(i)==2 for i in cusum_params_list), f"cusum_params_list:{cusum_params_list} must be a list of tuples each of size 2"
        assert isinstance(ewma_params_list, list) and all(isinstance(i, tuple) and len(i)==2 for i in ewma_params_list), f"ewma_params_list:{ewma_params_list} must be a list of tuples each of size 2"
        assert isinstance(z_list, list) and all(isinstance(i, (float, int))  for i in z_list), f"z_list:{z_list} must be a list of floats or ints"
        assert isinstance(alpha_list, list) and all(isinstance(i, (float, int))  for i in alpha_list), f"alpha_list:{alpha_list} must be a list of floats or ints"
        assert isinstance(tm_params_list, list) and all(isinstance(i, tuple) and len(i)==2 for i in tm_params_list), f"tm_params_list:{tm_params_list} must be a list of tuples each of size 2"
        assert isinstance(wm_params_list, list) and all(isinstance(i, tuple) and len(i)==2 for i in wm_params_list), f"wm_params_list:{wm_params_list} must be a list of tuples each of size 2"
        assert isinstance(swm_params_list, list) and all(isinstance(i, int) for i in swm_params_list), f"swm_params_list:{swm_params_list} must be a list of ints each of size 1"
        assert isinstance(ctm_params_list, list) and all(isinstance(i, tuple) and len(i)==2 for i in ctm_params_list), f"ctm_params_list:{ctm_params_list} must be a list of tuples each of size 2"
        assert outlier_position is None or isinstance(outlier_position, str), f"outlier_position:{outlier_position} must be either None or a string"
        valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
        if outlier_position is not None:
            assert beta is None or (isinstance(beta, float) and 0 < beta < 1), f"beta:{beta} must be a float within the range [0,1]"
            assert outlier_ratio is None or (isinstance(outlier_ratio, float) and 0 < outlier_ratio < 1), f"outlier_ratio:{outlier_ratio} must be a float within the range (0,1)"
            assert isinstance(asymmetric_ratio, float) and 0 <= asymmetric_ratio <= 1, f"{asymmetric_ratio} should be a float between [0,1]."
            if isinstance(outlier_position, str):
                if outlier_position not in valid_positions:
                    raise ValueError(f"Invalid outlier position. Options are: {valid_positions}")
            else:
                raise TypeError("outlier_position should be only one of the valid string.")
        self.n_sam_bef_cp = n_sam_bef_cp
        self.n_sam_aft_cp = n_sam_aft_cp
        self.gap_sizes = gap_sizes
        self.variances = variances
        self.seeds = seeds
        self.burnin = burnin
        self.cusum_params_list = cusum_params_list
        self.ewma_params_list = ewma_params_list
        self.z_list = z_list
        self.alpha_list = alpha_list
        self.tm_params_list = tm_params_list
        self.wm_params_list = wm_params_list
        self.swm_params_list = swm_params_list
        self.ctm_params_list = ctm_params_list
        self.outlier_position = outlier_position
        self.beta = beta
        self.outlier_ratio = outlier_ratio
        self.asymmetric_ratio = asymmetric_ratio

    def generate_no_outlier_grid_data(self, seed:int):
        """
        Generate a grid of different types of streaming data, including data with and without change points, 
        different gap sizes, and variances. All streaming data starts with zero mean, and the variance is the same 
        for the data stream before and after the change point.

        Parameters:
        seed (int): The seed to control data generation.

        Returns:
        simulate_data_list (list): A list of tuples, each containing data and the corresponding true change point, 
                                mean gap size, and variance.
        """
        assert isinstance(seed, int), f"seed:{seed} must be a integer"
        simulate_data_list = []
        np.random.seed(seed)
        # Without change in mean but different variance
        for variance in self.variances:
            data_without_change = np.random.normal(scale=np.sqrt(variance),size=self.n_sam_bef_cp + self.n_sam_aft_cp)
            simulate_data_list.append((data_without_change, None, 0, variance))
        # With increase/decrease in mean and different variance
        for gap_size in self.gap_sizes:
            for variance in self.variances:
                # Mean increase
                data_with_increase = np.append(np.random.normal(size=self.n_sam_bef_cp, scale=np.sqrt(variance)), 
                                            np.random.normal(loc=gap_size, scale=np.sqrt(variance), size=self.n_sam_aft_cp))
                simulate_data_list.append((data_with_increase, self.n_sam_bef_cp, gap_size, variance))
                # Mean decrease
                data_with_decrease = np.append(np.random.normal(size=self.n_sam_bef_cp, scale=np.sqrt(variance)), 
                                            np.random.normal(loc=-gap_size, scale=np.sqrt(variance), size=self.n_sam_aft_cp))
                simulate_data_list.append((data_with_decrease, self.n_sam_bef_cp, - gap_size, variance))
        return simulate_data_list

    def generate_with_outliers_grid_data(self, seed:int):
        """
        Generate a grid of different types of streaming data, including data with and without change points, 
        different gap sizes, and variances. All streaming data starts with zero mean, and the variance is the same 
        for the data stream before and after the change point. Outliers are also inserted into the data stream.

        Parameters:
        seed (int): The seed to control data generation.

        Returns:
        simulate_data_list (list): A list of tuples, each containing data and the corresponding true change point, 
                                mean gap size, variance, and outlier indices.
        """
        assert isinstance(seed, int), f"seed:{seed} must be a integer"
        # Check user-provided input for outlier position
        simulate_data_list = []
        np.random.seed(seed)
        # Without change in mean
        for variance in self.variances:
            data_without_outliers = np.random.normal(scale=np.sqrt(variance),size=self.n_sam_bef_cp + self.n_sam_aft_cp)
            outinj = OutlierInjector(data_without_outliers, self.n_sam_bef_cp, self.n_sam_aft_cp, self.burnin, variance, 0, variance, 
                                    self.beta, outlier_ratio=self.outlier_ratio, outlier_position=self.outlier_position,
                                    asymmetric_ratio=self.asymmetric_ratio)
            data_with_outliers = outinj.insert_outliers()
            simulate_data_list.append((data_with_outliers, None, 0, variance, outinj.outlier_indices)) # extra list of outlier indices
        # With increase/decrease in mean and different variance
        for gap_size in self.gap_sizes:
            for variance in self.variances:
                # Mean increase
                data_with_increase = np.append(np.random.normal(size=self.n_sam_bef_cp, scale=np.sqrt(variance)), 
                                        np.random.normal(loc=gap_size, scale=np.sqrt(variance), size=self.n_sam_aft_cp))
                outinj = OutlierInjector(data_with_increase, self.n_sam_bef_cp, self.n_sam_aft_cp, self.burnin, variance, gap_size, 
                                    variance, self.beta, outlier_ratio=self.outlier_ratio, outlier_position=self.outlier_position,
                                    asymmetric_ratio=self.asymmetric_ratio)
                data_with_increase_outliers = outinj.insert_outliers()
                simulate_data_list.append((data_with_increase_outliers, self.n_sam_bef_cp, gap_size, variance, outinj.outlier_indices))
                # Mean decrease
                data_with_decrease = np.append(np.random.normal(size=self.n_sam_bef_cp, scale=np.sqrt(variance)), 
                                        np.random.normal(loc=-gap_size, scale=np.sqrt(variance), size=self.n_sam_aft_cp))
                outinj = OutlierInjector(data_with_decrease, self.n_sam_bef_cp, self.n_sam_aft_cp, self.burnin, variance, -gap_size, 
                                    variance, self.beta, outlier_ratio=self.outlier_ratio, outlier_position=self.outlier_position,
                                    asymmetric_ratio=self.asymmetric_ratio)
                data_with_decrease_outliers = outinj.insert_outliers()
                simulate_data_list.append((data_with_decrease_outliers, self.n_sam_bef_cp, -gap_size, variance, outinj.outlier_indices))
        return simulate_data_list

    def grid_C_E_params_eval(self):
        """
        Computes the mean and standard deviation of average run lengths (ARLs) for both CUSUM and EWMA models
        across a variety of data parameters, including mean gap size and variance. This is done for multiple 
        seeds, hence incorporating variability in data generation. 

        The performance summary is achieved by simulating data, calculating ARL for both CUSUM and EWMA models 
        and then grouping these results for different model parameters and data characteristics. The results 
        are stored as attributes of the class instance, and also returned by the function.

        Parameters:
        None

        Returns:
        performance_table (DataFrame): A pandas DataFrame containing detailed ARL values for each model, 
        data characteristics and seed.
        performance_summary (DataFrame): A pandas DataFrame summarizing the mean and std of ARL0 and ARL1 for each 
        model and data characteristics. The dataframe is rounded to 4 decimal places for readability.
        """
        arl_values = []
        for seed in self.seeds:
            if self.outlier_position is None:
                simulate_data_list = self.generate_no_outlier_grid_data(seed) # simulate data
                for data, true_cp, gap_size, variance in simulate_data_list:
                    for cusum_k, cusum_h in self.cusum_params_list:
                        arl0, arl1 = arl_cusum(data, self.burnin, cusum_k, cusum_h, true_cp) # arl for cusum
                        arl_values.append((f'CUSUM ({cusum_k},{cusum_h})', f'MG:{gap_size} Var:{variance}', arl0, arl1, seed))
                    for ewma_rho, ewma_k in self.ewma_params_list:
                        arl0, arl1 = arl_ewma(data, self.burnin, ewma_rho, ewma_k, true_cp) # arl for ewma
                        arl_values.append((f'EWMA ({ewma_rho},{ewma_k})', f'MG:{gap_size} Var:{variance}', arl0, arl1, seed))
            else:
                simulate_data_list = self.generate_with_outliers_grid_data(seed) # simulate data with outliers
                for data, true_cp, gap_size, variance, outlier_ind in simulate_data_list:
                    for cusum_k, cusum_h in self.cusum_params_list:
                        arl0, arl1 = arl_cusum(data, self.burnin, cusum_k, cusum_h, true_cp) # arl for cusum
                        arl_values.append((f'CUSUM ({cusum_k},{cusum_h})', f'MG:{gap_size} Var:{variance}', arl0, arl1, seed, outlier_ind))
                    for ewma_rho, ewma_k in self.ewma_params_list:
                        arl0, arl1 = arl_ewma(data, self.burnin, ewma_rho, ewma_k, true_cp) # arl for ewma
                        arl_values.append((f'EWMA ({ewma_rho},{ewma_k})', f'MG:{gap_size} Var:{variance}', arl0, arl1, seed, outlier_ind))

        # transform the data into pandas dataframe and compute the mean and variance using groupby
        if self.outlier_position is None:
            performance_table = pd.DataFrame(arl_values, columns=['Model (Parameters)', f'Data (len:{self.n_sam_aft_cp+self.n_sam_bef_cp}, TCP:{self.n_sam_bef_cp})', 'ARL0', 'ARL1', 'seed'])
        else:
            performance_table = pd.DataFrame(arl_values, columns=['Model (Parameters)', f'Data (len:{self.n_sam_aft_cp+self.n_sam_bef_cp}, TCP:{self.n_sam_bef_cp})', 'ARL0', 'ARL1', 'seed', 'outlier_ind'])
        performance_summary = performance_table.groupby([f'Data (len:{self.n_sam_aft_cp+self.n_sam_bef_cp}, TCP:{self.n_sam_bef_cp})',
                                                        'Model (Parameters)']).agg({'ARL0':['mean', 'std'], 'ARL1':['mean', 'std']}).reset_index()
        # extract gap size and variance from 'Data' column
        performance_table[['Gap Size', 'Data Var']] = performance_table[f'Data (len:{self.n_sam_aft_cp+self.n_sam_bef_cp}, TCP:{self.n_sam_bef_cp})'].str.extract('MG:(-?\d+) Var:(\d+)')
        # Convert 'Gap Size' and 'Variance' to numeric
        performance_table['Gap Size'] = pd.to_numeric(performance_table['Gap Size'])
        performance_table['Data Var'] = pd.to_numeric(performance_table['Data Var'])
        # Take absolute value of gap size
        performance_table['Gap Size'] = performance_table['Gap Size'].abs()
        self.C_E_performance_table = performance_table
        self.C_E_performance_summary = performance_summary
        return performance_table, performance_summary.round(4) # 4 decimal places
    
    def grid_robust_params_eval(self):
        """
        The method computes the average run lengths (ARLs) for various robust models across different data parameters 
        including mean gap size and variance, for multiple seeds to include variability in data generation. 

        The process starts by creating the Cartesian product of the z and alpha values, to iterate over all possible
        combinations. All parameters lists are then made of equal length for ease of iteration, and the data 
        are generated with or without outliers as per the user's specification.

        For each combination of data, z and alpha values, and control parameters, the ARLs are computed for each model and 
        stored in a list.

        A Pandas DataFrame is created from the list of ARLs, including the model parameters, z and alpha values,
        data characteristics, ARL0, ARL1, and seed.

        The DataFrame is then grouped by data characteristics, z and alpha values, and model parameters, and 
        the mean and standard deviation of ARL0 and ARL1 are computed.

        Gap size and variance are extracted from the 'Data' column of the DataFrame and converted to numeric format.

        Parameters:
        None

        Returns:
        performance_table (DataFrame): A DataFrame containing the ARL values for each model, data characteristics,
        z and alpha values, and seed. Also includes an 'outlier_ind' column if data are generated with outliers.
        
        performance_summary (DataFrame): A DataFrame summarizing the mean and standard deviation of ARL0 and ARL1
        for each combination of data characteristics, z and alpha values, and model parameters. The DataFrame is
        rounded to four decimal places for readability.
        """
        arl_values = []
        z_alpha_list = list(product(self.z_list, self.alpha_list))
        max_len = max(len(self.tm_params_list), len(self.wm_params_list), len(self.swm_params_list), len(self.ctm_params_list))
        # Extend the parameters lists with None to make them of equal length
        tm_params_list = self.tm_params_list + [None]*(max_len-len(self.tm_params_list))
        wm_params_list = self.wm_params_list + [None]*(max_len-len(self.wm_params_list))
        swm_params_list = self.swm_params_list + [None]*(max_len-len(self.swm_params_list))
        ctm_params_list = self.ctm_params_list + [None]*(max_len-len(self.ctm_params_list))
        for seed in self.seeds:
            if self.outlier_position is None:
                simulate_data_list = self.generate_no_outlier_grid_data(seed) # simulate data
                for data, true_cp, gap_size, variance in simulate_data_list:
                    for z, alpha in z_alpha_list:
                        for i in range(max_len):
                            tm_params = tm_params_list[i]
                            wm_params = wm_params_list[i]
                            swm_params = swm_params_list[i]
                            ctm_params = ctm_params_list[i]
                            method_params = {
                                'T': tm_params,
                                'W': wm_params,
                                'M': swm_params,
                                'CT': ctm_params
                            }
                            results = arl_robust_mean(
                                data, self.burnin, swm_params, 
                                tm_params[0] if tm_params is not None else None, wm_params[0] if wm_params is not None else None, 
                                ctm_params[0] if ctm_params is not None else None, tm_params[1] if tm_params is not None else None, 
                                wm_params[1] if wm_params is not None else None, ctm_params[1] if ctm_params is not None else None, 
                                z, alpha, true_cp)
                            for method, method_arl in results.items():
                                arl0 = method_arl['arl0']
                                arl1 = method_arl['arl1']
                                params = method_params[method]
                                arl_values.append((f'{method} (z:{z},a:{alpha},{params})', f'z:{z} alpha:{alpha}', f'MG:{gap_size} Var:{variance}', arl0, arl1, seed))
            else:
                simulate_data_list = self.generate_with_outliers_grid_data(seed) # simulate data with outliers
                for data, true_cp, gap_size, variance, outlier_ind in simulate_data_list:
                    for z, alpha in z_alpha_list:
                        for i in range(max_len):
                            tm_params = tm_params_list[i]
                            wm_params = wm_params_list[i]
                            swm_params = swm_params_list[i]
                            ctm_params = ctm_params_list[i]
                            method_params = {
                                'T': tm_params,
                                'W': wm_params,
                                'M': swm_params,
                                'CT': ctm_params
                            }
                            results = arl_robust_mean(
                                data, self.burnin, swm_params, 
                                tm_params[0] if tm_params is not None else None, wm_params[0] if wm_params is not None else None, 
                                ctm_params[0] if ctm_params is not None else None, tm_params[1] if tm_params is not None else None, 
                                wm_params[1] if wm_params is not None else None, ctm_params[1] if ctm_params is not None else None, 
                                z, alpha, true_cp)
                            for method, method_arl in results.items():
                                arl0 = method_arl['arl0']
                                arl1 = method_arl['arl1']
                                params = method_params[method]
                                arl_values.append((f'{method} (z:{z},a:{alpha},{params})', f'z:{z} alpha:{alpha}', f'MG:{gap_size} Var:{variance}', arl0, arl1, seed, outlier_ind))

        # transform the data into pandas dataframe and compute the mean and variance using groupby
        if self.outlier_position is None:
            performance_table = pd.DataFrame(arl_values, columns=['Model (Parameters)', 'z and alpha',
                                                                  f'Data (len:{self.n_sam_aft_cp+self.n_sam_bef_cp}, TCP:{self.n_sam_bef_cp})', 
                                                                  'ARL0', 'ARL1', 'seed'])
        else:
            performance_table = pd.DataFrame(arl_values, columns=['Model (Parameters)', 'z and alpha',
                                                                  f'Data (len:{self.n_sam_aft_cp+self.n_sam_bef_cp}, TCP:{self.n_sam_bef_cp})', 
                                                                  'ARL0', 'ARL1', 'seed', 'outlier_ind'])
        performance_summary = performance_table.groupby([f'Data (len:{self.n_sam_aft_cp+self.n_sam_bef_cp}, TCP:{self.n_sam_bef_cp})', 'z and alpha',
                                                        'Model (Parameters)']).agg({'ARL0':['mean', 'std'], 'ARL1':['mean', 'std']}).reset_index()
        # extract gap size and variance from 'Data' column
        performance_table[['Gap Size', 'Data Var']] = performance_table[f'Data (len:{self.n_sam_aft_cp+self.n_sam_bef_cp}, TCP:{self.n_sam_bef_cp})'].str.extract('MG:(-?\d+) Var:(\d+)')
        # Convert 'Gap Size' and 'Variance' to numeric
        performance_table['Gap Size'] = pd.to_numeric(performance_table['Gap Size'])
        performance_table['Data Var'] = pd.to_numeric(performance_table['Data Var'])
        # Take absolute value of gap size
        performance_table['Gap Size'] = performance_table['Gap Size'].abs()
        # Transform the model index in proper form
        def transform_robust_index(index):
            model = str(index).split(' ')[0]
            if model != 'M':
                z = re.search(r'z:(\d+\.\d+)',index).group(1)
                alpha = re.search(r'a:(\d+(?:\.\d+)?)', index).group(1)
                p = re.search(r'\((\d+\.+\d+), ', index).group(1)
                lsw = re.search(r', (\d+)\)', index).group(1)
                model_name = f"{model} ({z}, {alpha}, {p}, {lsw})"
            else:
                z = re.search(r'z:(\d+\.\d+)',index).group(1)
                alpha = re.search(r'a:(\d+(?:\.\d+)?)', index).group(1)
                lsw = re.search(r',(\d+)\)', index).group(1)
                model_name = f"{model} ({z}, {alpha}, {lsw})"
            return model_name
        performance_table['Model (Parameters)'] = performance_table['Model (Parameters)'].apply(transform_robust_index)
        self.robust_performance_table = performance_table
        self.robust_performance_summary = performance_summary
        return performance_table, performance_summary.round(4) # 4 decimal places

    def plot_C_E_ARL0_graphs(self, save:bool=True, each_G:bool=True, each_G_V:bool=True, all_CUSUM:bool=True, each_CUSUM:bool=True, all_EWMA:bool=True, each_EWMA:bool=True, dpi:int=500):
        """
        This function creates different types of box plots to visualize ARL0 values for different conditions.
        
        Parameters:
        save (bool, optional): If True, the function will save each plot to a sub-folder in the 'ARL_0_graphs' directory.
        each_G (bool, optional): If True, the function will create a boxplot for each unique gap size.
        each_G_V (bool, optional): If True, the function will create a boxplot for each unique gap size and variance.
        all_CUSUM (bool, optional): If True, the function will create a boxplot for the CUSUM model.
        each_CUSUM (bool, optional): If True, the function will create a boxplot for each unique model parameter for CUSUM.
        all_EWMA (bool, optional): If True, the function will create a boxplot for the EWMA model.
        each_EWMA (bool, optional): If True, the function will create a boxplot for each unique model parameter for EWMA.
        dpi (int, optional): The resolution in dots per inch for saved figures (default to be 500).

        Returns:
        None: The function generates plots
        """
        if not hasattr(self, 'C_E_performance_table'):
            self.grid_C_E_params_eval()
        per_table = self.C_E_performance_table.copy()
        # Assertions to validate input data types
        assert isinstance(save, bool), f"The save:{save} parameter must be a boolean."
        assert isinstance(per_table, pd.DataFrame), "per_table must be a pandas DataFrame."
        assert isinstance(each_G, bool), f"each_G:{each_G} must be a boolean value."
        assert isinstance(each_G_V, bool), f"each_G_V:{each_G_V} must be a boolean value."
        assert isinstance(all_CUSUM, bool), f"all_CUSUM:{all_CUSUM} must be a boolean value."
        assert isinstance(each_CUSUM, bool), f"each_CUSUM:{each_CUSUM} must be a boolean value."
        assert isinstance(all_EWMA, bool), f"all_EWMA:{all_EWMA} must be a boolean value."
        assert isinstance(each_EWMA, bool), f"each_EWMA:{each_EWMA} must be a boolean value."
        assert isinstance(dpi, int) and dpi > 0, f"The dpi:{dpi} parameter must be a positive integer."
        # Separate CUSUM and EWMA rows
        cusum_table = per_table[per_table['Model (Parameters)'].str.contains('CUSUM')] # Select col that have CUSUM
        ewma_table = per_table[per_table['Model (Parameters)'].str.contains('EWMA')] # Select col that have EWMA
        # unique model parameters for CUSUM and EWMA
        cusum_params = cusum_table['Model (Parameters)'].unique()
        ewma_params = ewma_table['Model (Parameters)'].unique()
        # Set style and palette
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.color_palette("crest", as_cmap=True)
        n_colors = len(cusum_params) + len(ewma_params) # the total number of unique parameters
        colors = sns.color_palette("crest", n_colors=n_colors)
        colors_each = sns.color_palette("crest", n_colors=len(self.variances))

        if save:
            base_dir = os.path.join("Plots", 'ARL_0_graphs')
            os.makedirs(base_dir, exist_ok=True) # make the directory to save arl0 graphs

        # Create a boxplot for each gap size
        if each_G == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_gap_size'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for gap_size in per_table['Gap Size'].unique():
                subset_df = per_table[per_table['Gap Size'] == gap_size]
                plt.figure(figsize=(20, 8))
                ax =sns.boxplot(x='Model (Parameters)', y='ARL0', data=subset_df, palette=colors)
                ax.tick_params(labelsize=10)
                if self.outlier_position is None:
                    plt.title(f'Boxplot of $ARL_0$ for Various Models on Streaming Data with a Mean Gap Size of {gap_size} (No Outliers)', fontsize=18)
                else:
                    plt.title(f'Boxplot of $ARL_0$ for Various Models on Streaming Data with a Mean Gap Size of {gap_size} (Outliers in {self.outlier_position} Period)', fontsize=18)
                plt.ylabel('$ARL_0$', fontsize=13)
                plt.xlabel('Model (Parameters)', fontsize=14)
                plt.xticks(rotation=30)
                if save:
                    # Define the filename based on the specific graph parameters.
                    filename = f"arl0_gap_{gap_size}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                    plt.tight_layout()
                    plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                plt.show()

        # Create a boxplot for each gap size and variance size
        if each_G_V == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_gap_&_var'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for gap_size in per_table['Gap Size'].unique():
                for vari in per_table['Data Var'].unique():
                    subset_df = per_table[(per_table['Gap Size'] == gap_size) & (per_table['Data Var'] == vari)]
                    plt.figure(figsize=(20, 8))
                    ax =sns.boxplot(x='Model (Parameters)', y='ARL0', data=subset_df, palette=colors)
                    ax.tick_params(labelsize=10)
                    if self.outlier_position is None:
                        plt.title(f'Boxplot of $ARL_0$ for Various Models on Streaming Data with a Mean Gap Size of {gap_size} and Data Variance of {vari} (No Outliers)', fontsize=18)
                    else:
                        plt.title(f'Boxplot of $ARL_0$ for Various Models on Streaming Data with a Mean Gap Size of {gap_size}, Data Variance of {vari} (Outliers in {self.outlier_position} Period)', fontsize=18)
                    plt.ylabel('$ARL_0$', fontsize=13)
                    plt.xlabel('Model (Parameters)', fontsize=14)
                    plt.xticks(rotation=30)
                    if save:
                        # Define the filename based on the specific graph parameters.
                        filename = f"arl0_gap_{gap_size}_var{vari}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                        plt.tight_layout()
                        plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                    plt.show()

        # Create box plots for CUSUM model
        if all_CUSUM == True:
            plt.figure(figsize=(14, 8))
            ax =sns.boxplot(data=cusum_table, x='Gap Size', y='ARL0', hue='Data Var', palette=colors_each)
            ax.legend(fontsize=14)
            ax.tick_params(labelsize=14)
            if self.outlier_position is None:
                plt.title('$ARL_0$ Values of All CUSUM Models in Streaming Data Without Outliers', fontsize=18)
            else:
                plt.title(f'$ARL_0$ Values of All CUSUM Models in Streaming Data with Outliers in {self.outlier_position} Period', fontsize=18)
            plt.ylabel('$ARL_0$', fontsize=14)
            plt.xlabel('Gap Size', fontsize=14)
            if save:
                filename = f"arl0_all_cusum_models_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                plt.tight_layout()
                plt.savefig(os.path.join(base_dir, filename), dpi=dpi, format='pdf')
            plt.show()

        # Create box plots for EWMA model
        if all_EWMA == True:
            plt.figure(figsize=(14, 8))
            ax =sns.boxplot(data=ewma_table, x='Gap Size', y='ARL0', hue='Data Var', palette=colors_each)
            ax.legend(fontsize=14)
            ax.tick_params(labelsize=14)
            if self.outlier_position is None:
                plt.title('$ARL_0$ Values of All EWMA Models in Streaming Data Without Outliers', fontsize=18)
            else:
                plt.title(f'$ARL_0$ Values of All EWMA Models in Streaming Data with Outliers in {self.outlier_position} Period', fontsize=18)
            plt.ylabel('$ARL_0$', fontsize=14)
            plt.xlabel('Gap Size', fontsize=14)
            if save:
                filename = f"arl0_all_ewma_models_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                plt.tight_layout()
                plt.savefig(os.path.join(base_dir, filename), dpi=dpi, format='pdf')
            plt.show()

        # Loop through each unique model parameters for CUSUM and create box plot
        if each_CUSUM == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_cusum_model'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for param in cusum_params:
                plt.figure(figsize=(14, 8))
                ax =sns.boxplot(data=cusum_table[cusum_table['Model (Parameters)'] == param], x='Gap Size', y='ARL0', hue='Data Var', palette=colors_each)
                ax.legend(fontsize=14)
                ax.tick_params(labelsize=14)
                if self.outlier_position is None:
                    plt.title(f'The Values of $ARL_0$ for Model {param} under Different Streaming Data Settings, Without Outliers', fontsize=16)
                else:
                    plt.title(f'The Values of $ARL_0$ for Model {param} under Different Streaming Data Settings with Outliers in {self.outlier_position} period', fontsize=16)
                plt.ylabel('$ARL_0$', fontsize=14)
                plt.xlabel('Gap Size', fontsize=14)
                if save:
                    # Define the filename based on the specific graph parameters.
                    filename = f"arl0_{param}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                    plt.tight_layout()
                    plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                plt.show()

        # Loop through each unique model parameters for EWMA and create box plot
        if each_EWMA == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_ewma_model'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for param in ewma_params:
                plt.figure(figsize=(14, 8))
                ax =sns.boxplot(data=ewma_table[ewma_table['Model (Parameters)'] == param], x='Gap Size', y='ARL0', hue='Data Var', palette=colors_each)
                ax.legend(fontsize=14)
                ax.tick_params(labelsize=14)
                if self.outlier_position is None:
                    plt.title(f'The Values of $ARL_0$ for Model {param} under Different Streaming Data Settings, Without Outliers', fontsize=16)
                else:
                    plt.title(f'The Values of $ARL_0$ for Model {param} under Different Streaming Data Settings with Outliers in {self.outlier_position} period', fontsize=16)
                plt.ylabel('$ARL_0$', fontsize=14)
                plt.xlabel('Gap Size', fontsize=14)
                if save:
                    # Define the filename based on the specific graph parameters.
                    filename = f"arl0_{param}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                    plt.tight_layout()
                    plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                plt.show()
    
    def plot_robust_ARL0_graphs(self, save:bool=True, each_G:bool=True, each_G_V:bool=True, all_Methods:bool=True, each_Method:bool=True, dpi:int=500):
        if not hasattr(self, 'robust_performance_table'):
            self.grid_robust_params_eval()
        per_table = self.robust_performance_table.copy()
        # Assertions to validate input data types
        assert isinstance(save, bool), f"The save:{save} parameter must be a boolean."
        assert isinstance(per_table, pd.DataFrame), "per_table must be a pandas DataFrame."
        assert isinstance(each_G, bool), f"each_G:{each_G} must be a boolean value."
        assert isinstance(each_G_V, bool), f"each_G_V:{each_G_V} must be a boolean value."
        assert isinstance(all_Methods, bool), f"all_Methods:{all_Methods} must be a boolean value."
        assert isinstance(each_Method, bool), f"each_Method:{each_Method} must be a boolean value."
        assert isinstance(dpi, int) and dpi > 0, f"The dpi:{dpi} parameter must be a positive integer."
        
        # Separate unique methods rows
        per_table[['Model', 'Parameters']] = per_table['Model (Parameters)'].str.split(" ", n = 1, expand = True)

        # Get the tables for each method
        tm_table = per_table[per_table['Model'] == 'T'] # Select rows that have T
        wm_table = per_table[per_table['Model'] == 'W'] # Select rows that have W
        swm_table = per_table[per_table['Model'] == 'M'] # Select rows that have SW
        ctm_table = per_table[per_table['Model'] == 'CT'] # Select rows that have CT

        # Set style and palette
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.color_palette("crest", as_cmap=True)
        n_colors = len(self.tm_params_list) + len(self.wm_params_list) + len(self.swm_params_list) + len(self.ctm_params_list) # the total number of unique parameters
        colors = sns.color_palette("crest", n_colors=n_colors)
        colors_each = sns.color_palette("crest", n_colors=len(self.variances))

        if save:
            base_dir = os.path.join("Plots", 'Robust_Params_graphs', 'ARL_0_graphs')
            os.makedirs(base_dir, exist_ok=True) # make the directory to save graphs

        # Create boxplots for each gap size and z & alpha value
        if each_G == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_gap_za'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for za in per_table['z and alpha'].unique():
                subset_za = per_table[per_table['z and alpha'] == za]
                for gap_size in subset_za['Gap Size'].unique():
                    subset_df = subset_za[subset_za['Gap Size'] == gap_size]
                    unique_models = subset_df['Model (Parameters)'].unique()
                    color_mapping = {model: color for model, color in zip(unique_models, colors)}
                    plt.figure(figsize=(20, 8))
                    ax = sns.boxplot(x='Model (Parameters)', y='ARL0', data=subset_df, palette=color_mapping)
                    ax.tick_params(labelsize=10)
                    if self.outlier_position is None:
                        plt.title(f'Boxplot of $ARL_0$ for Various Models on Streaming Data with z and alpha: {za}, Mean Gap Size of {gap_size} (No Outliers)', fontsize=18)
                    else:
                        plt.title(f'Boxplot of $ARL_0$ for Various Models on Streaming Data with z and alpha: {za}, Mean Gap Size of {gap_size} (Outliers in {self.outlier_position} Period)', fontsize=18)
                    plt.ylabel('$ARL_0$', fontsize=13)
                    plt.xlabel('Model (Parameters)', fontsize=14)
                    plt.xticks(rotation=30)
                    if save:
                        # Define the filename based on the specific graph parameters.
                        filename = f"arl0_gap_{gap_size}_za_{za}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                        plt.tight_layout()
                        plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                    plt.show()

        # Create boxplots for each gap size, variance size and z & alpha value
        if each_G_V == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_gap_&_var_za'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for za in per_table['z and alpha'].unique():
                subset_za = per_table[per_table['z and alpha'] == za]
                for gap_size in subset_za['Gap Size'].unique():
                    for vari in subset_za['Data Var'].unique():
                        subset_df = subset_za[(subset_za['Gap Size'] == gap_size) & (subset_za['Data Var'] == vari)]
                        unique_models = subset_df['Model (Parameters)'].unique()
                        color_mapping = {model: color for model, color in zip(unique_models, colors)}
                        plt.figure(figsize=(20, 8))
                        ax =sns.boxplot(x='Model (Parameters)', y='ARL0', data=subset_df, palette=color_mapping)
                        ax.tick_params(labelsize=10)
                        if self.outlier_position is None:
                            plt.title(f'Boxplot of $ARL_0$ for Various Models on Streaming Data with z and alpha: {za}, Mean Gap Size of {gap_size} and Data Variance of {vari} (No Outliers)', fontsize=18)
                        else:
                            plt.title(f'Boxplot of $ARL_0$ for Various Models on Streaming Data with z and alpha: {za}, Mean Gap Size of {gap_size}, Data Variance of {vari} (Outliers in {self.outlier_position} Period)', fontsize=18)
                        plt.ylabel('$ARL_0$', fontsize=13)
                        plt.xlabel('Model (Parameters)', fontsize=14)
                        plt.xticks(rotation=30)
                        if save:
                            # Define the filename based on the specific graph parameters.
                            filename = f"arl0_gap_{gap_size}_var{vari}_za_{za}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                            plt.tight_layout()
                            plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                        plt.show()
        
        # Create boxplots for all four models describing their general performances in each data mean gap and variance setting
        if all_Methods == True:
            models = {
                'T': tm_table,
                'W': wm_table,
                'M': swm_table,
                'CT': ctm_table
            }
            # Iterate over each model
            for model_name, model_table in models.items():
                # Create a boxplot
                plt.figure(figsize=(14, 8))
                ax = sns.boxplot(data=model_table, x='Gap Size', y='ARL0', hue='Data Var', palette=colors_each)
                ax.legend(fontsize=14)
                ax.tick_params(labelsize=14)
                # Titles and labels
                if self.outlier_position is None:
                    plt.title(f'$ARL_0$ Values of {model_name} Model in Streaming Data Without Outliers', fontsize=18)
                else:
                    plt.title(f'$ARL_0$ Values of {model_name} Model in Streaming Data with Outliers in {self.outlier_position} Period', fontsize=18)
                plt.ylabel('$ARL_0$', fontsize=14)
                plt.xlabel('Gap Size', fontsize=14)
                # Save the plot
                if save:
                    filename = f"arl0_{model_name}_model_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                    plt.tight_layout()
                    plt.savefig(os.path.join(base_dir, filename), dpi=dpi, format='pdf')
                plt.show()

        # Create boxplots for each parameter setting in all four models describing their general performances in each data mean gap and variance setting
        if each_Method == True:
            models = {
                'T': tm_table,
                'W': wm_table,
                'M': swm_table,
                'CT': ctm_table
            }
            for model_name, model_table in models.items():
                # Extract unique parameters for the current model
                model_params = model_table['Model (Parameters)'].unique()
                # Create directory for this model
                model_dir = os.path.join(base_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                # Iterate over each unique parameter and create a boxplot 
                for params in model_params:
                    z_alpha_values = model_table.loc[model_table['Model (Parameters)']==params, 'z and alpha'].values[0] # receive z and alpha value
                    control_parameters = params.split('(')[-1].split(')')[0] # split for the last ( and first ) to receive control parameter
                    plt.figure(figsize=(14, 8))
                    ax = sns.boxplot(data=model_table[model_table['Model (Parameters)']==params], x='Gap Size', y='ARL0', hue='Data Var', palette=colors_each)
                    ax.legend(fontsize=14)
                    ax.tick_params(labelsize=14)
                    if self.outlier_position is None:
                        plt.title(f'The Values of $ARL_0$ for Model {model_name} ({z_alpha_values}, Control Parameters: {control_parameters}) under Different Streaming Data Settings, Without Outliers', fontsize=16)
                    else:
                        plt.title(f'The Values of $ARL_0$ for Model {model_name} ({z_alpha_values}, Control Parameters: {control_parameters}) under Different Streaming Data Settings with Outliers in {self.outlier_position} period', fontsize=16)
                    plt.ylabel('$ARL_0$', fontsize=14)
                    plt.xlabel('Gap Size', fontsize=14)
                    # Save the plot
                    if save:
                        filename = f"arl0_{params}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                        plt.tight_layout()
                        plt.savefig(os.path.join(model_dir, filename), dpi=dpi, format='pdf')
                    plt.show()
        

    def plot_C_E_ARL1_graphs(self, save:bool=True, each_G:bool=True, each_G_V:bool=True, all_CUSUM:bool=True, each_CUSUM:bool=True, all_EWMA:bool=True, each_EWMA:bool=True, dpi:int=500):
        """
        This function creates different types of box plots to visualize ARL1 values for different conditions.
        
        Parameters:
        save (bool, optional): If True, the function will save each plot to a sub-folder in the 'ARL_0_graphs' directory.
        each_G (bool, optional): If True, the function will create a boxplot for each unique gap size.
        each_G_V (bool, optional): If True, the function will create a boxplot for each unique gap size and variance.
        all_CUSUM (bool, optional): If True, the function will create a boxplot for the CUSUM model.
        each_CUSUM (bool, optional): If True, the function will create a boxplot for each unique model parameter for CUSUM.
        all_EWMA (bool, optional): If True, the function will create a boxplot for the EWMA model.
        each_EWMA (bool, optional): If True, the function will create a boxplot for each unique model parameter for EWMA.
        dpi (int, optional): The resolution in dots per inch for saved figures (default to be 500).

        Returns:
        None: The function generates plots
        """
        if not hasattr(self, 'C_E_performance_table'):
                self.grid_C_E_params_eval()
        per_table = self.C_E_performance_table.copy()
        # Assertions to validate input data types
        assert isinstance(save, bool), "The save parameter must be a boolean."
        assert isinstance(per_table, pd.DataFrame), "per_table must be a pandas DataFrame."
        assert isinstance(each_G, bool), f"each_G:{each_G} must be a boolean value."
        assert isinstance(each_G_V, bool), f"each_G_V:{each_G_V} must be a boolean value."
        assert isinstance(all_CUSUM, bool), f"all_CUSUM:{all_CUSUM} must be a boolean value."
        assert isinstance(each_CUSUM, bool), f"each_CUSUM:{each_CUSUM} must be a boolean value."
        assert isinstance(all_EWMA, bool), f"all_EWMA:{all_EWMA} must be a boolean value."
        assert isinstance(each_EWMA, bool), f"each_EWMA:{each_EWMA} must be a boolean value."
        assert isinstance(dpi, int) and dpi > 0, f"The dpi:{dpi} parameter must be a positive integer."
        # Remove the no change point data as it is meaningless
        per_table = per_table[per_table['Gap Size'] != 0]
        # Separate CUSUM and EWMA rows
        cusum_table = per_table[per_table['Model (Parameters)'].str.contains('CUSUM')] # Select col that have CUSUM
        ewma_table = per_table[per_table['Model (Parameters)'].str.contains('EWMA')] # Select col that have EWMA
        # Unique model parameters for CUSUM and EWMA
        cusum_params = cusum_table['Model (Parameters)'].unique()
        ewma_params = ewma_table['Model (Parameters)'].unique()
        # Set style and palette
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.color_palette("crest", as_cmap=True)
        n_colors = len(cusum_params) + len(ewma_params) # the total number of unique parameters
        colors = sns.color_palette("crest", n_colors=n_colors)
        colors_each = sns.color_palette("crest", n_colors=len(self.variances))

        if save:
            base_dir = os.path.join("Plots", 'ARL_1_graphs')
            os.makedirs(base_dir, exist_ok=True) # make the directory to save arl1 graphs

        # Create a boxplot for each gap size
        if each_G == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_gap_size'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for gap_size in per_table['Gap Size'].unique():
                subset_df = per_table[per_table['Gap Size'] == gap_size]
                plt.figure(figsize=(20, 8))
                ax =sns.boxplot(x='Model (Parameters)', y='ARL1', data=subset_df, palette=colors)
                ax.tick_params(labelsize=10)
                if self.outlier_position is None:
                    plt.title(f'Boxplot of $ARL_1$ for Various Models on Streaming Data with a Mean Gap Size of {gap_size} (No Outliers)', fontsize=18)
                else:
                    plt.title(f'Boxplot of $ARL_1$ for Various Models on Streaming Data with a Mean Gap Size of {gap_size} (Outliers in {self.outlier_position} Period)', fontsize=18)
                plt.ylabel('$ARL_1$', fontsize=14)
                plt.xlabel('Model (Parameters)', fontsize=14)
                plt.xticks(rotation=30)
                if save:
                    # Define the filename based on the specific graph parameters.
                    filename = f"arl1_gap_{gap_size}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                    plt.tight_layout()
                    plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                plt.show()

        # Create a boxplot for each gap size and variance size
        if each_G_V == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_gap_&_var'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for gap_size in per_table['Gap Size'].unique():
                for vari in per_table['Data Var'].unique():
                    subset_df = per_table[(per_table['Gap Size'] == gap_size) & (per_table['Data Var'] == vari)]
                    plt.figure(figsize=(20, 8))
                    ax =sns.boxplot(x='Model (Parameters)', y='ARL1', data=subset_df, palette=colors)
                    ax.tick_params(labelsize=10)
                    if self.outlier_position is None:
                        plt.title(f'Boxplot of $ARL_1$ for Various Models on Streaming Data with a Mean Gap Size of {gap_size} and Data Variance of {vari} (No Outliers)', fontsize=18)
                    else:
                        plt.title(f'Boxplot of $ARL_1$ for Various Models on Streaming Data with a Mean Gap Size of {gap_size}, Data Variance of {vari} (Outliers in {self.outlier_position} Period)', fontsize=18)
                    plt.ylabel('$ARL_1$', fontsize=14)
                    plt.xlabel('Model (Parameters)', fontsize=14)
                    plt.xticks(rotation=30)
                    if save:
                        # Define the filename based on the specific graph parameters.
                        filename = f"arl1_gap_{gap_size}_var{vari}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                        plt.tight_layout()
                        plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                    plt.show()

        # Create box plots for CUSUM model
        if all_CUSUM == True:
            plt.figure(figsize=(14, 8))
            ax =sns.boxplot(data=cusum_table, x='Gap Size', y='ARL1', hue='Data Var', palette=colors_each)
            ax.legend(fontsize=14)
            ax.tick_params(labelsize=14)
            if self.outlier_position is None:
                plt.title(f'$ARL_1$ Values of All CUSUM Models in Streaming Data Without Outliers', fontsize=18)
            else:
                plt.title(f'$ARL_1$ Values of All CUSUM Models in Streaming Data with Outliers in {self.outlier_position} Period', fontsize=18)
            plt.ylabel('$ARL_1$', fontsize=14)
            plt.xlabel('Gap Size', fontsize=14)
            if save:
                filename = f"arl1_all_cusum_models_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                plt.tight_layout()
                plt.savefig(os.path.join(base_dir, filename), dpi=dpi, format='pdf')
            plt.show()

        # Create box plots for EWMA model
        if all_EWMA == True:
            plt.figure(figsize=(14, 8))
            ax =sns.boxplot(data=ewma_table, x='Gap Size', y='ARL1', hue='Data Var', palette=colors_each)
            ax.legend(fontsize=14)
            ax.tick_params(labelsize=14)
            if self.outlier_position is None:
                plt.title(f'$ARL_1$ Values of All EWMA Models in Streaming Data Without Outliers', fontsize=18)
            else:
                plt.title(f'$ARL_1$ Values of All EWMA Models in Streaming Data with Outliers in {self.outlier_position} Period', fontsize=18)
            plt.ylabel('$ARL_1$', fontsize=14)
            plt.xlabel('Gap Size', fontsize=14)
            if save:
                filename = f"arl1_all_ewma_models_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                plt.tight_layout()
                plt.savefig(os.path.join(base_dir, filename), dpi=dpi, format='pdf')
            plt.show()

        # Loop through each unique model parameters for CUSUM and create box plot
        if each_CUSUM == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_cusum_model'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for param in cusum_params:
                plt.figure(figsize=(14, 8))
                ax =sns.boxplot(data=cusum_table[cusum_table['Model (Parameters)'] == param], x='Gap Size', y='ARL1', hue='Data Var', palette=colors_each)
                ax.legend(fontsize=14)
                ax.tick_params(labelsize=14)
                if self.outlier_position is None:
                    plt.title(f'The Values of $ARL_1$ for Model {param} under Different Streaming Data Settings, Without Outliers', fontsize=16)
                else:
                    plt.title(f'The Values of $ARL_1$ for Model {param} under Different Streaming Data Settings with Outliers in {self.outlier_position} period', fontsize=16)
                plt.ylabel('$ARL_1$', fontsize=14)
                plt.xlabel('Gap Size', fontsize=14)
                if save:
                    # Define the filename based on the specific graph parameters.
                    filename = f"arl1_{param}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                    plt.tight_layout()
                    plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                plt.show()

        # Loop through each unique model parameters for EWMA and create box plot
        if each_EWMA == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_ewma_model'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for param in ewma_params:
                plt.figure(figsize=(14, 8))
                ax =sns.boxplot(data=ewma_table[ewma_table['Model (Parameters)'] == param], x='Gap Size', y='ARL1', hue='Data Var', palette=colors_each)
                ax.legend(fontsize=14)
                ax.tick_params(labelsize=14)
                if self.outlier_position is None:
                    plt.title(f'The Values of $ARL_1$ for Model {param} under Different Streaming Data Settings, Without Outliers', fontsize=16)
                else:
                    plt.title(f'The Values of $ARL_1$ for Model {param} under Different Streaming Data Settings with Outliers in {self.outlier_position} period', fontsize=16)
                plt.ylabel('$ARL_1$', fontsize=14)
                plt.xlabel('Gap Size', fontsize=14)
                if save:
                    filename = f"arl1_{param}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                    plt.tight_layout()
                    plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                plt.show()

    def plot_robust_ARL1_graphs(self, save:bool=True, each_G:bool=True, each_G_V:bool=True, all_Methods:bool=True, each_Method:bool=True, dpi:int=500):
        if not hasattr(self, 'robust_performance_table'):
            self.grid_robust_params_eval()
        per_table = self.robust_performance_table.copy()
        # Assertions to validate input data types
        assert isinstance(save, bool), f"The save:{save} parameter must be a boolean."
        assert isinstance(per_table, pd.DataFrame), "per_table must be a pandas DataFrame."
        assert isinstance(each_G, bool), f"each_G:{each_G} must be a boolean value."
        assert isinstance(each_G_V, bool), f"each_G_V:{each_G_V} must be a boolean value."
        assert isinstance(all_Methods, bool), f"all_Methods:{all_Methods} must be a boolean value."
        assert isinstance(each_Method, bool), f"each_Method:{each_Method} must be a boolean value."
        assert isinstance(dpi, int) and dpi > 0, f"The dpi:{dpi} parameter must be a positive integer."
        # Remove the no change point data as it is meaningless
        per_table = per_table[per_table['Gap Size'] != 0]
        # Separate unique methods rows
        per_table[['Model', 'Parameters']] = per_table['Model (Parameters)'].str.split(" ", n = 1, expand = True)

        # Get the tables for each method
        tm_table = per_table[per_table['Model'] == 'T'] # Select rows that have T
        wm_table = per_table[per_table['Model'] == 'W'] # Select rows that have W
        swm_table = per_table[per_table['Model'] == 'M'] # Select rows that have M
        ctm_table = per_table[per_table['Model'] == 'CT'] # Select rows that have CT

        # Set style and palette
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.color_palette("crest", as_cmap=True)
        n_colors = len(self.tm_params_list) + len(self.wm_params_list) + len(self.swm_params_list) + len(self.ctm_params_list) # the total number of unique parameters
        colors = sns.color_palette("crest", n_colors=n_colors)
        colors_each = sns.color_palette("crest", n_colors=len(self.variances))

        if save:
            base_dir = os.path.join("Plots", 'Robust_Params_graphs', 'ARL_1_graphs')
            os.makedirs(base_dir, exist_ok=True) # make the directory to save graphs

        # Create boxplots for each gap size and z & alpha value
        if each_G == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_gap_za'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for za in per_table['z and alpha'].unique():
                subset_za = per_table[per_table['z and alpha'] == za]
                for gap_size in subset_za['Gap Size'].unique():
                    subset_df = subset_za[subset_za['Gap Size'] == gap_size]
                    unique_models = subset_df['Model (Parameters)'].unique()
                    color_mapping = {model: color for model, color in zip(unique_models, colors)}
                    plt.figure(figsize=(20, 8))
                    ax =sns.boxplot(x='Model (Parameters)', y='ARL1', data=subset_df, palette=color_mapping)
                    ax.tick_params(labelsize=10)
                    if self.outlier_position is None:
                        plt.title(f'Boxplot of $ARL_1$ for Various Models on Streaming Data with z and alpha: {za}, Mean Gap Size of {gap_size} (No Outliers)', fontsize=18)
                    else:
                        plt.title(f'Boxplot of $ARL_1$ for Various Models on Streaming Data with z and alpha: {za}, Mean Gap Size of {gap_size} (Outliers in {self.outlier_position} Period)', fontsize=18)
                    plt.ylabel('$ARL_1$', fontsize=13)
                    plt.xlabel('Model (Parameters)', fontsize=14)
                    plt.xticks(rotation=30)
                    if save:
                        # Define the filename based on the specific graph parameters.
                        filename = f"arl1_gap_{gap_size}_za_{za}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                        plt.tight_layout()
                        plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                    plt.show()

        # Create boxplots for each gap size, variance size and z & alpha value
        if each_G_V == True:
            if save:
                # Define the graph type directory name based on the current boxplot.
                graph_type = 'each_gap_&_var_za'
                graph_dir = os.path.join(base_dir, graph_type)
                os.makedirs(graph_dir, exist_ok=True)
            for za in per_table['z and alpha'].unique():
                subset_za = per_table[per_table['z and alpha'] == za]
                for gap_size in subset_za['Gap Size'].unique():
                    for vari in subset_za['Data Var'].unique():
                        subset_df = subset_za[(subset_za['Gap Size'] == gap_size) & (subset_za['Data Var'] == vari)]
                        unique_models = subset_df['Model (Parameters)'].unique()
                        color_mapping = {model: color for model, color in zip(unique_models, colors)}
                        plt.figure(figsize=(20, 8))
                        ax =sns.boxplot(x='Model (Parameters)', y='ARL1', data=subset_df, palette=color_mapping)
                        ax.tick_params(labelsize=10)
                        if self.outlier_position is None:
                            plt.title(f'Boxplot of $ARL_1$ for Various Models on Streaming Data with z and alpha: {za}, Mean Gap Size of {gap_size} and Data Variance of {vari} (No Outliers)', fontsize=18)
                        else:
                            plt.title(f'Boxplot of $ARL_1$ for Various Models on Streaming Data with z and alpha: {za}, Mean Gap Size of {gap_size}, Data Variance of {vari} (Outliers in {self.outlier_position} Period)', fontsize=18)
                        plt.ylabel('$ARL_1$', fontsize=13)
                        plt.xlabel('Model (Parameters)', fontsize=14)
                        plt.xticks(rotation=30)
                        if save:
                            # Define the filename based on the specific graph parameters.
                            filename = f"arl1_gap_{gap_size}_var{vari}_za_{za}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                            plt.tight_layout()
                            plt.savefig(os.path.join(graph_dir, filename), dpi=dpi, format='pdf')
                        plt.show()
        
        # Create boxplots for all four models describing their general performances in each data mean gap and variance setting
        if all_Methods == True:
            models = {
                'T': tm_table,
                'W': wm_table,
                'M': swm_table,
                'CT': ctm_table
            }
            # Iterate over each model
            for model_name, model_table in models.items():
                # Create a boxplot
                plt.figure(figsize=(14, 8))
                ax = sns.boxplot(data=model_table, x='Gap Size', y='ARL1', hue='Data Var', palette=colors_each)
                ax.legend(fontsize=14)
                ax.tick_params(labelsize=14)
                # Titles and labels
                if self.outlier_position is None:
                    plt.title(f'$ARL_1$ Values of {model_name} Model in Streaming Data Without Outliers', fontsize=18)
                else:
                    plt.title(f'$ARL_1$ Values of {model_name} Model in Streaming Data with Outliers in {self.outlier_position} Period', fontsize=18)
                plt.ylabel('$ARL_1$', fontsize=14)
                plt.xlabel('Gap Size', fontsize=14)
                # Save the plot
                if save:
                    filename = f"arl1_{model_name}_model_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                    plt.tight_layout()
                    plt.savefig(os.path.join(base_dir, filename), dpi=dpi, format='pdf')
                plt.show()

        # Create boxplots for each parameter setting in all four models describing their general performances in each data mean gap and variance setting
        if each_Method == True:
            models = {
                'T': tm_table,
                'W': wm_table,
                'M': swm_table,
                'CT': ctm_table
            }
            for model_name, model_table in models.items():
                # Extract unique parameters for the current model
                model_params = model_table['Model (Parameters)'].unique()
                # Create directory for this model
                model_dir = os.path.join(base_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                # Iterate over each unique parameter and create a boxplot 
                for params in model_params:
                    z_alpha_values = model_table.loc[model_table['Model (Parameters)']==params, 'z and alpha'].values[0] # receive z and alpha value
                    control_parameters = params.split('(')[-1].split(')')[0] # split for the last ( and first ) to receive control parameter
                    plt.figure(figsize=(14, 8))
                    ax = sns.boxplot(data=model_table[model_table['Model (Parameters)']==params], x='Gap Size', y='ARL1', hue='Data Var', palette=colors_each)
                    ax.legend(fontsize=14)
                    ax.tick_params(labelsize=14)
                    if self.outlier_position is None:
                        plt.title(f'The Values of $ARL_1$ for Model {model_name} ({z_alpha_values}, Control Parameters: {control_parameters}) under Different Streaming Data Settings, Without Outliers', fontsize=16)
                    else:
                        plt.title(f'The Values of $ARL_1$ for Model {model_name} ({z_alpha_values}, Control Parameters: {control_parameters}) under Different Streaming Data Settings with Outliers in {self.outlier_position} period', fontsize=16)
                    plt.ylabel('$ARL_1$', fontsize=14)
                    plt.xlabel('Gap Size', fontsize=14)
                    # Save the plot
                    if save:
                        filename = f"arl1_{params}_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf"
                        plt.tight_layout()
                        plt.savefig(os.path.join(model_dir, filename), dpi=dpi, format='pdf')
                    plt.show()

    def plot_best_models(self, save:bool=False, dpi:int=500):
        """
        This function takes a pandas dataframe and plots the ARL0 and ARL1 values for the best CUSUM and EWMA models for each gap size.
        The dataframe should have the columns 'Gap Size' and 'Model (Parameters)' and also ('ARL0', 'mean'), ('ARL0', 'std'), ('ARL1', 'mean'), and ('ARL1', 'std').

        Parameters:
        save (bool, optional): If True, the function will save each plot to a sub-folder in the 'ARL_0_graphs' directory.
        dpi (int, optional): The resolution in dots per inch for saved figures.

        Returns:
        None: The function generates plots
        """
        if not hasattr(self, 'C_E_performance_table'):
            self.grid_C_E_params_eval()
        C_E_per_table = self.C_E_performance_table
        if not hasattr(self, 'robust_performance_table'):
            self.grid_robust_params_eval()
        robust_per_table = self.robust_performance_table
        # Assert that input is a pandas DataFrame
        assert isinstance(save, bool), "The save parameter must be a boolean."
        assert isinstance(dpi, int) and dpi > 0, f"The dpi:{dpi} parameter must be a positive integer."
        # Find the table that contains mean and std of ARL0 & ARL1 for each model and each gap size
        C_E_per_table = C_E_per_table.groupby([f'Gap Size','Model (Parameters)']).agg({'ARL0':['mean', 'std'], 
                                                                            'ARL1':['mean', 'std']}).reset_index()
        robust_per_table = robust_per_table.groupby([f'Gap Size','Model (Parameters)']).agg({'ARL0':['mean', 'std'], 
                                                                            'ARL1':['mean', 'std']}).reset_index()
        # Best models for cusum and ewma of different gap sizes
        best_arl0_cusums = pd.DataFrame()
        best_arl0_ewmas = pd.DataFrame()
        best_arl1_cusums = pd.DataFrame()
        best_arl1_ewmas = pd.DataFrame()

        for gap in C_E_per_table['Gap Size'].unique():
            # Filter data for specific gap
            gap_data = C_E_per_table[C_E_per_table['Gap Size'] == gap]
            # Separate tables for cusum and ewma
            cusum_data = gap_data[gap_data['Model (Parameters)'].str.contains('CUSUM')]
            ewma_data = gap_data[gap_data['Model (Parameters)'].str.contains('EWMA')]
            # Sort by mean and standard deviation
            cusum_data = cusum_data.sort_values(by=[('ARL0', 'mean'), ('ARL0', 'std')], ascending=[False, True])
            ewma_data = ewma_data.sort_values(by=[('ARL0', 'mean'), ('ARL0', 'std')], ascending=[False, True])    
            # Append the best model for ARL0
            best_arl0_cusums = pd.concat([best_arl0_cusums, cusum_data.iloc[[0]]])
            best_arl0_ewmas = pd.concat([best_arl0_ewmas, ewma_data.iloc[[0]]])  
            # Do the same for ARL1 but not for gap = 0
            if gap != 0:
                # Now do the same for ARL1, but remember that for ARL1, lower is better
                cusum_data = cusum_data.sort_values(by=[('ARL1', 'mean'), ('ARL1', 'std')], ascending=True)
                ewma_data = ewma_data.sort_values(by=[('ARL1', 'mean'), ('ARL1', 'std')], ascending=True)
                # Append the best model for ARL1
                best_arl1_cusums = pd.concat([best_arl1_cusums, cusum_data.iloc[[0]]])
                best_arl1_ewmas = pd.concat([best_arl1_ewmas, ewma_data.iloc[[0]]])

        # Best models for TM, WM, SWM, and CTM of different gap sizes
        best_arl0_tms = pd.DataFrame()
        best_arl1_tms = pd.DataFrame()
        best_arl0_wms = pd.DataFrame()
        best_arl1_wms = pd.DataFrame()
        best_arl0_swms = pd.DataFrame()
        best_arl1_swms = pd.DataFrame()
        best_arl0_ctms = pd.DataFrame()
        best_arl1_ctms = pd.DataFrame()

        for gap in robust_per_table['Gap Size'].unique():
            # Filter data for specific gap
            gap_data = robust_per_table[robust_per_table['Gap Size'] == gap]
            gap_data[['Model', 'Parameters']] = gap_data['Model (Parameters)'].str.split(" ", n = 1, expand = True)
            # Separate tables for TM, WM, SWM, and CTM
            tm_data = gap_data[gap_data['Model'] == 'T'] 
            wm_data = gap_data[gap_data['Model'] == 'W'] 
            swm_data = gap_data[gap_data['Model'] == 'M'] 
            ctm_data = gap_data[gap_data['Model'] == 'CT']
            # Sort by mean and standard deviation
            tm_data = tm_data.sort_values(by=[('ARL0', 'mean'), ('ARL0', 'std')], ascending=[False, True])
            wm_data = wm_data.sort_values(by=[('ARL0', 'mean'), ('ARL0', 'std')], ascending=[False, True])
            swm_data = swm_data.sort_values(by=[('ARL0', 'mean'), ('ARL0', 'std')], ascending=[False, True])
            ctm_data = ctm_data.sort_values(by=[('ARL0', 'mean'), ('ARL0', 'std')], ascending=[False, True])
            # Append the best model for ARL0
            best_arl0_tms = pd.concat([best_arl0_tms, tm_data.iloc[[0]]])
            best_arl0_wms = pd.concat([best_arl0_wms, wm_data.iloc[[0]]])
            best_arl0_swms = pd.concat([best_arl0_swms, swm_data.iloc[[0]]])
            best_arl0_ctms = pd.concat([best_arl0_ctms, ctm_data.iloc[[0]]])
            # Do the same for ARL1 but not for gap = 0
            if gap != 0:
                # Now do the same for ARL1, but remember that for ARL1, lower is better
                tm_data = tm_data.sort_values(by=[('ARL1', 'mean'), ('ARL1', 'std')], ascending=True)
                wm_data = wm_data.sort_values(by=[('ARL1', 'mean'), ('ARL1', 'std')], ascending=True)
                swm_data = swm_data.sort_values(by=[('ARL1', 'mean'), ('ARL1', 'std')], ascending=True)
                ctm_data = ctm_data.sort_values(by=[('ARL1', 'mean'), ('ARL1', 'std')], ascending=True)
                # Append the best model for ARL1
                best_arl1_tms = pd.concat([best_arl1_tms, tm_data.iloc[[0]]])
                best_arl1_wms = pd.concat([best_arl1_wms, wm_data.iloc[[0]]])
                best_arl1_swms = pd.concat([best_arl1_swms, swm_data.iloc[[0]]])
                best_arl1_ctms = pd.concat([best_arl1_ctms, ctm_data.iloc[[0]]])

        # Create evenly spaced x values for plots
        x_arl0 = np.arange(len(best_arl0_cusums))
        x_arl1 = np.arange(len(best_arl1_cusums))

        # Define offsets for annotation positions
        offsets_cusum = np.array([-0.05, 10])  # adjust as needed
        offsets_ewma = np.array([0.05, -5])  # adjust as needed
        offsets_tm = np.array([-0.05, 15])  # adjust as needed
        offsets_wm = np.array([0.05, 20])  # adjust as needed
        offsets_swm = np.array([-0.05, 25])  # adjust as needed
        offsets_ctm = np.array([0.05, 30]) # adjust as needed
        # Define colors
        color_cusum = '#003E74' # Imeprial blue
        color_ewma = '#379f9f' # Seaglass
        color_tm = '#02893B' # Dark Green
        color_wm = '#002147' # Navy
        color_swm = '#373A36' # Dark Grey
        color_ctm = '#BBCE00' # Lime

        # Plot for ARL0 of the best models for different gap sizes
        plt.figure(figsize=(16, 8))
        # CUSUM ARL0
        plt.plot(x_arl0, best_arl0_cusums[('ARL0', 'mean')], 'o-', color=color_cusum, label='CUSUM')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl0_cusums['Model (Parameters)']):
            plt.annotate(txt, (x_arl0[i], best_arl0_cusums[('ARL0', 'mean')].iloc[i]) + offsets_cusum, 
                        fontsize=12, color=color_cusum)
        # EWMA ARL0
        plt.plot(x_arl0, best_arl0_ewmas[('ARL0', 'mean')], 'x-', color=color_ewma, label='EWMA')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl0_ewmas['Model (Parameters)']):
            plt.annotate(txt, (x_arl0[i], best_arl0_ewmas[('ARL0', 'mean')].iloc[i]) + offsets_ewma, 
                        fontsize=12, color=color_ewma)
        # TM ARL0
        plt.plot(x_arl0, best_arl0_tms[('ARL0', 'mean')], 'o-', color=color_tm, label='T')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl0_tms.loc[:, 'Model (Parameters)']):
            plt.annotate(txt, (x_arl0[i], best_arl0_tms.loc[:, ('ARL0', 'mean')].iloc[i]) + offsets_tm, 
                        fontsize=12, color=color_tm) 
        # WM ARL0
        plt.plot(x_arl0, best_arl0_wms[('ARL0', 'mean')], 'x-', color=color_wm, label='W')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl0_wms.loc[:, 'Model (Parameters)']):
            plt.annotate(txt, (x_arl0[i], best_arl0_wms.loc[:, ('ARL0', 'mean')].iloc[i]) + offsets_wm, 
                        fontsize=12, color=color_wm)   
        # SWM ARL0
        plt.plot(x_arl0, best_arl0_swms[('ARL0', 'mean')], 'o-', color=color_swm, label='M')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl0_swms.loc[:, 'Model (Parameters)']):
            plt.annotate(txt, (x_arl0[i], best_arl0_swms.loc[:, ('ARL0', 'mean')].iloc[i]) + offsets_swm, 
                        fontsize=12, color=color_swm)
        # CTM ARL0
        plt.plot(x_arl0, best_arl0_ctms[('ARL0', 'mean')], 'x-', color=color_ctm, label='CT')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl0_ctms.loc[:, 'Model (Parameters)']):
            plt.annotate(txt, (x_arl0[i], best_arl0_ctms.loc[:, ('ARL0', 'mean')].iloc[i]) + offsets_ctm, 
                        fontsize=12, color=color_ctm) 
        if self.outlier_position is None:
            plt.title('Mean $ARL_0$ of Optimal Models for Streaming Data Across Different Mean Gap Sizes (No Outliers)', fontsize=16)
        else:
            plt.title(f'Mean $ARL_0$ of Optimal Models for Streaming Data Across Different Mean Gap Sizes (Outliers Present in {self.outlier_position} Period)', fontsize=16)
        plt.xlabel('Gap Size', fontsize=14)
        plt.ylabel('$ARL_0$ mean', fontsize=14)
        plt.xticks(x_arl0, best_arl0_cusums['Gap Size'], fontsize=14)
        plt.yticks(fontsize=14) 
        plt.legend(fontsize=14)
        if save:
            save_path = os.path.join("Plots", f"mean_arl0_best_models_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf")
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi)
        plt.show()

        # Plot for ARL1 of the best models for different gap sizes
        plt.figure(figsize=(16, 8))
        # CUSUM ARL1
        plt.plot(x_arl1, best_arl1_cusums[('ARL1', 'mean')], 'o-', color=color_cusum, label='CUSUM')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl1_cusums['Model (Parameters)']):
            plt.annotate(txt, (x_arl1[i], best_arl1_cusums[('ARL1', 'mean')].iloc[i]) + offsets_cusum, 
                        fontsize=12, color=color_cusum)
        # EWMA ARL1
        plt.plot(x_arl1, best_arl1_ewmas[('ARL1', 'mean')], 'x-', color=color_ewma, label='EWMA')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl1_ewmas['Model (Parameters)']):
            plt.annotate(txt, (x_arl1[i], best_arl1_ewmas[('ARL1', 'mean')].iloc[i]) + offsets_ewma, 
                        fontsize=12, color=color_ewma)
        # TM ARL1
        plt.plot(x_arl1, best_arl1_tms[('ARL1', 'mean')], 'o-', color=color_tm, label='T')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl1_tms.loc[:, 'Model (Parameters)']):
            plt.annotate(txt, (x_arl1[i], best_arl1_tms.loc[:, ('ARL1', 'mean')].iloc[i]) + offsets_tm, 
                        fontsize=12, color=color_tm)            
        # WM ARL1
        plt.plot(x_arl1, best_arl1_wms[('ARL1', 'mean')], 'x-', color=color_wm, label='W')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl1_wms.loc[:, 'Model (Parameters)']):
            plt.annotate(txt, (x_arl1[i], best_arl1_wms.loc[:, ('ARL1', 'mean')].iloc[i]) + offsets_wm, 
                        fontsize=12, color=color_wm)    
        # SWM ARL1
        plt.plot(x_arl1, best_arl1_swms[('ARL1', 'mean')], 'o-', color=color_swm, label='M')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl1_swms.loc[:, 'Model (Parameters)']):
            plt.annotate(txt, (x_arl1[i], best_arl1_swms.loc[:, ('ARL1', 'mean')].iloc[i]) + offsets_swm, 
                        fontsize=12, color=color_swm)
        # CTM ARL1
        plt.plot(x_arl1, best_arl1_ctms[('ARL1', 'mean')], 'x-', color=color_ctm, label='CT')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl1_ctms.loc[:, 'Model (Parameters)']):
            plt.annotate(txt, (x_arl1[i], best_arl1_ctms.loc[:, ('ARL1', 'mean')].iloc[i]) + offsets_ctm, 
                        fontsize=12, color=color_ctm)
        if self.outlier_position is None:
            plt.title('Mean $ARL_1$ of Optimal Models for Streaming Data Across Different Mean Gap Sizes (No Outliers)', fontsize=16)
        else:
            plt.title(f'Mean $ARL_1$ of Optimal Models for Streaming Data Across Different Mean Gap Sizes (Outliers Present in {self.outlier_position} Period)', fontsize=16)
        plt.xlabel('Gap Size', fontsize=14)
        plt.ylabel('$ARL_1$ mean', fontsize=14)
        plt.xticks(x_arl1, best_arl1_cusums['Gap Size'], fontsize=14)
        plt.yticks(fontsize=14) 
        plt.legend(fontsize=14)
        if save:
            save_path = os.path.join("Plots", f"mean_arl1_best_models_outliers_{self.outlier_position if self.outlier_position is not None else 'none'}.pdf")
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi)
        plt.show()

# ------------------Testing function for grid_C_E_params_eval function without outlier-------------------
# # For displaying the full pd.df
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 200)
# # Setup initial values
# n_sam_bef_cp = 500
# n_sam_aft_cp = 400
# gap_sizes = [1, 5, 10]
# variances = [1, 4, 9]
# seeds = [111, 222, 333, 666, 999]
# BURNIN = 100
# cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
# ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
# # simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
# grideval = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, 
#                             seeds, BURNIN, cusum_params_list, ewma_params_list, None)
# per_table, per_summary = grideval.grid_C_E_params_eval()

# per_summary
# per_table

# grideval.plot_C_E_ARL0_graphs(save=True)
# grideval.plot_C_E_ARL1_graphs(save=True)
# grideval.plot_best_models(save=True)
# ------------------End-------------------


# -------------------testing for outliers generation in GridEvaluation class--------------------
# # Setup initial values
# n_sam_bef_cp = 500
# n_sam_aft_cp = 400
# gap_sizes = [1, 5, 10]
# variances = [1, 4, 9]
# seeds = [111, 222, 333, 666, 999]
# BURNIN = 100
# cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
# ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
# valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
# outlier_position = valid_positions[0]
# beta = 1e-5
# outlier_ratio = 0.05
# asymmetric_ratio = 0.25
# # simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
# grideval_outliers = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, seeds, BURNIN,
#                              cusum_params_list, ewma_params_list, outlier_position, alpha, outlier_ratio, asymmetric_ratio)
# per_table, per_summary = grideval_outliers.grid_C_E_params_eval()
# grideval_outliers.plot_C_E_ARL0_graphs(save=True)
# grideval_outliers.plot_C_E_ARL1_graphs(save=True)
# grideval_outliers.plot_best_models(save=True)
# outlier_grid_data = grideval_outliers.generate_with_outliers_grid_data(seeds[0])
# outlier_grid_data[0][0].shape
# per_table.iloc[50]
# ------------------End-------------------

def simulate_stream_data(v=50, G=50, D=50, M=5000, S=[0.25, 0.5, 1, 3], sigma=1, seed=666):
    """
    This function generates simulated stream data with randomly spaced changepoints.

    Parameters:
    v (int): Average number of time points between changepoints. Should be a positive integer.
    G (int): Grace period to allow the algorithm to estimate the stream's parameters. Should be a positive integer.
    D (int): Detection period to allow the algorithm to detect a change. Should be a positive integer.
    M (int): Total number of changes in the stream. Should be a positive integer.
    S (list): Set of potential jump sizes. Should be a list of positive numbers.
    sigma (float): Standard deviation of the normal distributions. Should be a non-negative number.
    seed (int): Seed for the random number generator. 

    Returns:
    tuple: A tuple containing four np.arrays. 
    data_stream: The simulated stream data.
    tau_list: The list of changepoints.
    mu_list: The list of means of the blocks.
    size_list: The list of sizes of the blocks.
    """
    assert isinstance(v, int) and v > 0, f"v={v} must be a positive integer"
    assert isinstance(G, int) and G > 0, f"G={G} must be a positive integer"
    assert isinstance(D, int) and D > 0, f"D={D} must be a positive integer"
    assert isinstance(M, int) and M > 0, f"M={M} must be a positive integer"
    assert isinstance(S, list) and all(isinstance(s, (int, float)) and s > 0 for s in S), f"S={S} must be a list of positive numbers"
    assert isinstance(sigma, (int, float)) and sigma >= 0, f"sigma={sigma} must be a non-negative number"
    assert isinstance(seed, int), f"seed={seed} must be an integer"
    np.random.seed(seed) # set seed
    mu_list = [0]  # list of mean mu, first mean mu = 0
    # Generate tau values (changepoints) and size_list
    first_tau = G + np.random.poisson(v) # first changepoint
    tau_list = [first_tau]  # add it into the list
    size_list = [first_tau] # first size, equals to tau
    block = np.random.normal(mu_list[0], sigma, size_list[0])  # sample from normal distribution for the first stream
    data_stream =list(block) # initialise the data_stream list
    for i in range(1, M):
        tau_list.append(tau_list[i-1] + D + G + np.random.poisson(v))  # following tau
    # Generate data stream blocks using for loop
    for i in range(M):
        # Compute new mean
        theta = np.random.choice([+1, -1])  # sample the random sign
        delta = np.random.choice(S)  # sample the random jump size
        mu = mu_list[i] + theta * delta  # new mean
        mu_list.append(mu) # add mu into the list
        # Compute block size of the random samples between two changepoints
        if i < M - 1:
            size = tau_list[i+1] - tau_list[i]  # next tau - current tau
        else:
            size = G + D + v  # last block
        size_list.append(size) # append the size
        block = np.random.normal(mu, sigma, size)  # sample from normal distribution
        data_stream.extend(block) # extend the data_stream list
    # transform to np.array
    data_stream, tau_list, mu_list, size_list= np.array(data_stream), np.array(tau_list), np.array(mu_list),  np.array(size_list)
    return data_stream, tau_list, mu_list, size_list

def stream_data_plot(data_stream, tau_list, G=50):
    """
    Function to plot simulate data stream with true changepoints and shadowed burn-in periods.
    
    Parameters:
    - data_stream (np.array): The generated stream data.
    - tau_list (np.array): Array of changepoints (tau values).
    - G (int): Length of the burn-in period. Default is 50.
    
    Returns:
    - A matplotlib figure showing the stream data with changepoints and burn-in periods.
    """
    plt.figure(figsize=(45, 6))
    plt.plot(data_stream, label='Stream Data')
    plt.fill_betweenx([min(data_stream), max(data_stream)], 0, G, color='gray', alpha=0.25, label='Burn-in')
    for i, tau in enumerate(tau_list):
        plt.axvline(x=tau, color='firebrick', linestyle='--')
        plt.annotate(f'$\\tau_{i+1}$', xy=(tau, min(data_stream)), xytext=(tau, min(data_stream)), 
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.fill_betweenx([min(data_stream), max(data_stream)], tau+G, tau, color='gray', alpha=0.25)  # shadows the area of burn-in
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title('Stream Data with Changepoints', fontsize=20)
    plt.legend(fontsize=14)
    plt.show()

# -------------------streaming data simulation with plot test code--------------------

# data_stream, tau_list, mu_list, size_list = simulate_stream_data(v=50, G=50, D=50, M=50, S=[0.25, 0.5, 1, 3], sigma=1, seed=666)
# stream_data_plot(data_stream, tau_list)
# tau_list.shape
# size_list.sum()
# data_stream.shape
# ------------------End-------------------
