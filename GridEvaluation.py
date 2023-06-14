import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ARLFunc import arl_cusum, arl_ewma
from Outliers import OutlierInjector


class GridDataEvaluate:
    """
    A class to evaluate the performance of the CUSUM and EWMA models in the context of change point detection.
    The class uses grid parameters to simulate data and calculate the Average Run Length (ARL).
    
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
        alpha (float, optional): The threshold probability of occurrence for the outliers (default is None, should between (0,1)).
        outlier_ratio (float, optional): The ratio of data points of the given outlier position period to be considered outliers (default is None, should be between (0,1)).

    Methods:
        generate_no_outlier_grid_data():
            Simulates grid data that have no outlier for the provided parameters.
        generate_with_outliers_grid_data():
            Simulates grid data that have outliers for the provided parameters.
        grid_params_eval():
            Evaluates the performance of control charts with different parameters by calculating the ARLs.
        plot_ARL0_graphs(each_G:bool, each_G_V:bool, all_CUSUM:bool, each_CUSUM:bool, all_EWMA:bool, each_EWMA:bool):
            Plots different types of box plots to visualize ARL0 values under various conditions.
        plot_ARL1_graphs():
            Plot function for ARL1. Specific parameters to be provided based on implementation.
        plot_best_models():
            Plot function for best models of CUSUM and EWMA w.r.t. ARL0 and ARL1 values.
    """
    def __init__(self, n_sam_bef_cp:int, n_sam_aft_cp:int, gap_sizes:list, variances:list,  seeds:list, burnin:int, 
                 cusum_params_list:list, ewma_params_list:list, outlier_position:str, alpha:float=None, outlier_ratio:float=None):
        assert isinstance(n_sam_bef_cp, int) and n_sam_bef_cp > 0, f"n_sam_bef_cp:{n_sam_bef_cp} must be a positive integer"
        assert isinstance(n_sam_aft_cp, int) and n_sam_aft_cp > 0, f"n_sam_aft_cp:{n_sam_aft_cp} must be a positive integer"
        assert isinstance(gap_sizes, (list, int, float)) and (all(isinstance(i, (int, float)) and i >= 0 for i in gap_sizes) if isinstance(gap_sizes, list) else gap_sizes >= 0), f"gap_sizes:{gap_sizes} must be a number or list of non-negetive numbers"
        assert isinstance(variances, (list, int, float)) and (all(isinstance(i, (int, float)) and i >= 0 for i in variances) if isinstance(variances, list) else variances >= 0), f"variances:{variances} must be a number or list of non-negetive numbers"
        assert isinstance(seeds, (list, int)) and (all(isinstance(i, int) for i in seeds) if isinstance(seeds, list) else seeds>=0), f"seeds:{seeds} must be a non-negative integer or list of non-negative integers"
        assert isinstance(burnin, int) and burnin >= 0, f"burnin:{burnin} must be a non-negative integer"
        assert burnin < n_sam_bef_cp, f"Value of burnin:{burnin} should smaller than n_sam_bef_cp:{n_sam_bef_cp}"
        assert isinstance(cusum_params_list, list) and all(isinstance(i, tuple) and len(i)==2 for i in cusum_params_list), f"cusum_params_list:{cusum_params_list} must be a list of tuples each of size 2"
        assert isinstance(ewma_params_list, list) and all(isinstance(i, tuple) and len(i)==2 for i in ewma_params_list), f"ewma_params_list:{ewma_params_list} must be a list of tuples each of size 2"
        assert outlier_position is None or isinstance(outlier_position, str), f"outlier_position:{outlier_position} must be either None or a string"
        valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
        if outlier_position is not None:
            assert alpha is None or (isinstance(alpha, float) and 0 < alpha < 1), f"alpha:{alpha} must be a float within the range [0,1]"
            assert outlier_ratio is None or (isinstance(outlier_ratio, float) and 0 < outlier_ratio < 1), f"outlier_ratio:{outlier_ratio} must be a float within the range (0,1)"
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
        self.outlier_position = outlier_position
        self.alpha = alpha
        self.outlier_ratio = outlier_ratio

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
        # Without change in mean but different variance
        for variance in self.variances:
            data_without_outliers = np.random.normal(scale=np.sqrt(variance),size=self.n_sam_bef_cp + self.n_sam_aft_cp)
            outinj = OutlierInjector(data_without_outliers, self.n_sam_bef_cp, self.n_sam_aft_cp, self.burnin, variance, 0, variance, 
                                    self.alpha, outlier_ratio=self.outlier_ratio, outlier_position=self.outlier_position)
            data_with_outliers = outinj.insert_outliers()
            simulate_data_list.append((data_with_outliers, None, 0, variance, outinj.outlier_indices)) # extra list of outlier indices
        # With increase/decrease in mean and different variance
        for gap_size in self.gap_sizes:
            for variance in self.variances:
                # Mean increase
                data_with_increase = np.append(np.random.normal(size=self.n_sam_bef_cp, scale=np.sqrt(variance)), 
                                        np.random.normal(loc=gap_size, scale=np.sqrt(variance), size=self.n_sam_aft_cp))
                outinj = OutlierInjector(data_with_increase, self.n_sam_bef_cp, self.n_sam_aft_cp, self.burnin, variance, gap_size, 
                                    variance, self.alpha, outlier_ratio=self.outlier_ratio, outlier_position=self.outlier_position)
                data_with_increase_outliers = outinj.insert_outliers()
                simulate_data_list.append((data_with_increase_outliers, self.n_sam_bef_cp, gap_size, variance, outinj.outlier_indices))
                # Mean decrease
                data_with_decrease = np.append(np.random.normal(size=self.n_sam_bef_cp, scale=np.sqrt(variance)), 
                                        np.random.normal(loc=-gap_size, scale=np.sqrt(variance), size=self.n_sam_aft_cp))
                outinj = OutlierInjector(data_with_decrease, self.n_sam_bef_cp, self.n_sam_aft_cp, self.burnin, variance, -gap_size, 
                                    variance, self.alpha, outlier_ratio=self.outlier_ratio, outlier_position=self.outlier_position)
                data_with_decrease_outliers = outinj.insert_outliers()
                simulate_data_list.append((data_with_decrease_outliers, self.n_sam_bef_cp, -gap_size, variance, outinj.outlier_indices))
        return simulate_data_list

    def grid_params_eval(self):
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
        self.performance_table = performance_table
        self.performance_summary = performance_summary
        return performance_table, performance_summary.round(4) # 4 decimal places

    def plot_ARL0_graphs(self, each_G:bool=False, each_G_V:bool=False, all_CUSUM:bool=False, each_CUSUM:bool=False, all_EWMA:bool=False, each_EWMA:bool=False):
        """
        This function creates different types of box plots to visualize ARL0 values for different conditions.
        
        Parameters:
        each_G (bool): If True, the function will create a boxplot for each unique gap size.
        each_G_V (bool): If True, the function will create a boxplot for each unique gap size and variance.
        all_CUSUM (bool): If True, the function will create a boxplot for the CUSUM model.
        each_CUSUM (bool): If True, the function will create a boxplot for each unique model parameter for CUSUM.
        all_EWMA (bool): If True, the function will create a boxplot for the EWMA model.
        each_EWMA (bool): If True, the function will create a boxplot for each unique model parameter for EWMA.
        
        Returns:
        None: The function generates plots
        """
        if not hasattr(self, 'performance_table'):
            self.grid_params_eval()
        per_table = self.performance_table
        # Assertions to validate input data types
        assert isinstance(per_table, pd.DataFrame), "per_table must be a pandas DataFrame."
        assert isinstance(each_G, bool), f"each_G:{each_G} must be a boolean value."
        assert isinstance(each_G_V, bool), f"each_G_V:{each_G_V} must be a boolean value."
        assert isinstance(all_CUSUM, bool), f"all_CUSUM:{all_CUSUM} must be a boolean value."
        assert isinstance(each_CUSUM, bool), f"each_CUSUM:{each_CUSUM} must be a boolean value."
        assert isinstance(all_EWMA, bool), f"all_EWMA:{all_EWMA} must be a boolean value."
        assert isinstance(each_EWMA, bool), f"each_EWMA:{each_EWMA} must be a boolean value."
        # Separate CUSUM and EWMA rows
        cusum_table = per_table[per_table['Model (Parameters)'].str.contains('CUSUM')] # Select col that have CUSUM
        ewma_table = per_table[per_table['Model (Parameters)'].str.contains('EWMA')] # Select col that have EWMA
        # unique model parameters for CUSUM and EWMA
        cusum_params = cusum_table['Model (Parameters)'].unique()
        ewma_params = ewma_table['Model (Parameters)'].unique()
        # Set style and palette
        sns.set_style("whitegrid")
        sns.set_palette("viridis")

        # Create a boxplot for each gap size
        if each_G == True:
            for gap_size in per_table['Gap Size'].unique():
                subset_df = per_table[per_table['Gap Size'] == gap_size]
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='Model (Parameters)', y='ARL0', data=subset_df)
                if self.outlier_position is None:
                    plt.title(f'Boxplot of $ARL_0$ for Gap Size {gap_size}')
                else:
                    plt.title(f'Boxplot of $ARL_0$ for Gap Size {gap_size} with outliers in {self.outlier_position} period')
                plt.ylabel('$ARL_0$')
                plt.xlabel('Model (Parameters)')
                plt.xticks(rotation=30)
                plt.show()

        # Create a boxplot for each gap size and variance size
        if each_G_V == True:
            for gap_size in per_table['Gap Size'].unique():
                for vari in per_table['Data Var'].unique():
                    subset_df = per_table[(per_table['Gap Size'] == gap_size) & (per_table['Data Var'] == vari)]
                    plt.figure(figsize=(20, 10))
                    sns.boxplot(x='Model (Parameters)', y='ARL0', data=subset_df)
                    if self.outlier_position is None:
                        plt.title(f'Boxplot of $ARL_0$ for Gap Size {gap_size} and Data Variance {vari}')
                    else:
                        plt.title(f'Boxplot of $ARL_0$ for Gap Size {gap_size} and Data Variance {vari} with outliers in {self.outlier_position} period')
                    plt.ylabel('$ARL_0$')
                    plt.xlabel('Model (Parameters)')
                    plt.xticks(rotation=30)
                    plt.show()

        # Create box plots for CUSUM model
        if all_CUSUM == True:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=cusum_table, x='Gap Size', y='ARL0', hue='Data Var')
            if self.outlier_position is None:
                plt.title('CUSUM Model')
            else:
                plt.title(f'CUSUM Model with outliers in {self.outlier_position} period')
            plt.show()

        # Create box plots for EWMA model
        if all_EWMA == True:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=ewma_table, x='Gap Size', y='ARL0', hue='Data Var')
            if self.outlier_position is None:
                plt.title('EWMA Model')
            else:
                plt.title(f'EWMA Model with outliers in {self.outlier_position} period')
            plt.show()

        # Loop through each unique model parameters for CUSUM and create box plot
        if each_CUSUM == True:
            for param in cusum_params:
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=cusum_table[cusum_table['Model (Parameters)'] == param], x='Gap Size', y='ARL0', hue='Data Var')
                if self.outlier_position is None:
                    plt.title(f'Model: {param}')
                else:
                    plt.title(f'Model: {param} with outliers in {self.outlier_position} period')
                plt.show()

        # Loop through each unique model parameters for EWMA and create box plot
        if each_EWMA == True:
            for param in ewma_params:
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=ewma_table[ewma_table['Model (Parameters)'] == param], x='Gap Size', y='ARL0', hue='Data Var')
                if self.outlier_position is None:
                    plt.title(f'Model: {param}')
                else:
                    plt.title(f'Model: {param} with outliers in {self.outlier_position} period')
                plt.show()

    def plot_ARL1_graphs(self, each_G:bool=False, each_G_V:bool=False, all_CUSUM:bool=False, each_CUSUM:bool=False, all_EWMA:bool=False, each_EWMA:bool=False):
        """
        This function creates different types of box plots to visualize ARL1 values for different conditions.
        
        Parameters:
        each_G (bool): If True, the function will create a boxplot for each unique gap size.
        each_G_V (bool): If True, the function will create a boxplot for each unique gap size and variance.
        all_CUSUM (bool): If True, the function will create a boxplot for the CUSUM model.
        each_CUSUM (bool): If True, the function will create a boxplot for each unique model parameter for CUSUM.
        all_EWMA (bool): If True, the function will create a boxplot for the EWMA model.
        each_EWMA (bool): If True, the function will create a boxplot for each unique model parameter for EWMA.
        
        Returns:
        None: The function generates plots
        """
        if not hasattr(self, 'performance_table'):
                self.grid_params_eval()
        per_table = self.performance_table
        # Assertions to validate input data types
        assert isinstance(per_table, pd.DataFrame), "per_table must be a pandas DataFrame."
        assert isinstance(each_G, bool), f"each_G:{each_G} must be a boolean value."
        assert isinstance(each_G_V, bool), f"each_G_V:{each_G_V} must be a boolean value."
        assert isinstance(all_CUSUM, bool), f"all_CUSUM:{all_CUSUM} must be a boolean value."
        assert isinstance(each_CUSUM, bool), f"each_CUSUM:{each_CUSUM} must be a boolean value."
        assert isinstance(all_EWMA, bool), f"all_EWMA:{all_EWMA} must be a boolean value."
        assert isinstance(each_EWMA, bool), f"each_EWMA:{each_EWMA} must be a boolean value."
        # Remove the no change point data as it is meaningless
        per_table = per_table[per_table['Gap Size'] != 0]
        # Separate CUSUM and EWMA rows
        cusum_table = per_table[per_table['Model (Parameters)'].str.contains('CUSUM')] # Select col that have CUSUM
        ewma_table = per_table[per_table['Model (Parameters)'].str.contains('EWMA')] # Select col that have EWMA
        # Unique model parameters for CUSUM and EWMA
        cusum_params = cusum_table['Model (Parameters)'].unique()
        ewma_params = ewma_table['Model (Parameters)'].unique()
        # Set style and palette
        sns.set_style("whitegrid")
        sns.set_palette("viridis")

        # Create a boxplot for each gap size
        if each_G == True:
            for gap_size in per_table['Gap Size'].unique():
                subset_df = per_table[per_table['Gap Size'] == gap_size]
                plt.figure(figsize=(20, 10))
                sns.boxplot(x='Model (Parameters)', y='ARL1', data=subset_df)
                if self.outlier_position is None:
                    plt.title(f'Boxplot of $ARL_1$ for Gap Size {gap_size}')
                else:
                    plt.title(f'Boxplot of $ARL_1$ for Gap Size {gap_size} with outliers in {self.outlier_position} period')
                plt.ylabel('$ARL_1$')
                plt.xlabel('Model (Parameters)')
                plt.xticks(rotation=30)
                plt.show()

        # Create a boxplot for each gap size and variance size
        if each_G_V == True:
            for gap_size in per_table['Gap Size'].unique():
                for vari in per_table['Data Var'].unique():
                    subset_df = per_table[(per_table['Gap Size'] == gap_size) & (per_table['Data Var'] == vari)]
                    plt.figure(figsize=(20, 10))
                    sns.boxplot(x='Model (Parameters)', y='ARL1', data=subset_df)
                    if self.outlier_position is None:
                        plt.title(f'Boxplot of $ARL_1$ for Gap Size {gap_size} and Data Variance {vari}')
                    else:
                        plt.title(f'Boxplot of $ARL_1$ for Gap Size {gap_size} and Data Variance {vari} with outliers in {self.outlier_position} period')
                    plt.ylabel('$ARL_1$')
                    plt.xlabel('Model (Parameters)')
                    plt.xticks(rotation=30)
                    plt.show()

        # Create box plots for CUSUM model
        if all_CUSUM == True:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=cusum_table, x='Gap Size', y='ARL1', hue='Data Var')
            if self.outlier_position is None:
                plt.title('CUSUM Model')
            else:
                plt.title(f'CUSUM Model with outliers in {self.outlier_position} period')

            plt.show()

        # Create box plots for EWMA model
        if all_EWMA == True:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=ewma_table, x='Gap Size', y='ARL1', hue='Data Var')
            if self.outlier_position is None:
                plt.title(f'EWMA Model')
            else:
                plt.title(f'EWMA Model with outliers in {self.outlier_position} period')
            plt.show()

        # Loop through each unique model parameters for CUSUM and create box plot
        if each_CUSUM == True:
            for param in cusum_params:
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=cusum_table[cusum_table['Model (Parameters)'] == param], x='Gap Size', y='ARL1', hue='Data Var')
                if self.outlier_position is None:
                    plt.title(f'Model: {param}')
                else:
                    plt.title(f'Model: {param} with outliers in {self.outlier_position} period')
                plt.show()

        # Loop through each unique model parameters for EWMA and create box plot
        if each_EWMA == True:
            for param in ewma_params:
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=ewma_table[ewma_table['Model (Parameters)'] == param], x='Gap Size', y='ARL1', hue='Data Var')
                if self.outlier_position is None:
                    plt.title(f'Model: {param}')
                else:
                    plt.title(f'Model: {param} with outliers in {self.outlier_position} period')
                plt.show()

    def plot_best_models(self):
        """
        This function takes a pandas dataframe and plots the ARL0 and ARL1 values for the best CUSUM and EWMA models for each gap size.
        The dataframe should have the columns 'Gap Size' and 'Model (Parameters)' and also ('ARL0', 'mean'), ('ARL0', 'std'), ('ARL1', 'mean'), and ('ARL1', 'std').

        Returns:
        None: The function generates plots
        """
        if not hasattr(self, 'performance_table'):
            self.grid_params_eval()
        per_table = self.performance_table
        # Assert that input is a pandas DataFrame
        assert isinstance(per_table, pd.DataFrame), "Input per_table must be a pandas DataFrame."
        # Find the table that contains mean and std of ARL0 & ARL1 for each model and each gap size
        per_table = per_table.groupby([f'Gap Size','Model (Parameters)']).agg({'ARL0':['mean', 'std'], 
                                                                            'ARL1':['mean', 'std']}).reset_index()

        # Best models for cusum and ewma of different gap sizes
        best_arl0_cusums = pd.DataFrame()
        best_arl0_ewmas = pd.DataFrame()
        best_arl1_cusums = pd.DataFrame()
        best_arl1_ewmas = pd.DataFrame()

        for gap in per_table['Gap Size'].unique():
            # Filter data for specific gap
            gap_data = per_table[per_table['Gap Size'] == gap]
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

        # Create evenly spaced x values for plots
        x_arl0 = np.arange(len(best_arl0_cusums))
        x_arl1 = np.arange(len(best_arl1_cusums))

        # Define offsets for annotation positions
        offsets_cusum = np.array([-0.05, 1])  # adjust as needed
        offsets_ewma = np.array([0.05, -1])  # adjust as needed

        # Define colors
        color_cusum = 'royalblue'
        color_ewma = 'tomato'

        # Plot for ARL0 of the best models for different gap sizes
        plt.figure(figsize=(10, 8))
        # CUSUM ARL0
        plt.plot(x_arl0, best_arl0_cusums[('ARL0', 'mean')], 'o-', color=color_cusum, label='CUSUM')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl0_cusums['Model (Parameters)']):
            plt.annotate(txt, (x_arl0[i], best_arl0_cusums[('ARL0', 'mean')].iloc[i]) + offsets_cusum, 
                        fontsize=10, color=color_cusum)
        # EWMA ARL0
        plt.plot(x_arl0, best_arl0_ewmas[('ARL0', 'mean')], 'x-', color=color_ewma, label='EWMA')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl0_ewmas['Model (Parameters)']):
            plt.annotate(txt, (x_arl0[i], best_arl0_ewmas[('ARL0', 'mean')].iloc[i]) + offsets_ewma, 
                        fontsize=10, color=color_ewma)
        if self.outlier_position is None:
            plt.title('Mean $ARL_0$ values of best CUSUM/EWMA model for different Gap Sizes')
        else:
            plt.title(f'Mean $ARL_0$ values of best CUSUM/EWMA model for different Gap Sizes with outliers in {self.outlier_position} period')
        plt.xlabel('Gap Size')
        plt.ylabel('$ARL_0$ mean')
        plt.xticks(x_arl0, best_arl0_cusums['Gap Size'])
        plt.legend()
        plt.show()

        # Plot for ARL1 of the best models for different gap sizes
        plt.figure(figsize=(10, 8))
        # CUSUM ARL1
        plt.plot(x_arl1, best_arl1_cusums[('ARL1', 'mean')], 'o-', color=color_cusum, label='CUSUM')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl1_cusums['Model (Parameters)']):
            plt.annotate(txt, (x_arl1[i], best_arl1_cusums[('ARL1', 'mean')].iloc[i]) + offsets_cusum, 
                        fontsize=10, color=color_cusum)
        # EWMA ARL1
        plt.plot(x_arl1, best_arl1_ewmas[('ARL1', 'mean')], 'x-', color=color_ewma, label='EWMA')
        # Annotate with model parameters
        for i, txt in enumerate(best_arl1_ewmas['Model (Parameters)']):
            plt.annotate(txt, (x_arl1[i], best_arl1_ewmas[('ARL1', 'mean')].iloc[i]) + offsets_ewma, 
                        fontsize=10, color=color_ewma)
        if self.outlier_position is None:
            plt.title('Mean $ARL_0$ values of best CUSUM/EWMA model for different Gap Sizes')
        else:
            plt.title(f'Mean $ARL_0$ values of best CUSUM/EWMA model for different Gap Sizes with outliers in {self.outlier_position} period')
        plt.xlabel('Gap Size')
        plt.ylabel('$ARL_1$ mean')
        plt.xticks(x_arl1, best_arl1_cusums['Gap Size'])
        plt.legend()
        plt.show()

# ------------------Testing function-------------------
# For displaying the full pd.df
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# n_sam_bef_cp = 400
# n_sam_aft_cp = 500
# gap_sizes = [1, 5, 10]
# variances = [1, 4, 9]
# seeds = [111, 666, 999]
# BURNIN = 50
# cusum_params_list = [(1.50, 1.61), (1.25, 1.99), (1.00, 2.52), (0.75, 3.34), (0.50, 4.77), (0.25, 8.01)]
# ewma_params_list = [(1.00,3.090),(0.75,3.087),(0.50,3.071),(0.40,3.054),(0.30,3.023),(0.25,2.998),(0.20,2.962),(0.10,2.814),(0.05,2.615),(0.03,2.437)]
# outlier_position = 'in-control'
# alpha = 1e-5
# outlier_ratio = 0.01
# # Without outliers
# grideval = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, 
#                             seeds, BURNIN, cusum_params_list, ewma_params_list, None)
# per_table, per_summary = grideval.grid_params_eval()
# per_summary
# grideval.plot_ARL0_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
# grideval.plot_ARL1_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
# grideval.plot_best_models()
# # For data with outliers
# # simulate_data_list = simulate_grid_data(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, SEED)
# grideval = GridDataEvaluate(n_sam_bef_cp, n_sam_aft_cp, gap_sizes, variances, seeds, BURNIN,
#                              cusum_params_list, ewma_params_list, outlier_position, alpha, outlier_ratio)
# outlier_grid_data = grideval.generate_with_outliers_grid_data(111)
# outlier_grid_data[0][0].shape
# per_table, per_summary = grideval.grid_params_eval()
# per_table.iloc[50]
# grideval.plot_ARL0_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
# grideval.plot_ARL1_graphs(each_G=True, all_CUSUM=True, all_EWMA=True, each_G_V=True)
# grideval.plot_best_models()
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
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Stream Data with Changepoints')
    plt.legend()
    plt.show()

# -------------------streaming data simulation with plot test code--------------------

# data_stream, tau_list, mu_list, size_list = simulate_stream_data(v=50, G=50, D=50, M=50, S=[0.25, 0.5, 1, 3], sigma=1, seed=666)
# stream_data_plot(data_stream, tau_list)
# tau_list.shape
# size_list.sum()
# data_stream.shape
# ------------------End-------------------
