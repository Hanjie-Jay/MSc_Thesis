import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class OutlierInjector:
    """
    This class represents an Outlier Injector that is responsible for injecting outliers into
    the dataset in to the place specified by outlier_position.

    Attributes:
        data (np.ndarray): The input original data to inject outliers.
        n_sam_bef_cp (int): The number of samples before the change point, it is also the true change point position.
        n_sam_aft_cp (int): The number of samples after the change point.
        burnin (int): The number of initial observations to estimate the mean and variance of dataset.
        in_control_var (float): The variance for in-control period of the streaming dataset.
        out_control_mean (float): The mean for out-of-control period of the streaming dataset.
        out_control_var (float): The variance for out-of-control period of the streaming dataset.
        alpha (float): The threshold probability of occurrence for the outliers.
        outlier_position (str): The position to insert outliers ('in-control', 'out-of-control', 'both_in_and_out', 'burn-in').
        outlier_ratio (float, optional): The ratio of data points of the given outlier position period to be considered outliers (default is 0.01, should be between (0,1)).
        in_control_mean (float, optional): The mean for in-control period of the streaming dataset(default is 0).

    Methods:
        calculate_thresholds():
            Calculates the thresholds for outlier insertion for both in-control and out-of-control period.
        add_outliers(num_outliers, indices, lower_threshold, upper_threshold):
            Adds a specific number of outliers at random positions within given indices for in-control or
            out-of-control period.
        add_outliers_for_both(num_outliers, indices, ic_lower_threshold, ic_upper_threshold, oc_lower_threshold, oc_upper_threshold):
            Adds a specific number of outliers at random positions within given indices for both in-control and
            out-of-control period.
        insert_outliers():
            Inserts the outliers into the data at the specified positions.
    """
    def __init__(self, data:np.ndarray, n_sam_bef_cp:int, n_sam_aft_cp:int, burnin:int, in_control_var:float, 
                 out_control_mean:float, out_control_var:float, alpha:float, outlier_position:str, 
                 outlier_ratio:float=0.01, in_control_mean:float=0):
        # Define valid options
        valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
        assert isinstance(data, np.ndarray), "Data should be a numpy array."
        assert isinstance(n_sam_bef_cp, int), f"n_sam_bef_cp:{n_sam_bef_cp} should be an integer."
        assert isinstance(n_sam_aft_cp, int), f"n_sam_aft_cp:{n_sam_aft_cp} should be an integer."
        assert isinstance(burnin, int), f"burnin:{burnin} should be an integer."
        assert burnin < n_sam_bef_cp, f"Value of burnin:{burnin} should smaller than n_sam_bef_cp:{n_sam_bef_cp}"
        assert isinstance(in_control_var, (int, float)) and in_control_var >= 0, f"{in_control_var} should be a non-negative number (int or float)."
        assert isinstance(out_control_mean, (int, float)), f"{out_control_mean} should be a number (int or float)."
        assert isinstance(out_control_var, (int, float)) and out_control_var >= 0, f"{out_control_var} should be a non-negative number (int or float)."
        assert isinstance(alpha, float) and 0 <= alpha <= 1, f"{alpha} should be a float between (0,1)."
        assert isinstance(outlier_ratio, float) and 0 < outlier_ratio < 1, f"{outlier_ratio} should be a float between (0,1)."
        assert isinstance(in_control_mean, (int, float)), f"{in_control_mean} should be a number (int or float)."
        # Check user-provided input for outlier position
        if outlier_position is not None:
            if isinstance(outlier_position, str):
                if outlier_position not in valid_positions:
                    raise ValueError(f"Invalid outlier position. Options are: {valid_positions}")
            else:
                raise TypeError("outlier_position should be only one of the valid string.")
        else:
            raise ValueError("You must provide an outlier_position to add")
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
        self.outlier_position = outlier_position
        self.outlier_indices = []

    def calculate_thresholds(self):
        """
        Calculate the in-control and out-of-control thresholds using the provided alpha level and mean and standard deviations.

        Returns:
        tuple: A tuple containing in-control and out-of-control lower and upper thresholds.
        """
        in_control_lower_threshold = norm.ppf(self.alpha/2, loc=self.in_control_mean, scale=self.in_control_std)
        in_control_upper_threshold = norm.ppf(1 - self.alpha/2, loc=self.in_control_mean, scale=self.in_control_std)
        out_control_lower_threshold = norm.ppf(self.alpha/2, loc=self.out_control_mean, scale=self.out_control_std)
        out_control_upper_threshold = norm.ppf(1 - self.alpha/2, loc=self.out_control_mean, scale=self.out_control_std)
        return in_control_lower_threshold, in_control_upper_threshold, out_control_lower_threshold, out_control_upper_threshold

    def add_outliers(self, num_outliers:int, indices:np.ndarray, lower_threshold:float, upper_threshold:float):
        """
        Helper function to add a specified number of outliers into random positions within the provided indices. The outliers
        are drawn from a uniform distribution that is positively skewed based on the lower and upper thresholds.

        Parameters:
        num_outliers (int): The number of outliers to insert.
        indices (ndarray): The array of indices where outliers can be inserted.
        lower_threshold (float): The lower threshold from which outliers will be drawn.
        upper_threshold (float): The upper threshold from which outliers will be drawn.
        """
        assert len(indices) >= num_outliers, f"Can't sample more outliers ({num_outliers}) than available indices {indices}."
        assert len(indices) > 0, f"Indices f{indices} array can't be empty."
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

    def add_outliers_for_both(self, num_outliers:int, indices:np.ndarray, ic_lower_threshold:float, 
                              ic_upper_threshold:float, oc_lower_threshold:float, oc_upper_threshold:float):
        """
        Helper function to add a specified number of outliers into random positions within the provided indices for both 
        in-control and out-of-control periods. The outliers are drawn from a uniform distribution that is positively 
        skewed based on the in-control and out-of-control lower and upper thresholds.

        Parameters:
        num_outliers (int): The number of outliers to insert.
        indices (ndarray): The array of indices where outliers can be inserted.
        ic_lower_threshold (float): The lower threshold for in-control period from which outliers will be drawn.
        ic_upper_threshold (float): The upper threshold for in-control period from which outliers will be drawn.
        oc_lower_threshold (float): The lower threshold for out-of-control period from which outliers will be drawn.
        oc_upper_threshold (float): The upper threshold for out-of-control period from which outliers will be drawn.
        """
        assert len(indices) >= num_outliers, "Can't sample more outliers than available indices."
        assert len(indices) > 0, "Indices array can't be empty."
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
        
    def insert_outliers(self):
        """
        Main function to insert outliers into the data at specified positions. The outliers are inserted based on 
        the outlier_position attribute specified during object initialisation.

        Returns:
        ndarray: The input data array with outliers inserted.
        """
        in_control_lower_threshold, in_control_upper_threshold, out_control_lower_threshold, out_control_upper_threshold = self.calculate_thresholds()
        if self.outlier_position == 'in-control':
            # specify indices for 'in-control' period
            in_control_indices = np.arange(self.burnin, self.n_sam_bef_cp)
            num_outliers = int(self.outlier_ratio * (self.n_sam_bef_cp - self.burnin))
            self.add_outliers(num_outliers, in_control_indices, in_control_lower_threshold, in_control_upper_threshold)
        
        elif self.outlier_position == 'out-of-control':
            # specify indices for 'out-of-control' period
            out_of_control_indices = np.arange(self.n_sam_bef_cp, self.n_sam_bef_cp + self.n_sam_aft_cp)
            num_outliers = int(self.outlier_ratio * self.n_sam_aft_cp)
            self.add_outliers(num_outliers, out_of_control_indices, out_control_lower_threshold, out_control_upper_threshold)

        elif self.outlier_position == 'burn-in':
            # specify indices for 'burn-in' period
            burnin_indices = np.arange(self.burnin)
            num_outliers = int(self.outlier_ratio * self.burnin)
            self.add_outliers(num_outliers, burnin_indices, in_control_lower_threshold, in_control_upper_threshold)
        
        elif self.outlier_position == 'both_in_and_out':
            # specify indices for 'both_in_and_out' period
            both_in_and_out_indices = np.arange(self.burnin, self.n_sam_bef_cp + self.n_sam_aft_cp)
            num_outliers = int(self.outlier_ratio * (self.n_sam_bef_cp + self.n_sam_aft_cp - self.burnin))
            self.add_outliers_for_both(num_outliers, both_in_and_out_indices, in_control_lower_threshold, 
                                       in_control_upper_threshold, out_control_lower_threshold, out_control_upper_threshold)
        self.num_outliers = num_outliers
        return self.data
    
    def plot_data(self):
        """
        Plotting function for visualising the original data and data with outliers
        """
        out_data = np.copy(self.data)  # Copy of data with outliers
        original_data = np.copy(self.data)  # Copy of original data
        plt.figure(figsize=(12, 6))
        plt.plot(out_data, color='gold', label='Data with Outliers')
        plt.plot(original_data, color="royalblue", label='Original Data')
        plt.axvline(x=self.burnin, color='gray', linestyle=':', label="End of burn-in")
        if self.in_control_mean != self.out_control_mean:
            plt.axvline(x=self.n_sam_bef_cp, color='firebrick', linestyle='--', label="True CP")
        plt.scatter(self.outlier_indices, out_data[self.outlier_indices], color='violet', zorder=3, label='Outliers')
        plt.title(f'Comparison between Original Data and Data with Outliers in {self.outlier_position} period')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

# -------------------testing for outliers class--------------------
# n_sam_bef_cp = 500
# n_sam_aft_cp = 400
# variance = 4
# burnin = 50
# gap_size = 5
# alpha = 0.001
# valid_positions = ['in-control', 'out-of-control', 'both_in_and_out', 'burn-in']
# outlier_position = valid_positions[2]
# outlier_ratio = 0.05
# data_1 = np.append(np.random.normal(size=n_sam_bef_cp, scale=np.sqrt(variance)), 
#                        np.random.normal(size=n_sam_aft_cp,loc=gap_size, scale=np.sqrt(variance)))
# outinj = OutlierInjector(data_1.copy() ,n_sam_bef_cp, n_sam_aft_cp, burnin, variance, 
#                          gap_size, variance, alpha, outlier_position, outlier_ratio)
# out_data = outinj.insert_outliers()
# outinj.outlier_indices
# outinj.plot_data()
# outinj.num_outliers
# ------------------End-------------------