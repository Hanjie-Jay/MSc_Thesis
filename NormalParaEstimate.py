import numpy as np
import matplotlib.pyplot as plt

class Normal_Mean_Var_Estimator:
    """
    This class calculates the estimated mean and variance of streaming data. 
    It assumes that the data follows a normal distribution.

    Attributes:
        x (np.ndarray): The input streaming data. It's a fixed dataset, but the values are coming in a stream.
        n (int): The number of observations in the data.

    Methods:
        sliding_window_mean(omega: int) -> np.ndarray:
            Calculates the sliding window mean of the data.
        ewma_mean(rho: float) -> np.ndarray:
            Calculates the exponentially weighted moving averages of the data.
        forgetting_factor_mean(lam: float) -> np.ndarray:
            Calculates the forgetting factor mean of the data.
        forgetting_factor_var(lam: float) -> np.ndarray:
            Calculates the forgetting factor variance of the data.
        var_with_unknown_mean(ddof: int) -> float:
            Calculates the variance of the data using the sample mean with a degrees of freedom adjustment.
        var_with_known_mean(mean: float) -> float:
            Calculates the variance of the data using a known mean.
        sample_mean() -> float:
            Calculates the mean of the data.
    
    Note: The above methods is only suitable for one dimensional streaming dataset
    """

    def __init__(self, x: np.ndarray):
        """
        Initialize the ControlChart class with input data.

        Parameters:
        x (np.array): Input streaming data (fixed data, but coming in a stream).
        """
        assert isinstance(x, np.ndarray), f"Input={x} must be a numpy array"
        self.x = x
        self.n = len(x) # number of observation

    def sliding_window_mean(self, omega:int):
        """
        Calculates the sliding window mean of the data.

        Parameters:
            omega (int): The size of the observation window. It must be a positive integer.

        Returns:
            np.ndarray: An array of the calculated sliding window means.

        Raises:
            AssertionError: If omega is not a positive integer.

        Note:
            See Big Data textbook C9.1 P149 for more information on this algorithm.
        """
        assert isinstance(omega, int) and omega > 0, f"omega={omega} must be a positive integer"
        ## Pre-allocated array for the mean
        sli_mean = np.zeros(self.x.shape[0])
        sli_mean[0] = self.x[0]
        ## Loop over the observations
        for i, obs in enumerate(self.x):
            if i >= omega:
                ## If the index is larger than omega, use the recursive definiton
                sli_mean[i] = ((omega * sli_mean[i-1]) + self.x[i] - self.x[i-omega]) / omega
            elif i > 0:
                ## Calculate the standard mean if i < omega
                sli_mean[i] = (i * sli_mean[i-1] + obs) / (i+1)
            ## Return sliding window mean
        return sli_mean

    def ewma_mean(self, rho):
        """
        Calculates the exponentially weighted moving averages (EWMA) of the data.

        Parameters:
            rho (float): The weight given to the last measurement. It must be a float in the range (0, 1).

        Returns:
            np.ndarray: An array of the calculated EWMA mean.

        Raises:
            AssertionError: If rho is not a float in the range (0, 1).

        Note:
            See Big Data textbook C9.1.2 P150 for more information on this algorithm.
        """
        assert isinstance(rho, float) and 0 < rho < 1, f"rho={rho} must be a float in the range (0, 1)"
        ## Pre-allocated array for the mean
        ewma_mean = np.zeros(self.x.shape[0])
        ## Loop over the observations
        for i, _ in enumerate(self.x):
            if i == 0:
                ewma_mean[i] = self.x[i]
            else:
                ewma_mean[i] = (1-rho) * ewma_mean[i-1] + rho * self.x[i]
        ## Return EWMA
        return ewma_mean

    def forgetting_factor_mean(self, lam):
        """
        Calculates the forgetting factor mean of the streaming data.

        Parameters:
            lam (float): The forgetting factor value. It must be a float or an int in the range [0, 1].

        Returns:
            np.ndarray: An array of the calculated forgetting factor means.

        Raises:
            AssertionError: If lam is not a float or an int in the range [0, 1].

        Note:
            See Big Data textbook C9.2.1 P155 for more information on this algorithm.
        """
        assert isinstance(lam, (int, float)) and 0 <= lam <= 1, f"lamda={lam} must be a float or int in the range [0, 1]"
        ## Vectors for sums, weights and means with initialisation
        sum = np.zeros(self.n)
        sum[0] = self.x[0]
        weight = np.zeros(self.n)
        weight[0] = 1
        mean = np.zeros(self.n)
        mean[0] = sum[0] / weight[0]
        ## Update sums, weights and means
        for i in range(1,self.n):
            sum[i] = lam * sum[i-1] + self.x[i]
            weight[i] = lam * weight[i-1] + 1
            mean[i] = sum[i] / weight[i]
        ## Return final mean
        return mean

    def forgetting_factor_var(self, lam):
        """
        Calculates the forgetting factor variance of the data.

        Parameters:
            lam (float): The forgetting factor value. It must be a float in the range (0, 1).

        Returns:
            np.ndarray: An array of the calculated forgetting factor variances.

        Raises:
            AssertionError: If lam is not a float in the range (0, 1).

        Note:
            See Big Data textbook C9.2.2 P156 for more information on this algorithm.
        """
        assert isinstance(lam, float) and 0 < lam < 1, f"lamda={lam} must be a float in the range (0, 1)"
        ## Initialise quantities required in the streaming estimate
        r = np.zeros(self.n)
        r[0] = 0
        w = np.zeros(self.n)
        w[0] = 1
        w2 = np.zeros(self.n)
        w2[0] = w[0]
        u = np.zeros(self.n)
        u[0] = 1 ## (w[0]**2 - w2[0]) / w[0]
        v2 = np.zeros(self.n)
        v2[0] = r[0] / u[0]
        s = np.zeros(self.n)
        s[0] = self.x[0]
        xbar = np.zeros(self.n)
        xbar[0] = s[0] / w[0]
        ## Loop for all observations
        for i in range(1,self.n):
            ## Update quantities
            s[i] = lam * s[i-1] + self.x[i]
            w[i] = lam * w[i-1] + 1
            w2[i] = (lam ** 2) * w2[i-1] + 1
            u[i] = (w[i]**2 - w2[i]) / w[i]
            r[i] = lam * r[i-1] + (w[i] - 1) / w[i] * (self.x[i] - xbar[i-1])**2
            xbar[i] = s[i] / w[i]
            v2[i] = r[i] / u[i]
        ## Return FF estimate of the variance
        return v2

    def var_with_unknown_mean(self, ddof:int=1):
        """
        Calculate the unbiased estimate value of variance of data assuming the mean is unknown.
        
        Parameters:
        ddof (int, optional): Delta degrees of freedom for variance calculation. Default is 1.
        
        Returns:
        float: Variance of the data with the sample mean.
        """
        assert isinstance(ddof, int) and ddof > 0, f"ddof={ddof} must be a positive int"
        est_mean = np.mean(self.x)
        var = np.sum((self.x - est_mean)**2) / (self.n - ddof)
        return var

    def var_with_known_mean(self, mean:float):
        """
        Calculate the population variance of data assuming the mean is known.

        Parameters:
        mean (float): The known mean of the data.

        Returns:
        float: Variance of the data with the known mean.
        """
        assert isinstance(mean, (float, int)), f"mean={mean} must be a float or int"
        var = np.sum((self.x - mean)**2) / self.n
        return var

    def sample_mean(self):
        """
        Calculate the mean of the data.

        Returns:
        float: Mean of the data.
        """
        return np.mean(self.x)


# Test for the functions in Normal_Mean_Var_Estimator

# np.random.seed(111)
# data = np.append(np.random.normal(size=100), np.random.normal(size=200, loc=5))
# estimator = Normal_Mean_Var_Estimator(data)
# # sliding_window_mean
# sliding_window_mean2 = estimator.sliding_window_mean(omega=2)
# sliding_window_mean5 = estimator.sliding_window_mean(omega=5)
# sliding_window_mean10 = estimator.sliding_window_mean(omega=10)
# sliding_window_mean20 = estimator.sliding_window_mean(omega=20)
# sliding_window_mean50 = estimator.sliding_window_mean(omega=50)
# plt.figure(figsize=(10, 6))
# plt.plot(data, label="data")
# plt.plot(sliding_window_mean2, label='swm_2')
# plt.plot(sliding_window_mean5, label='swm_5')
# plt.plot(sliding_window_mean10, label='swm_10')
# plt.plot(sliding_window_mean20, label='swm_20')
# plt.plot(sliding_window_mean50, label='swm_50')
# plt.title("sliding window mean")
# plt.legend()

# # ewma_mean
# ewma_mean01 = estimator.ewma_mean(rho=0.01)
# ewma_mean05 = estimator.ewma_mean(rho=0.05)
# ewma_mean10 = estimator.ewma_mean(rho=0.10)
# ewma_mean15 = estimator.ewma_mean(rho=0.15)
# ewma_mean25 = estimator.ewma_mean(rho=0.25)
# plt.figure(figsize=(10, 6))
# plt.plot(data, label="data")
# plt.plot(ewma_mean01, label='ewma_01')
# plt.plot(ewma_mean05, label='ewma_05')
# plt.plot(ewma_mean10, label='ewma_10')
# plt.plot(ewma_mean15, label='ewma_15')
# plt.plot(ewma_mean25, label='ewma_25')
# plt.title("EWMA mean")
# plt.legend()

# # forgetting_factor_mean
# forgetting_factor_mean50 = estimator.forgetting_factor_mean(lam=0.50)
# forgetting_factor_mean75 = estimator.forgetting_factor_mean(lam=0.75)
# forgetting_factor_mean85 = estimator.forgetting_factor_mean(lam=0.85)
# forgetting_factor_mean95 = estimator.forgetting_factor_mean(lam=0.95)
# forgetting_factor_mean99 = estimator.forgetting_factor_mean(lam=0.99)
# plt.figure(figsize=(10, 6))
# plt.plot(data, label="data")
# plt.plot(forgetting_factor_mean50, label='ff_50')
# plt.plot(forgetting_factor_mean75, label='ff_75')
# plt.plot(forgetting_factor_mean85, label='ff_85')
# plt.plot(forgetting_factor_mean95, label='ff_95')
# plt.plot(forgetting_factor_mean99, label='ff_99')
# plt.title("forgetting factor mean")
# plt.legend()

# # forgetting_factor_var
# forgetting_factor_var01 = estimator.forgetting_factor_var(lam=0.01)
# forgetting_factor_var75 = estimator.forgetting_factor_var(lam=0.75)
# forgetting_factor_var85 = estimator.forgetting_factor_var(lam=0.85)
# forgetting_factor_var95 = estimator.forgetting_factor_var(lam=0.95)
# forgetting_factor_var99 = estimator.forgetting_factor_var(lam=0.99)
# plt.figure(figsize=(10, 6))
# plt.plot(forgetting_factor_var01, label='ff_v_01')
# plt.plot(forgetting_factor_var75, label='ff_v_75')
# plt.plot(forgetting_factor_var85, label='ff_v_85')
# plt.plot(forgetting_factor_var95, label='ff_v_95')
# plt.plot(forgetting_factor_var99, label='ff_v_99')
# plt.title("forgetting factor variance")
# plt.legend()
