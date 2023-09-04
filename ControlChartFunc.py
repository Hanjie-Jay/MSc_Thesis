import warnings
import numpy as np
from bisect import insort
from scipy.signal import tukey
from scipy.stats import iqr
from scipy.stats import mstats
from collections import deque

class ControlChart:
    """
    This class encapsulates a set of control chart methods used to detect changes in streaming data. These methods include both traditional and robust methods: the Cumulative Sum (CUSUM) control chart, the Exponentially Weighted Moving Average (EWMA) control chart, and robust methods that use a sequence of means calculated with the trimmed mean, winsorized mean, cosine-tapered mean, and median.

    Attributes:
        x (np.ndarray): The input streaming data. It's a fixed dataset, but the values are coming in a stream.
        n (int): The number of observations in the data.

    Methods:
        cumsum_val(k:float, h:float, mu:float, sigma:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Calculates the CUSUM control chart statistic values and both upper and lower alerts for the data.
        cusum_detect(k:float, h:float, mu:float, sigma:float) -> Optional[int]:
            Detects the first change point in the data using the CUSUM method.
        ewma_val(rho:float, k:float, mu:float, sigma:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Calculates the EWMA control chart statistic values and both upper and lower alerts for the data.
        ewma_detect(rho:float, k:float, mu:float, sigma:float) -> Optional[int]:
            Detects the first change point in the data using the EWMA method.
        compute_robust_methods_mean_seq(median_window_length:int, trimmed_ratio:float, winsorized_ratio:float, cosine_ratio:float,
                                        trimmed_window_length:int, winsorized_window_length:int, cosine_window_length:int, burnin_data:np.array=None):
            Computes the sequence of "means" of the data using four robust methods for the corresponding slidingwindow length: sliding window median, trimmed mean, winsorized mean, and cosine-tapered mean. 
        sliding_window_median_CI_val(z_val:float, h_val:float, mu:float, sigma:float):
            Calculates upper and lower statistics and alerts using the sliding window median confidence interval method.
        sliding_window_median_CI_detect(z_val:float, h_val:float, mu:float, sigma:float, burnin:int):
            Detects the first point of change in the data using the sliding window median confidence interval method.
        trimmed_mean_CI_val(z_val:float, h_val:float, mu:float, sigma:float):
            Calculates upper and lower statistics and alerts using the trimmed mean confidence interval method.
        trimmed_mean_CI_detect(z_val:float, h_val:float, mu:float, sigma:float, burnin:int):
            Detects the first point of change in the data using the trimmed mean confidence interval method.
        winsorized_mean_CI_val(z_val:float, h_val:float, mu:float, sigma:float):
            Calculates upper and lower statistics and alerts using the winsorized mean confidence interval method.
        winsorized_mean_CI_detect(z_val:float, h_val:float, mu:float, sigma:float, burnin:int):
            Detects the first point of change in the data using the winsorized mean confidence interval method.
        cosine_tapered_mean_CI_val(z_val:float, h_val:float, mu:float, sigma:float):
            Calculates upper and lower statistics and alerts using the cosine tapered mean confidence interval method.
        cosine_tapered_mean_CI_detect(z_val:float, h_val:float, mu:float, sigma:float, burnin:int):
            Detects the first point of change in the data using the cosine tapered mean confidence interval method.
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

    def cusum_val(self, k:float, h:float, mu:float, sigma:float):
        """
        Function for calculating both upper and lower statistics and alert value in a CUSUM control chart.
        
        Parameters:
        x (np.array): input streaming data (fixed data, but coming in a stream).
        k (float): reference value or slack value. It should be in the range (0, inf).
        h (float): decision threshold for the alert. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.
        
        Returns:
        tuple: A tuple containing four np.arrays. 
        s_plus: CUSUM control chart value for upper alert.
        s_minus: CUSUM control chart value for lower alert.
        au: array indicating whether an upper alert is triggered at each point.
        al: array indicating whether a lower alert is triggered at each point.

        Note:
        See Big Data textbook C9.3.3 P160 for more information on this algorithm.
        """
        assert isinstance(k, (int, float)) and k > 0, f"k={k} must be a positive number"
        assert isinstance(h, (int, float)) and h > 0, f"h={h} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        ## Vectors for control chart and alerts
        s_plus = np.zeros(self.n) # control chart for detecting increase
        s_minus = np.zeros(self.n) # control chart for detecting decrease
        au = np.zeros(self.n) # upper alert
        al = np.zeros(self.n) # lower alert
        ## Initialize control chart
        # s_plus[0] = np.max([0, self.x[0] - k * mu])
        # s_minus[0] = np.min([0, self.x[0] - k * mu])
        s_plus[0] = mu # another def, Dean
        s_minus[0] = mu # another def, Dean
        ## Update control chart and alerts
        for i in range(1,self.n):
            # s_plus[i] = np.max([0, (1-au[i-1]) * s_plus[i-1] + self.x[i] - k * mu]) # calculate upper CUSUM
            # s_minus[i] = np.min([0, (1-al[i-1]) * s_minus[i-1] + self.x[i] - k * mu]) # calculate lower CUSUM
            # au[i] = int(s_plus[i] > (h * sigma)) # detect increase
            # al[i] = int(s_minus[i] < -(h * sigma)) # detect decrease
            # Another definition, Dean 2016
            s_plus[i] = np.max([0, s_plus[i-1] + (self.x[i] - mu) / sigma - k]) # calculate upper CUSUM
            s_minus[i] = np.max([0, s_minus[i-1] - (self.x[i] - mu) / sigma - k]) # calculate lower CUSUM
            au[i] = int(s_plus[i] > h) # detect increase
            al[i] = int(s_minus[i] > h) # detect decrease
        ## Return final chart and alerts
        return s_plus, s_minus, au, al

    def __check_alerts(self, au: np.array, al: np.array):
        """
        Check for the first occurrence of an alert in the given alert arrays.
        
        Parameters:
        au (np.array): Array of upper alerts.
        al (np.array): Array of lower alerts.
        
        Returns:
        int or None: The index of the first detected alert, or None if no alert is detected.
        """
        upper_alert_ind = np.where(au > 0)[0]
        lower_alert_ind = np.where(al > 0)[0]
        # check for upper alert
        if len(upper_alert_ind) > 0:
            first_upper_alert = upper_alert_ind[0]
        else:
            first_upper_alert = None
        # check for lower alert
        if len(lower_alert_ind) > 0:
            first_lower_alert = lower_alert_ind[0]
        else:
            first_lower_alert = None
        # return the alert
        if first_upper_alert is not None and first_lower_alert is not None:
            return min(first_upper_alert, first_lower_alert)
        elif first_upper_alert is not None:
            return first_upper_alert
        elif first_lower_alert is not None:
            return first_lower_alert
        else:
            return None

    def cusum_detect(self, k:float, h:float, mu:float, sigma:float, burnin:int):
        """
        Detects the first point of change in the data using the CUSUM method with burn-in added.

        Parameters:
        k (float or int): Reference value or slack value. It should be a positive number.
        h (float or int): Decision threshold for the alert. It should be a positive number.
        mu (float or int): The (estimated) mean of the data.
        sigma (float or int): The (estimated) standard deviation of the data. It should be a non-negative number.
        burnin (int): The number of examples used for burn-in to estimate the mean and variance.

        Returns:
        int or None: The index of the first detected alert, or None if no alert is detected.

        Raises:
        AssertionError: If any of the input parameters are not of the expected type or violate the constraints.
        """
        assert isinstance(k, (int, float)) and k > 0, f"k={k} must be a positive number"
        assert isinstance(h, (int, float)) and h > 0, f"h={h} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        assert isinstance(burnin, int) and burnin >= 0, f"burnin={burnin} should be a non-negative integer"
        # compute the statistics value for cumsum
        _, _, au, al = self.cusum_val(k, h, mu, sigma)
        # return the detected the alert
        ind = self.__check_alerts(au, al) 
        if ind is not None:
            ind += burnin 
        else:
            warnings.warn("No change point detected, detected index set to None")
        return ind

    def ewma_val(self, rho:float, k:float, mu:float, sigma:float):
        """
        Calculate both upper and lower statistics, variance and alert in an Exponential Weighted Moving Average (EWMA) control chart.
        
        Parameters:
        rho (float): The control parameter for the weight of the current input. It should be in the range [0, 1].
        k (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.

        Returns:
        tuple: A tuple containing four np.arrays. 
        The first array is the calculated stream of statistic, the second is the variance of statistic,
        the third is the alerts for increase, and the fourth is the alerts for decrease.
        
        Note:
        See Big Data textbook C9.3.5 P162 for more information on this algorithm.   
        """
        assert isinstance(rho, float) and 0 <= rho <= 1, f"rho={rho} must be a float in the range [0, 1]"
        assert isinstance(k, (int, float)) and k > 0, f"k={k} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        ## Vectors for control chart, alerts and variance
        s = np.zeros(self.n)
        v = np.zeros(self.n)
        au = np.zeros(self.n)
        al = np.zeros(self.n)
        s[0] = mu
        v[0] = (rho * sigma) ** 2 
        au[0] = int(s[0] > (mu + k * v[0]))
        al[0] = int(s[0] < (mu - k * v[0]))
        ## Update control chart, alerts and variance
        for i in range(1,self.n):
            s[i] = (1-rho) * s[i-1] + rho * self.x[i]
            v[i] = rho / (2-rho) * (1 - (1-rho) ** (2*i)) * (sigma ** 2)
            au[i] = int(s[i] > (mu + k * np.sqrt(v[i]))) # detect increase
            al[i] = int(s[i] < (mu - k * np.sqrt(v[i]))) # detect decrease
        ## Return final chart, variance and alerts
        return s, v, au, al
    
    def ewma_detect(self, rho:float, k:float, mu:float, sigma:float, burnin:int):
        """
        Detects the first point of change in the data using the EWMA method.

        Parameters:
        rho (float): The control parameter for the weight of the current input. It should be in the range (0, 1).
        k (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.
        burnin (int): The number of examples used for burn-in to estimate the mean and variance.

        Returns:
        int or None: The index of the first detected alert, or None if no alert is detected.

        Raises:
        AssertionError: If any of the input parameters are not of the expected type or violate the constraints.
        """
        assert isinstance(rho, float) and 0 < rho < 1, f"rho={rho} must be a float in the range (0, 1)"
        assert isinstance(k, (int, float)) and k > 0, f"k={k} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        assert isinstance(burnin, int) and burnin >= 0, f"burnin={burnin} should be a non-negative integer"
        # compute the statistics value for cumsum
        _, _, au, al = self.ewma_val(rho, k, mu, sigma)
        # return the detected the alert
        ind = self.__check_alerts(au, al) 
        if ind is not None:
            ind += burnin 
        else:
            warnings.warn("No change point detected, detected index set to None")
        return ind
    
    def compute_robust_methods_mean_seq(self, median_window_length:int, trimmed_ratio:float, winsorized_ratio:float, cosine_ratio:float,
                                        trimmed_window_length:int, winsorized_window_length:int, cosine_window_length:int, burnin_data:np.array=None):
        """
        This function computes the sequence of means of the data using three robust methods: trimmed mean, winsorized mean, and cosine-tapered mean. 
        It then stores these sequences in the instance variables `trimmed_mean`, `winsorized_mean`, and `cosine_tapered_mean`, respectively. 
        It also computes the sliding window median of the data and stores it in the instance variable `sliding_window_median`.

        Parameters:
        median_window_length (int): The length of the sliding window for computing the median. Must be less than or equal to the number of observations.
        trimmed_ratio (float): The proportion of values to trim from both ends for the trimmed mean calculation. Must be in the range [0,1].
        winsorized_ratio (float): The proportion of values to replace from both ends for the winsorized mean calculation. Must be in the range [0,1].
        cosine_ratio (float): The proportion of values to taper for the cosine-tapered mean calculation. Must be in the range [0,1].
        trimmed_window_length (int): The number of recent values to consider for the trimmed mean calculation.
        winsorized_window_length (int): The number of recent values to consider for the winsorized mean calculation.
        cosine_window_length (int): The number of recent values to consider for the cosine tapered mean calculation.
        burnin_data (np.array): Initial set of observations for estimating the parameters. Default to None

        Note:
        This function should be called before any function that uses the instance variables `trimmed_mean`, `winsorized_mean`, `cosine_tapered_mean`, or `sliding_window_median`.

        Raises:
        AssertionError: If the input parameters are not in the expected formats or ranges.
        """
        robust_method = RobustMethods(self.x.copy(), burnin_data)
        if burnin_data is not None:
            self._burnin = len(burnin_data)
        else:
            self._burnin = 1
        if median_window_length is not None:
            assert isinstance(median_window_length, int) and 0 < median_window_length <= self.n, f"Median window length={median_window_length} must be a positive integer less than or equal to the number of observations={self.n}"
            self._sliding_window_median = robust_method.sliding_window_median(window_length=median_window_length)
        robust_mean_seq = robust_method.compute_mean_sequence(trimmed_ratio=trimmed_ratio, winsorized_ratio=winsorized_ratio, cosine_ratio=cosine_ratio,
                                                                trimmed_window_length=trimmed_window_length, winsorized_window_length=winsorized_window_length,
                                                                cosine_window_length=cosine_window_length)
        if trimmed_ratio is not None:
            assert isinstance(trimmed_ratio, float) and 0 <= trimmed_ratio <= 1, f"trimmed_ratio={trimmed_ratio} must be a float in the range [0, 1]"        
            assert isinstance(trimmed_window_length, int) and trimmed_window_length <= self.n, f"Trimmed window length={trimmed_window_length} must be an integer less than or equal to the number of observations={self.n}"
            self._trimmed_mean = robust_mean_seq['trimmed']
        if winsorized_ratio is not None:
            assert isinstance(winsorized_ratio, float) and 0 <= winsorized_ratio <= 1, f"winsorized_ratio={winsorized_ratio} must be a float in the range [0, 1]"        
            assert isinstance(winsorized_window_length, int) and winsorized_window_length <= self.n, f"Winsorized window length={winsorized_window_length} must be an integer less than or equal to the number of observations={self.n}"
            self._winsorized_mean = robust_mean_seq['winsorized']
        if cosine_ratio is not None:
            assert isinstance(cosine_ratio, float) and 0 <= cosine_ratio <= 1, f"cosine_ratio={cosine_ratio} must be a float in the range [0, 1]"        
            assert isinstance(cosine_window_length, int) and cosine_window_length <= self.n, f"Cosine tapered window length={cosine_window_length} must be an integer less than or equal to the number of observations={self.n}"
            self._cosine_tapered_mean = robust_mean_seq['cosine']

    def sliding_window_median_CI_val(self, z_val:float, h_val:float, mu:float, sigma:float):
        """
        Function for calculating both upper and lower statistics and alert value using the sliding window median confidence interval method.
        
        Parameters:
        z_val (float): The control parameter for deciding the width of confidence interval. It should be in the range (0, inf).
        h_val (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.

        Returns:
        tuple: A tuple containing four np.arrays. 
        s: Upper statistics value for upper alert.
        t: Lower statistics value for lower alert.
        au: array indicating whether an upper alert is triggered at each point.
        al: array indicating whether a lower alert is triggered at each point.
        """
        assert isinstance(z_val, (int, float)) and z_val > 0, f"z_val={z_val} must be a positive number"
        assert isinstance(h_val, (int, float)) and h_val > 0, f"h_val={h_val} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        if not hasattr(self, '_sliding_window_median'):
            raise ValueError("Must run compute_robust_methods_mean_seq() before calling sliding_window_median_CI_val()")
        ## Vectors for control chart statistics and alerts
        s = np.zeros(self.n)
        t = np.zeros(self.n)
        au = np.zeros(self.n)
        al = np.zeros(self.n)
        burnin = self._burnin
        s[0] = mu
        t[0] = mu
        au[0] = int(s[0] > mu + h_val)
        al[0] = int(t[0] < mu - h_val)
        ## Update control chart, alerts and variance
        for i in range(1, self.n):
            s[i] = self._sliding_window_median[i-1] + z_val * sigma / np.sqrt(burnin)
            t[i] = self._sliding_window_median[i-1] - z_val * sigma / np.sqrt(burnin)
            au[i] = int(s[i] > (mu + h_val)) # detect increase
            al[i] = int(t[i] < (mu - h_val)) # detect decrease
        ## Return final upper and lower statistics and alerts
        return s, t, au, al

    def sliding_window_median_CI_detect(self, z_val:float, h_val:float, mu:float, sigma:float):
        """
        Detects the first point of change in the data using the sliding window median confidence interval method.

        Parameters:
        z_val (float): The control parameter for deciding the width of confidence interval. It should be in the range (0, inf).
        h_val (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.
        burnin (int): The number of examples used for burn-in to estimate the mean and variance.

        Returns:
        int or None: The index of the first detected alert, or None if no alert is detected.

        Raises:
        AssertionError: If any of the input parameters are not of the expected type or violate the constraints.
        """
        assert isinstance(z_val, (int, float)) and z_val > 0, f"z_val={z_val} must be a positive number"
        assert isinstance(h_val, (int, float)) and h_val > 0, f"h_val={h_val} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        if not hasattr(self, '_sliding_window_median'):
            raise ValueError("Must run compute_robust_methods_mean_seq() before calling sliding_window_median_CI_detect()")
        # compute the statistics value for cumsum
        _, _, au, al = self.sliding_window_median_CI_val(z_val, h_val, mu, sigma)
        # return the detected the alert
        ind = self.__check_alerts(au, al) 
        burnin = self._burnin
        if ind is not None:
            ind += burnin 
        else:
            warnings.warn("No change point detected, detected index set to None")
        return ind
    
    def trimmed_mean_CI_val(self, z_val:float, h_val:float, mu:float, sigma:float):
        """
        Function for calculating both upper and lower statistics and alert value using the trimmed mean confidence interval method.
        
        Parameters:
        z_val (float): The control parameter for deciding the width of confidence interval. It should be in the range (0, inf).
        h_val (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.
        
        Returns:
        tuple: A tuple containing four np.arrays. 
        s: Upper statistics value for upper alert.
        t: Lower statistics value for lower alert.
        au: array indicating whether an upper alert is triggered at each point.
        al: array indicating whether a lower alert is triggered at each point.
        """
        assert isinstance(z_val, (int, float)) and z_val > 0, f"z_val={z_val} must be a positive number"
        assert isinstance(h_val, (int, float)) and h_val > 0, f"h_val={h_val} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        if not hasattr(self, '_trimmed_mean'):
            raise ValueError("Must run compute_robust_methods_mean_seq() before calling trimmed_mean_CI_val()")
        ## Vectors for control chart statistics and alerts
        s = np.zeros(self.n)
        t = np.zeros(self.n)
        au = np.zeros(self.n)
        al = np.zeros(self.n)
        burnin = self._burnin
        s[0] = mu
        t[0] = mu
        au[0] = int(s[0] > mu + h_val)
        al[0] = int(t[0] < mu - h_val)
        ## Update control chart, alerts and variance
        for i in range(1, self.n):
            s[i] = self._trimmed_mean[i-1] + z_val * sigma / np.sqrt(burnin)
            t[i] = self._trimmed_mean[i-1] - z_val * sigma / np.sqrt(burnin)
            au[i] = int(s[i] > (mu + h_val)) # detect increase
            al[i] = int(t[i] < (mu - h_val)) # detect decrease
        ## Return final upper and lower statistics and alerts
        return s, t, au, al

    def trimmed_mean_CI_detect(self, z_val:float, h_val:float, mu:float, sigma:float):
        """
        Detects the first point of change in the data using the trimmed mean confidence interval method.

        Parameters:
        z_val (float): The control parameter for deciding the width of confidence interval. It should be in the range (0, inf).
        h_val (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.
        burnin (int): The number of examples used for burn-in to estimate the mean and variance.

        Returns:
        int or None: The index of the first detected alert, or None if no alert is detected.

        Raises:
        AssertionError: If any of the input parameters are not of the expected type or violate the constraints.
        """
        assert isinstance(z_val, (int, float)) and z_val > 0, f"z_val={z_val} must be a positive number"
        assert isinstance(h_val, (int, float)) and h_val > 0, f"h_val={h_val} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        if not hasattr(self, '_trimmed_mean'):
            raise ValueError("Must run compute_robust_methods_mean_seq() before calling trimmed_mean_CI_detect()")
        burnin = self._burnin
        # compute the statistics value for cumsum
        _, _, au, al = self.trimmed_mean_CI_val(z_val, h_val, mu, sigma)
        # return the detected the alert
        ind = self.__check_alerts(au, al) 
        if ind is not None:
            ind += burnin 
        else:
            warnings.warn("No change point detected, detected index set to None")
        return ind
    
    def winsorized_mean_CI_val(self, z_val:float, h_val:float, mu:float, sigma:float):
        """
        Function for calculating both upper and lower statistics and alert value using the winsorized mean confidence interval method.
        
        Parameters:
        z_val (float): The control parameter for deciding the width of confidence interval. It should be in the range (0, inf).
        h_val (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.
        
        Returns:
        tuple: A tuple containing four np.arrays. 
        s: Upper statistics value for upper alert.
        t: Lower statistics value for lower alert.
        au: array indicating whether an upper alert is triggered at each point.
        al: array indicating whether a lower alert is triggered at each point.
        """
        assert isinstance(z_val, (int, float)) and z_val > 0, f"z_val={z_val} must be a positive number"
        assert isinstance(h_val, (int, float)) and h_val > 0, f"h_val={h_val} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        if not hasattr(self, '_winsorized_mean'):
            raise ValueError("Must run compute_robust_methods_mean_seq() before calling winsorized_mean_CI_val()")
        ## Vectors for control chart statistics and alerts
        s = np.zeros(self.n)
        t = np.zeros(self.n)
        au = np.zeros(self.n)
        al = np.zeros(self.n)
        burnin = self._burnin
        s[0] = mu
        t[0] = mu
        au[0] = int(s[0] > mu + h_val)
        al[0] = int(t[0] < mu - h_val)
        ## Update control chart, alerts and variance
        for i in range(1, self.n):
            s[i] = self._winsorized_mean[i-1] + z_val * sigma / np.sqrt(burnin)
            t[i] = self._winsorized_mean[i-1] - z_val * sigma / np.sqrt(burnin)
            au[i] = int(s[i] > (mu + h_val)) # detect increase
            al[i] = int(t[i] < (mu - h_val)) # detect decrease
        ## Return final upper and lower statistics and alerts
        return s, t, au, al

    def winsorized_mean_CI_detect(self, z_val:float, h_val:float, mu:float, sigma:float):
        """
        Detects the first point of change in the data using the winsorized mean confidence interval method.

        Parameters:
        z_val (float): The control parameter for deciding the width of confidence interval. It should be in the range (0, inf).
        h_val (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.

        Returns:
        int or None: The index of the first detected alert, or None if no alert is detected.

        Raises:
        AssertionError: If any of the input parameters are not of the expected type or violate the constraints.
        """
        assert isinstance(z_val, (int, float)) and z_val > 0, f"z_val={z_val} must be a positive number"
        assert isinstance(h_val, (int, float)) and h_val > 0, f"h_val={h_val} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        if not hasattr(self, '_winsorized_mean'):
            raise ValueError("Must run compute_robust_methods_mean_seq() before calling winsorized_mean_CI_detect()")
        burnin = self._burnin
        # compute the statistics value for cumsum
        _, _, au, al = self.winsorized_mean_CI_val(z_val, h_val, mu, sigma)
        # return the detected the alert
        ind = self.__check_alerts(au, al) 
        if ind is not None:
            ind += burnin 
        else:
            warnings.warn("No change point detected, detected index set to None")
        return ind
    
    def cosine_tapered_mean_CI_val(self, z_val:float, h_val:float, mu:float, sigma:float):
        """
        Function for calculating both upper and lower statistics and alert value using the cosine tapered mean confidence interval method.
        
        Parameters:
        z_val (float): The control parameter for deciding the width of confidence interval. It should be in the range (0, inf).
        h_val (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.
        
        Returns:
        tuple: A tuple containing four np.arrays. 
        s: Upper statistics value for upper alert.
        t: Lower statistics value for lower alert.
        au: array indicating whether an upper alert is triggered at each point.
        al: array indicating whether a lower alert is triggered at each point.
        """
        assert isinstance(z_val, (int, float)) and z_val > 0, f"z_val={z_val} must be a positive number"
        assert isinstance(h_val, (int, float)) and h_val > 0, f"h_val={h_val} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        if not hasattr(self, '_cosine_tapered_mean'):
            raise ValueError("Must run compute_robust_methods_mean_seq() before calling cosine_tapered_mean_CI_val()")
        ## Vectors for control chart statistics and alerts
        s = np.zeros(self.n)
        t = np.zeros(self.n)
        au = np.zeros(self.n)
        al = np.zeros(self.n)
        burnin = self._burnin
        s[0] = mu
        t[0] = mu
        au[0] = int(s[0] > mu + h_val)
        al[0] = int(t[0] < mu - h_val)
        ## Update control chart, alerts and variance
        for i in range(1, self.n):
            s[i] = self._cosine_tapered_mean[i-1] + z_val * sigma / np.sqrt(burnin)
            t[i] = self._cosine_tapered_mean[i-1] - z_val * sigma / np.sqrt(burnin)
            au[i] = int(s[i] > (mu + h_val)) # detect increase
            al[i] = int(t[i] < (mu - h_val)) # detect decrease
        ## Return final upper and lower statistics and alerts
        return s, t, au, al

    def cosine_tapered_mean_CI_detect(self, z_val:float, h_val:float, mu:float, sigma:float):
        """
        Detects the first point of change in the data using the cosine tapered mean confidence interval method.

        Parameters:
        z_val (float): The control parameter for deciding the width of confidence interval. It should be in the range (0, inf).
        h_val (float): The control parameter for the alert function. It should be in the range (0, inf).
        mu (float): The (estimated) mean of the data.
        sigma (float): The (estimated) standard deviation of the data.

        Returns:
        int or None: The index of the first detected alert, or None if no alert is detected.

        Raises:
        AssertionError: If any of the input parameters are not of the expected type or violate the constraints.
        """
        assert isinstance(z_val, (int, float)) and z_val > 0, f"z_val={z_val} must be a positive number"
        assert isinstance(h_val, (int, float)) and h_val > 0, f"h_val={h_val} must be a positive number"
        assert isinstance(mu, (int, float)), f"Mean mu={mu} must be an int or float type"
        assert isinstance(sigma, (int, float)) and sigma >= 0, f"standard deviation sigma={sigma} must be a non-negative number"
        if not hasattr(self, '_cosine_tapered_mean'):
            raise ValueError("Must run compute_robust_methods_mean_seq() before calling cosine_tapered_mean_CI_detect()")
        burnin = self._burnin
        # compute the statistics value for cumsum
        _, _, au, al = self.cosine_tapered_mean_CI_val(z_val, h_val, mu, sigma)
        # return the detected the alert
        ind = self.__check_alerts(au, al) 
        if ind is not None:
            ind += burnin 
        else:
            warnings.warn("No change point detected, detected index set to None")
        return ind





class RobustMethods:
    """
    This class provides methods for calculating robust statistics for streaming data. It includes functions to compute the mean and variance sequences using different robust methods: trimmed mean, winsorized mean, cosine-tapered mean, median absolute deviation (MAD), interquartile range (IQR), and winsorized variance.

    Attributes:
        x (np.ndarray): The input streaming data. It's a fixed dataset, but the values are coming in a stream.
        n (int): The number of observations in the data.

    Methods:
        compute_mean_sequence(trimmed_ratio:float, winsorized_ratio:float, cosine_ratio:float, 
                          trimmed_window_length:int, winsorized_window_length:int, cosine_window_length:int) -> dict:
        Computes the mean sequence of the data using three robust methods: trimmed mean, winsorized mean, and cosine tapered mean.

        compute_variance_sequence(winsorized_ratio:float) -> dict:
            Compute the variance sequence of the data using three robust methods: MAD, IQR, and winsorized variance.    

        sliding_window_median(window_length:int) -> list:
            Computes the median of a sliding window over the data.

    Private Methods:
        __insert_next_for_sorted_data() -> None:
            Inserts the next value from the data stream into the sorted list and sorts it.

        __trimmed_mean(ratio:float, window_length:int) -> float:
        Calculates the trimmed mean of the sorted data within the window.

        __winsorized_mean(ratio:float, window_length:int) -> float:
            Calculates the winsorized mean of the sorted data within the window.

        __cosine_tapered_mean(ratio:float, window_length:int) -> float:
            Calculates the cosine tapered mean of the sorted data within the window.
        
        __mad() -> float:
        Compute the Median Absolute Deviation (MAD) of the sorted data.

        __interquartile_range() -> float:
            Compute the Interquartile Range (IQR) of the sorted data.

        __winsorized_variance(ratio:float) -> float:
            Compute the winsorized variance for a specific proportion of values from sorted data.
    """
    def __init__(self, x: np.ndarray, burnin_x:np.ndarray=None):
        """
        Initialize the ControlChart class with input data.

        Parameters:
        x (np.array): Input streaming data (fixed data, but coming in a stream).
        burnin_x (np.array): Initial set of observations for estimating the parameters. Default to None
        """
        assert isinstance(x, np.ndarray), f"Input={x} must be a numpy array"
        self.x = x
        self.n = len(x) # number of observation
        self._current_ind = 0
        if burnin_x is not None:
            assert isinstance(burnin_x, np.ndarray), f"burnin_x={burnin_x} must be a numpy array"
            self._burnin_x = burnin_x
        else:
            self._burnin_x = None

    def sliding_window_median(self, window_length:int):
        """
        Compute the median of a sliding window over the data. (For the whole data stream)

        Parameters:
        window_length (int): The length of the sliding window. Must be smaller than the total number of the streaming data
        
        Returns:
        medians (list): The medians of each window of the whole data stream.
        """
        assert isinstance(window_length, int) and window_length <= self.n, f"Window length={window_length} must be an integer less than or equal to the number of observations={self.n}"
        medians = np.array([])
        for i in range(self.n):
            # Compute the median of the window and add it to the array
            if i < window_length:
                if self._burnin_x is not None:
                    window_data = np.concatenate((self._burnin_x, self.x[:i+1]))
                    window_median = np.median(window_data[-window_length:])
                else:
                    window_median = np.median(self.x[:i+1])
            else:
                window_median = np.median(self.x[i-window_length+1:i+1])
            medians = np.append(medians, window_median)
        return medians
    
    def __insert_next_val_and_sort(self):
        """
        Insert the next value from the data stream into the sorted list and sort it.
        """
        if self._current_ind < self.n:
            # Insert next values from the data stream for all three types
            for win_data_attr, sorted_data_attr in zip(['_window_data_trimmed', '_window_data_winsorized', '_window_data_cosine'],
                                                    ['_sorted_data_trimmed', '_sorted_data_winsorized', '_sorted_data_cosine']):
                if hasattr(self, win_data_attr) and hasattr(self, sorted_data_attr):
                    win_data = getattr(self, win_data_attr)
                    sorted_data = getattr(self, sorted_data_attr)
                    if len(win_data) == win_data.maxlen:
                        old_value = win_data[0]
                        sorted_data.remove(old_value)
                    win_data.append(self.x[self._current_ind])
                    insort(sorted_data, self.x[self._current_ind])
            self._current_ind += 1
        else:
            # Reset all six deques if they exist
            for attr in ['_window_data_trimmed', '_sorted_data_winsorized', '_window_data_cosine',
                        '_sorted_data_trimmed', '_sorted_data_winsorized', '_sorted_data_cosine']:
                if hasattr(self, attr):
                    getattr(self, attr).clear()

    def compute_mean_sequence(self, trimmed_ratio:float, winsorized_ratio:float, cosine_ratio:float, 
                          trimmed_window_length:int, winsorized_window_length:int, cosine_window_length:int):
        """
        This method computes the mean sequence of a given data stream using three robust methods: the trimmed mean, the winsorized mean, 
        and the cosine tapered mean. It does this by initializing deques for each method with the corresponding window length and if there is 
        burn-in data, it populates the deque with it and sorts them. If there's no burn-in data, it initializes empty sorted lists. 
        
        The method then generates a dictionary that contains the sequences, ratios, window lengths and computation functions for each method. 
        The method then iterates through the given data, updating the sorted lists and computing the mean values using the specified methods, 
        which are then added to their respective sequences in the dictionary. 
        
        Finally, the method returns a dictionary with the mean sequences as numpy arrays for each method.
        
        Parameters:
        trimmed_ratio (float): The proportion of values to trim for the trimmed mean calculation.
        winsorized_ratio (float): The proportion of values to replace for the winsorized mean calculation.
        cosine_ratio (float): The proportion of values to taper for the cosine tapered mean calculation.
        trimmed_window_length (int): The number of recent values to consider for the trimmed mean calculation.
        winsorized_window_length (int): The number of recent values to consider for the winsorized mean calculation.
        cosine_window_length (int): The number of recent values to consider for the cosine tapered mean calculation.
        
        Returns:
        robust_mean_seq (dict): A dictionary containing the mean sequences as numpy arrays for each method.
        """
        self._window_data_trimmed = deque(maxlen=trimmed_window_length) if trimmed_window_length else None
        self._window_data_winsorized = deque(maxlen=winsorized_window_length) if winsorized_window_length else None
        self._window_data_cosine = deque(maxlen=cosine_window_length) if cosine_window_length else None
        
        if self._burnin_x is not None:
            if trimmed_ratio is not None:
                self._window_data_trimmed.extend(self._burnin_x[-trimmed_window_length:])
                self._sorted_data_trimmed = sorted(list(self._window_data_trimmed))
            if winsorized_ratio is not None:
                self._window_data_winsorized.extend(self._burnin_x[-winsorized_window_length:])
                self._sorted_data_winsorized = sorted(list(self._window_data_winsorized))
            if cosine_ratio is not None:
                self._window_data_cosine.extend(self._burnin_x[-cosine_window_length:])
                self._sorted_data_cosine = sorted(list(self._window_data_cosine))
        else:
            self._sorted_data_trimmed = []
            self._sorted_data_winsorized = []
            self._sorted_data_cosine = []
        
        mean_seqs = {
            "trimmed": {
                "seq": [],
                "ratio": trimmed_ratio,
                "window_length": trimmed_window_length,
                "compute_fn": self.__trimmed_mean
            } if trimmed_ratio and trimmed_window_length else None,
            "winsorized": {
                "seq": [],
                "ratio": winsorized_ratio,
                "window_length": winsorized_window_length,
                "compute_fn": self.__winsorized_mean
            } if winsorized_ratio and winsorized_window_length else None,
            "cosine": {
                "seq": [],
                "ratio": cosine_ratio,
                "window_length": cosine_window_length,
                "compute_fn": self.__cosine_tapered_mean
            } if cosine_ratio and cosine_window_length else None
        }
        for _ in range(self.n):
            self.__insert_next_val_and_sort()
            # Compute all three types of means if necessary
            for method in mean_seqs.values():
                if method is not None:
                    mean_value = method["compute_fn"](method["ratio"], method["window_length"])
                    method["seq"].append(mean_value)
        robust_mean_seq = {mean_type: np.array(method["seq"]) for mean_type, method in mean_seqs.items() if method is not None}
        return robust_mean_seq

    def compute_variance_sequence(self, winsorized_ratio:float):
        """
        Compute the variance sequence of the data using three robust methods: median absolute deviation (MAD), interquartile range (IQR) and winsorized variance.
        
        Parameters:
        winsorized_ratio (float): The proportion of values (Two sided) to replace for the winsorized mean calculation. It should be in the range [0,1].
        
        Note:
        The return variance sequence of MAD and IQR are raw values, not the unbiased values for normal distribution.
        For the normal distribution, s.d. ~ 1.4826 * mad or iqr / 1.349.

        Returns:
        robust_var_seq (dict): A dictionary containing the var sequences calculated using each of the three robust methods.
        """
        mad_seq = []
        iqr_seq = []
        winsorized_var_seq = []
        for _ in range(self.n):
            self.__insert_next_val_and_sort()
            mad_seq.append(self.__mad())
            iqr_seq.append(self.__interquartile_range())
            winsorized_var_seq.append(self.__winsorized_variance(winsorized_ratio))
        robust_var_seq = {
            "mad": np.array(mad_seq),
            "iqr": np.array(iqr_seq),
            "winsorized": np.array(winsorized_var_seq)
        }
        return robust_var_seq

    def __trimmed_mean(self, ratio:float, window_length:int):
        """
        Calculate the trimmed mean of the sorted data. (For a specific sorted data)

        Parameters:
        ratio (float): The proportion of values to trim. Must be in the range [0,1]. 
        window_length (int): The number of recent values to consider. Must be smaller than the total number of the streaming data
        
        Returns:
        trimmed_mean (float): The trimmed mean of the data.
        """
        assert isinstance(ratio, (float, int)) and 0 <= ratio <= 1, f"ratio={ratio} must be a float or float in the range [0, 1]"
        assert isinstance(window_length, int) and window_length <= self.n, f"Window length={window_length} must be an integer less than or equal to the number of observations={self.n}"
        adjusted_window_length = min(window_length, len(self._sorted_data_trimmed))
        if self._current_ind < window_length:
            # Compute lower and upper cutoff indices
            self._tm_lower_ind = int(np.floor(adjusted_window_length) * ratio)
            self._tm_upper_ind = adjusted_window_length - self._tm_lower_ind
        # Trim the data and compute the mean
        trimmed_data = self._sorted_data_trimmed[self._tm_lower_ind:self._tm_upper_ind] # first select the value in the window, then slice it 
        trimmed_mean = np.mean(trimmed_data)
        return trimmed_mean

    def __winsorized_mean(self, ratio:float, window_length:int):
        """
        Calculate the winsorized mean of the sorted data. (For a specific sorted data)

        Parameters:
        ratio (float): The proportion of values to replace. Must be in the range [0,1].
        window_length (int): The number of recent values to consider. Must be smaller than the total number of the streaming data
        
        Returns:
        winsorized_mean (float): The winsorized mean of the data.
        """
        assert isinstance(ratio, (float, int)) and 0 <= ratio <= 1, f"ratio={ratio} must be a float or float in the range [0, 1]"
        assert isinstance(window_length, int) and window_length <= self.n, f"Window length={window_length} must be an integer less than or equal to the number of observations={self.n}"
        # # Compute lower and upper cutoff indices
        # lower_ind = int(np.floor(window_length) * ratio)
        # upper_ind = window_length - lower_ind
        # # Create a copy of the sorted data and Winsorize it
        # winsorized_data = self._sorted_data[-window_length:].copy()
        # winsorized_data[:lower_ind] = [winsorized_data[lower_ind]] * lower_ind
        # winsorized_data[upper_ind:] = [winsorized_data[upper_ind - 1]] * (window_length - upper_ind)

        # Apply winsorization on the data within the window
        winsorized_data = mstats.winsorize(np.array(self._sorted_data_winsorized), limits=[ratio, ratio])
        # Compute the mean
        winsorized_mean = np.mean(winsorized_data)
        return winsorized_mean
    
    def __cosine_tapered_mean(self, ratio:float, window_length:int):
        """
        Calculate the cosine tapered mean of the sorted data. (For a specific sorted data)

        Parameters:
        ratio (float): The proportion of values to taper. Must be in the range [0,1].
        window_length (int): The number of recent values to consider. Must be smaller than the total number of the streaming data
        
        Returns:
        cosine_tapered_mean (float): The cosine tapered mean of the data.
        """
        assert isinstance(ratio, (float, int)) and 0 <= ratio <= 1, f"ratio={ratio} must be a float or float in the range [0, 1]"
        assert isinstance(window_length, int) and window_length <= self.n, f"Window length={window_length} must be an integer less than or equal to the number of observations={self.n}"
        if self._current_ind == 1:
            # Initialised the weights
            self.ctm_weights = tukey(window_length, ratio*2) 
        # Apply the weight to the sorted data, if-else is for the case when we don't have burnin
        if self._current_ind > window_length:
            cosine_tapered_data = self.ctm_weights * self._sorted_data_cosine
        else:
            ctm_weight = tukey(len(self._sorted_data_cosine), ratio*2) 
            cosine_tapered_data = ctm_weight * self._sorted_data_cosine
        # Compute the mean
        cosine_tapered_mean = np.mean(cosine_tapered_data)
        return cosine_tapered_mean
        
    def __mad(self):
        """
        Compute the Median Absolute Deviation (MAD) of the sorted data. (For a specific sorted data)

        Returns:
        mad (float): The MAD of the sorted data.
        """
        median = np.median(self._sorted_data)
        mad = np.median(np.abs(self._sorted_data - median))
        return mad
    
    def __interquartile_range(self):
        """
        Compute the Interquartile Range (IQR) of the sorted data. (For a specific sorted data)

        Returns:
        iqr_value (float): The IQR of the sorted data.
        """
        iqr_value = iqr(self._sorted_data)
        return iqr_value
    
    def __winsorized_variance(self, ratio:float):
        """
        Calculate the winsorized variance of the sorted data. (For a specific sorted data)

        Parameters:
        ratio (float): The proportion of values to replace.
        
        Returns:
        winsorized_variance (float): The winsorized variance of the data.
        """
        assert isinstance(ratio, (float, int)) and 0 <= ratio <= 1, f"ratio={ratio} must be a float or float in the range [0, 1]"
        # Create a copy of the sorted data
        winsorized_data = self._sorted_data.copy()
        # Compute lower and upper cutoff indices
        lower_ind = int(np.floor(len(self._sorted_data)) * ratio)
        upper_ind = len(self._sorted_data) - lower_ind
        # Replace the lowest and highest values
        winsorized_data[:lower_ind] = [self._sorted_data[lower_ind]] * lower_ind
        winsorized_data[upper_ind:] = [self._sorted_data[upper_ind - 1]] * (len(self._sorted_data) - upper_ind) # remind that the ind start with 0
        # winsorized_data = mstats.winsorize(self._sorted_data, limits=[ratio, ratio])
        # Compute the variance
        winsorized_variance = np.var(winsorized_data)
        return winsorized_variance