import warnings
import numpy as np
from bisect import insort
from scipy.signal import tukey

class ControlChart:
    """
    This class implements the CUSUM and EWMA control charts for the detection of changes in streaming data.

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
        assert isinstance(burnin, int) and burnin >= 0, f"burnin={burnin} should be a non-negative value"
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
        assert isinstance(burnin, int) and burnin >= 0, f"burnin={burnin} should be a non-negative value"
        # compute the statistics value for cumsum
        _, _, au, al = self.ewma_val(rho, k, mu, sigma)
        # return the detected the alert
        ind = self.__check_alerts(au, al) 
        if ind is not None:
            ind += burnin 
        else:
            warnings.warn("No change point detected, detected index set to None")
        return ind


class RobustMethods:
    """
    This class implements various robust methods for computing the mean of streaming data, including trimmed mean, winsorized mean, and cosine tapered mean. It also provides a method for computing the sliding window median of the data.

    Attributes:
        x (np.ndarray): The input streaming data. It's a fixed dataset, but the values are coming in a stream.
        n (int): The number of observations in the data.

    Methods:
        compute_mean_sequence(trimmed_ratio:float, winsorized_ratio:float, cosine_ratio:float) -> dict:
            Computes the mean sequence of the data using three robust methods: trimmed mean, winsorized mean, and cosine tapered mean.

        sliding_window_median(window_length:int) -> list:
            Computes the median of a sliding window over the data.

    Private Methods:
        __insert_next_for_sorted_data() -> None:
            Inserts the next value from the data stream into the sorted list.

        __trimmed_mean(ratio:float) -> float:
            Calculates the trimmed mean of the sorted data.

        __winsorized_mean(ratio:float) -> float:
            Calculates the winsorized mean of the sorted data.

        __cosine_tapered_mean(ratio:float) -> float:
            Calculates the cosine tapered mean of the sorted data.
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
        self._current_ind = 0
        self._sorted_data = []

    def __insert_next_for__(self):
        """
        Insert the next value from the data stream into the sorted list and sort it.
        """
        if self._current_ind < self.n:
            insort(self._sorted_data, self.x[self._current_ind])
            self._current_ind += 1

    def compute_mean_sequence(self, trimmed_ratio:float, winsorized_ratio:float, cosine_ratio:float):
        """
        Compute the mean sequence of the data using three robust methods: trimmed mean, winsorized mean and cosine tapered mean.
        
        Parameters:
        trimmed_ratio (float): The proportion of values to trim for the trimmed mean calculation. (Two sided)
        winsorized_ratio (float): The proportion of values to replace for the winsorized mean calculation. (Two sided)
        cosine_ratio (float): The proportion of values to taper for the cosine tapered mean calculation. (Two sided)

        Returns:
        robust_mean_seq (dict): A dictionary containing the mean sequences calculated using each of the three robust methods.
        """
        trimmed_mean_seq = []
        winsorized_mean_seq = []
        cosine_mean_seq = []
        for _ in range(self.n):
            self.__insert_next_for__()
            trimmed_mean_seq.append(self.__trimmed_mean(trimmed_ratio))
            winsorized_mean_seq.append(self.__winsorized_mean(winsorized_ratio))
            cosine_mean_seq.append(self.__cosine_tapered_mean(cosine_ratio))
        robust_mean_seq = {
            "trimmed": np.array(trimmed_mean_seq),
            "winsorized": np.array(winsorized_mean_seq),
            "cosine": np.array(cosine_mean_seq)
        }
        return robust_mean_seq

    def __trimmed_mean(self, ratio:float):
        """
        Calculate the trimmed mean of the sorted data. (For a specific sorted data)

        Parameters:
        ratio (float): The proportion of values to trim.
        
        Returns:
        trimmed_mean (float): The trimmed mean of the data.
        """
        assert isinstance(ratio, (float, int)) and 0 <= ratio <= 1, f"ratio={ratio} must be a float or float in the range [0, 1]"
        # Compute lower and upper cutoff indices
        lower_ind = int(np.floor(len(self._sorted_data)) * ratio)
        upper_ind = len(self._sorted_data) - lower_ind
        # Trim the data and compute the mean
        trimmed_data = self._sorted_data[lower_ind:upper_ind]
        trimmed_mean = np.mean(trimmed_data)
        return trimmed_mean
    
    def __winsorized_mean(self, ratio:float):
        """
        Calculate the winsorized mean of the sorted data. (For a specific sorted data)

        Parameters:
        ratio (float): The proportion of values to replace.
        
        Returns:
        winsorized_mean (float): The winsorized mean of the data.
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
        # Compute the mean
        winsorized_mean = np.mean(winsorized_data)
        return winsorized_mean
    
    def __cosine_tapered_mean(self, ratio:float):
        """
        Calculate the cosine tapered mean of the sorted data. (For a specific sorted data)

        Parameters:
        ratio (float): The proportion of values to taper.
        
        Returns:
        cosine_tapered_mean (float): The cosine tapered mean of the data.
        """
        assert isinstance(ratio, (float, int)) and 0 <= ratio <= 1, f"ratio={ratio} must be a float or float in the range [0, 1]"
        # Initialised the weights
        weights = tukey(len(self._sorted_data), ratio) 
        # Apply the weight to the sorted data
        cosine_tapered_data = weights * self._sorted_data
        # Compute the mean
        cosine_tapered_mean = np.mean(cosine_tapered_data)
        return cosine_tapered_mean
    
    def sliding_window_median(self, window_length:int):
        """
        Compute the median of a sliding window over the data. (For the whole data stream)

        Parameters:
        window_length (int): The length of the sliding window.
        
        Returns:
        medians (list): The medians of each window of the whole data stream.
        """
        assert isinstance(window_length, int) and window_length <= self.n, f"Window length={window_length} must be an integer less than or equal to the number of observations={self.n}"
        medians = [] # for store a list of median values
        for i in range(self.n - window_length + 1):
            # Compute the median of the window and add it to the list
            window_median = np.median(self.x[i : i + window_length])
            medians.append(window_median)
        return medians