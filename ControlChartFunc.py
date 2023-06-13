import warnings
import numpy as np

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


