import warnings
import numpy as np
from NormalParaEstimate import Normal_Mean_Var_Estimator
from ControlChartFunc import ControlChart

def combine_alert_ind(upper_alert, lower_alert, burnin):
    """
    Combine and sort the upper alert indices and lower alert indices 
    """
    if len(upper_alert) == 0 and len(lower_alert) == 0:
        warnings.warn("Both upper_alert and lower_alert are empty. The function will return an empty list.")
        return []
    if len(upper_alert) == 0:
        warnings.warn("upper_alert is empty.")
        au_ind = []
    else:
        au_ind = np.where(upper_alert == 1.)[0]       
    if len(lower_alert) == 0:
        warnings.warn("lower_alert is empty.")
        al_ind = []
    else:
        al_ind = np.where(lower_alert == 1.)[0]      
    alert_ind = np.append(au_ind, al_ind) + burnin
    return np.sort(alert_ind)

def compute_arl0(alert_ind, true_cp, data_size):
    """
    Compute ARL0 given a list of alert indices and the true change point.
    """
    if true_cp == None:
        true_cp = np.inf
    false_alerts = [ind for ind in alert_ind if ind < true_cp]
    if len(false_alerts) > 1:
        # append 0 to the first position and compute the difference between
        arl0 = np.mean(np.diff(np.insert(false_alerts, 0, 0))) 
    elif len(false_alerts) == 1:
        arl0 = false_alerts[0]
    else:
        # No false alert, in order to aviod NA when computing mean and variance, we set it to a fix value and raise warning
        if true_cp != np.inf:
            arl0 = true_cp
            warnings.warn(f"No False alert detected, ARL0 is set to true changepoint position:{true_cp}.")
        else:
            arl0 = data_size # could be change to other number
            warnings.warn(f"No False alert detected, ARL0 is set to be the length of streaming data:{data_size}.")
    return arl0

def compute_arl1(alert_ind, true_cp, data_size):
    """
    Compute ARL1 given a list of alert indices and the true change point.
    """
    if true_cp == None:
        arl1 = np.nan
        warnings.warn("No true change point position, ARL1 is meaningless and set to np.nan.")
    else:
        first_true_alert = next((ind for ind in alert_ind if ind >= true_cp), None)
        if first_true_alert is not None:
            arl1 = first_true_alert - true_cp
        else:
            arl1 = data_size - true_cp # could be change to other number
            warnings.warn(f"No changepoint detected, ARL1 is set to be the length of out-of-control data:{data_size - true_cp}.")
    return arl1

def arl_cusum(data:np.ndarray, burnin:int, cusum_k:float, cusum_h:float, true_cp:int):
    """
    Generate the Average Run Length (ARL0 and ARL1) values of streaming data with control chart using CUSUM.
    This function is now limited to compute ARL for single change point data stream

    Parameters:
    data (numpy.array): Streaming data points.
    burnin (int): The number of initial data points used to estimate the mean and variance. Must be a positive integer
    cusum_k (float or int): Reference value or slack value. It should be a positive number.
    cusum_h (float or int): Decision threshold for the alert. It should be a positive number.
    true_cp (int or None): Actual point of change in data, can be None.

    Returns:
    arl0 (float): Average number of points until the first false alert.
    arl1 (float): Average number of points after the true change point until the first true detection.
    """
    assert isinstance(data, np.ndarray), f"Input={data} must be a numpy array"
    assert isinstance(burnin, int) and burnin >=0, f"burnin ({burnin}) must be a non-negative integer"
    assert (true_cp is None or (isinstance(true_cp, int) and true_cp >=0)), f"true_cp ({true_cp}) must be a non-negative integer or None"
    if true_cp is not None:
        assert burnin < true_cp, f"Value of burnin:{burnin} should smaller than true_cp:{true_cp}"
    assert isinstance(cusum_k, (int, float)) and cusum_k > 0, f"cusum_k={cusum_k} must be a positive number"
    assert isinstance(cusum_h, (int, float)) and cusum_h > 0, f"cusum_h={cusum_h} must be a positive number"
    if burnin >= len(data):
        raise ValueError(f"Burnin period ({burnin}) must be less than the length of the data ({len(data)})")
    data_burnin_est = Normal_Mean_Var_Estimator(data[:burnin])
    data_mean_burnin = data_burnin_est.sample_mean()
    data_var_burnin = data_burnin_est.var_with_unknown_mean()
    data_cc = ControlChart(data[burnin:])
    _, _, au, al = data_cc.cusum_val(k=cusum_k, h=cusum_h, mu=data_mean_burnin, 
                                                          sigma=np.sqrt(data_var_burnin))
    alert_ind = combine_alert_ind(au, al, burnin)
    arl0 = compute_arl0(alert_ind, true_cp, len(data))
    arl1 = compute_arl1(alert_ind, true_cp, len(data))
    return arl0, arl1

def arl_ewma(data:np.ndarray, burnin:int, ewma_rho:float, ewma_k:float, true_cp:int):
    """
    Calculate the Average Run Length (ARL0 and ARL1) values of streaming data for an EWMA control chart.
    This function is now limited to compute ARL for single change point data stream

    Parameters:
    data (numpy.array): Streaming data points.
    burnin (int): The number of initial data points used to estimate the mean and variance. Must be a positive integer
    ewma_rho (float): The control parameter for the weight of the current input. It should be in the range [0, 1]ß.
    ewma_k (float or int): The control parameter for the alert function. It should be in the range (0, inf).
    true_cp (int or None): Actual point of change in data, can be None.

    Returns:
    arl0 (float): Average number of points until the first false alert.
    arl1 (float): Average number of points after the true change point until the first true detection.
    """
    assert isinstance(data, np.ndarray), f"Input={data} must be a numpy array"
    assert isinstance(burnin, int) and burnin >=0, f"burnin ({burnin}) must be a non-negative integer"
    assert (true_cp is None or (isinstance(true_cp, int) and true_cp >=0)), f"true_cp ({true_cp}) must be a non-negative integer or None"
    if true_cp is not None:
        assert burnin < true_cp, f"Value of burnin:{burnin} should smaller than true_cp:{true_cp}"
    assert isinstance(ewma_rho, float) and 0 <= ewma_rho <= 1, f"ewma_rho={ewma_rho} must be a float in the range [0, 1]"
    assert isinstance(ewma_k, (int, float)) and ewma_k > 0, f"ewma_k={ewma_k} must be a positive number"
    if burnin >= len(data):
        raise ValueError(f"Burnin period ({burnin}) must be less than the length of the data ({len(data)})")
    data_burnin_est = Normal_Mean_Var_Estimator(data[:burnin])
    data_mean_burnin = data_burnin_est.sample_mean()
    data_var_burnin = data_burnin_est.var_with_unknown_mean()
    data_cc = ControlChart(data[burnin:])
    _, _, au, al = data_cc.ewma_val(rho=ewma_rho, k=ewma_k, mu=data_mean_burnin, 
                                                    sigma=np.sqrt(data_var_burnin))
    alert_ind = combine_alert_ind(au, al, burnin)
    arl0 = compute_arl0(alert_ind, true_cp, len(data))
    arl1 = compute_arl1(alert_ind, true_cp, len(data))
    return arl0, arl1

def arl_robust_mean(data:np.ndarray, burnin:int, median_window_length:int, trimmed_ratio:float, winsorized_ratio:float, cosine_ratio:float,
                    trimmed_window_length:int, winsorized_window_length:int, cosine_window_length:int,  z_val:float, alpha_val:float, true_cp:int):
    """
    Calculates the Average Run Length (ARL) values for a data stream, using various robust mean estimation methods. 

    The methods include sliding window median, sliding window trimmed mean, sliding window winsorized mean, and sliding window cosine tapered mean. Each method requires specific parameters. 
    
    If these parameters are not provided, their corresponding mean and ARL values will not be calculated. 
    
    The calculated ARL values are returned as a dictionary.

    Parameters:
    data (numpy.ndarray): Data stream to analyze.
    burnin (int): Number of initial data points for estimating the mean and variance. Should be a positive integer.
    median_window_length (int): Length of the sliding window for median estimation. Should be a positive integer <= total number of observations.
    trimmed_ratio (float): Proportion of values to trim from both ends for trimmed mean estimation. Should be in the range [0,1].
    winsorized_ratio (float): Proportion of values to replace from both ends for winsorized mean estimation. Should be in the range [0,1].
    cosine_ratio (float): Proportion of values to taper for cosine-tapered mean estimation. Should be in the range [0,1].
    trimmed_window_length (int): The number of recent values to consider for the trimmed mean calculation.
    winsorized_window_length (int): The number of recent values to consider for the winsorized mean calculation.
    cosine_window_length (int): The number of recent values to consider for the cosine tapered mean calculation.
    z_val (float): Control parameter for the confidence interval width. Should be in the range (0, inf).
    alpha_val (float): Control parameter for the alert function to decide the allowed fluctuation range. Should be in the range (0, inf).
    true_cp (int or None): Actual change point in the data stream, if known. This value can also be None.

    Returns:
    results (dict): Dictionary containing the ARL0 and ARL1 values for each method. If a method's specific parameter is not provided, its corresponding ARL values will not be included in the dictionary.
    """
    assert isinstance(data, np.ndarray), f"Input={data} must be a numpy array"
    data_len = len(data)
    assert isinstance(burnin, int) and burnin >=0, f"burnin ({burnin}) must be a non-negative integer"
    assert (true_cp is None or (isinstance(true_cp, int) and true_cp >=0)), f"true_cp ({true_cp}) must be a non-negative integer or None"
    if true_cp is not None:
        assert burnin < true_cp, f"Value of burnin:{burnin} should smaller than true_cp:{true_cp}"
    assert (median_window_length is None or (isinstance(median_window_length, int) and 0 < median_window_length <= data_len)), f"Median window length={median_window_length} must be an positive integer less than or equal to the number of observations={data_len-burnin}"
    assert (trimmed_ratio is None or (isinstance(trimmed_ratio, (int, float)) and 0 <= trimmed_ratio <= 1)), f"trimmed_ratio={trimmed_ratio} must be a float in the range [0, 1]"        
    assert (winsorized_ratio is None or (isinstance(winsorized_ratio, (int, float)) and 0 <= winsorized_ratio <= 1)), f"winsorized_ratio={winsorized_ratio} must be a float in the range [0, 1]"        
    assert (cosine_ratio is None or (isinstance(cosine_ratio, (int, float)) and 0 <= cosine_ratio <= 1)), f"cosine_ratio={cosine_ratio} must be a float in the range [0, 1]"            
    assert (trimmed_window_length is None or (isinstance(trimmed_window_length, int) and 0 < trimmed_window_length <= data_len)), f"Trimmed window length={trimmed_window_length} must be an positive integer less than or equal to the number of observations={data_len-burnin}"
    assert (winsorized_window_length is None or (isinstance(winsorized_window_length, int) and 0 < winsorized_window_length <= data_len)), f"Winsorized window length={winsorized_window_length} must be an positive integer less than or equal to the number of observations={data_len-burnin}"
    assert (cosine_window_length is None or (isinstance(cosine_window_length, int) and 0 < cosine_window_length <= data_len)), f"Cosine Tapered window length={cosine_window_length} must be an positive integer less than or equal to the number of observations={data_len-burnin}"
    if burnin >= data_len:
        raise ValueError(f"Burnin period ({burnin}) must be less than the length of the data ({data_len})")
    data_burnin_est = Normal_Mean_Var_Estimator(data[:burnin])
    data_mean_burnin = data_burnin_est.sample_mean()
    data_sd_burnin = np.sqrt(data_burnin_est.var_with_unknown_mean())
    h_val = alpha_val * data_sd_burnin
    data_cc = ControlChart(data[burnin:])

    data_cc.compute_robust_methods_mean_seq(median_window_length, trimmed_ratio, winsorized_ratio, cosine_ratio,
                                    trimmed_window_length, winsorized_window_length, cosine_window_length, 
                                    burnin_data=data[:burnin])
    results = {}

    # ARL for sliding window median confidence interval
    if median_window_length is not None:
        _, _, swm_au, swm_al = data_cc.sliding_window_median_CI_val(z_val, h_val, data_mean_burnin, data_sd_burnin)
        swm_alert_ind = combine_alert_ind(swm_au, swm_al, burnin)
        results["M"] = {"arl0": compute_arl0(swm_alert_ind, true_cp, data_len), 
                          "arl1": compute_arl1(swm_alert_ind, true_cp, data_len)}
    # ARL for sliding window trimmed mean confidence interval
    if trimmed_ratio is not None:
        _, _, tm_au, tm_al = data_cc.trimmed_mean_CI_val(z_val, h_val, data_mean_burnin, data_sd_burnin)
        tm_alert_ind = combine_alert_ind(tm_au, tm_al, burnin)
        results["T"] = {"arl0": compute_arl0(tm_alert_ind, true_cp, data_len), 
                         "arl1": compute_arl1(tm_alert_ind, true_cp, data_len)}
    # ARL for sliding window winsorized mean confidence interval
    if winsorized_ratio is not None:
        _, _, wm_au, wm_al = data_cc.winsorized_mean_CI_val(z_val, h_val, data_mean_burnin, data_sd_burnin)
        wm_alert_ind = combine_alert_ind(wm_au, wm_al, burnin)
        results["W"] = {"arl0": compute_arl0(wm_alert_ind, true_cp, data_len), 
                         "arl1": compute_arl1(wm_alert_ind, true_cp, data_len)}
    # ARL for sliding window cosine taper mean confidence interval
    if cosine_ratio is not None:
        _, _, ctm_au, ctm_al = data_cc.cosine_tapered_mean_CI_val(z_val, h_val, data_mean_burnin, data_sd_burnin)
        ctm_alert_ind = combine_alert_ind(ctm_au, ctm_al, burnin)
        results["CT"] = {"arl0": compute_arl0(ctm_alert_ind, true_cp, data_len), 
                          "arl1": compute_arl1(ctm_alert_ind, true_cp, data_len)}

    return results
