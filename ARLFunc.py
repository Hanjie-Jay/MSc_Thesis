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
    # print(false_alerts)
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
    burnin (int): Number of initial data points to estimate the mean and variance.
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
    data_ff_mean = data_burnin_est.sample_mean()
    data_ff_var = data_burnin_est.var_with_unknown_mean()
    data_cc = ControlChart(data[burnin:])
    _, _, d_c_au, d_c_al = data_cc.cusum_val(k=cusum_k, h=cusum_h, mu=data_ff_mean, 
                                                          sigma=np.sqrt(data_ff_var))
    alert_ind = combine_alert_ind(d_c_au, d_c_al, burnin)
    arl0 = compute_arl0(alert_ind, true_cp, len(data))
    arl1 = compute_arl1(alert_ind, true_cp, len(data))
    return arl0, arl1

def arl_ewma(data:np.ndarray, burnin:int, ewma_rho:float, ewma_k:float, true_cp:int):
    """
    Calculate the Average Run Length (ARL0 and ARL1) values of streaming data for an EWMA control chart.
    This function is now limited to compute ARL for single change point data stream

    Parameters:
    data (numpy.array): Streaming data points.
    burnin (int): Number of initial data points to estimate the mean and variance.
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
    data_ff_mean = data_burnin_est.sample_mean()
    data_ff_var = data_burnin_est.var_with_unknown_mean()
    data_cc = ControlChart(data[burnin:])
    _, _, d_e_au, d_e_al = data_cc.ewma_val(rho=ewma_rho, k=ewma_k, mu=data_ff_mean, 
                                                    sigma=np.sqrt(data_ff_var))
    alert_ind = combine_alert_ind(d_e_au, d_e_al, burnin)
    arl0 = compute_arl0(alert_ind, true_cp, len(data))
    arl1 = compute_arl1(alert_ind, true_cp, len(data))
    return arl0, arl1