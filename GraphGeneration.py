import warnings
import numpy as np
import matplotlib.pyplot as plt
from NormalParaEstimate import Normal_Mean_Var_Estimator
from ControlChartFunc import ControlChart

def __combine_alert_ind(upper_alert, lower_alert, burnin):
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
    alert_ind = alert_ind
    return np.sort(alert_ind)

def __first_true_detected_alert_ind(alert_ind, true_cp):
    if len(alert_ind)==0:
        first_true_alert = None
    else:
        if true_cp == None:
            first_true_alert = None
        else:
            first_true_alert = next((ind for ind in alert_ind if ind >= true_cp), None)
    return first_true_alert

def generate_cusum_chart(data, burnin, cusum_k, cusum_h, true_cp):
    """
    Generate a CUSUM control chart with burn-in period highlighted and alert values from a given dataset.

    Parameters:
    data (numpy.array): Streaming data points.
    burnin (int): Number of initial data points to estimate the mean and variance.
    cusum_k (float): Reference value for CUSUM control chart.
    cusum_h (float): Decision interval for CUSUM control chart.
    true_cp (int, optional): Actual point of change in data.

    Returns:
    None: This function outputs a (1,3) graph of CUSUM control chart, alert values chart and data chart.
    """
    assert isinstance(data, np.ndarray), f"Input={data} must be a numpy array"
    assert isinstance(burnin, int) and burnin >=0, f"burnin ({burnin}) must be a non-negative integer"
    assert (true_cp is None or (isinstance(true_cp, int) and true_cp >=0)), f"true_cp ({true_cp}) must be a non-negative integer or None"
    assert isinstance(cusum_k, (int, float)) and cusum_k > 0, f"cusum_k={cusum_k} must be a positive number"
    assert isinstance(cusum_h, (int, float)) and cusum_h > 0, f"cusum_h={cusum_h} must be a positive number"
    if burnin >= len(data):
        raise ValueError(f"Burnin period ({burnin}) must be less than the length of the data ({len(data)})")
    data_burnin_est = Normal_Mean_Var_Estimator(data[:burnin])
    data_ff_mean = data_burnin_est.sample_mean()
    data_ff_var = data_burnin_est.var_with_unknown_mean()
    data_cc = ControlChart(data[burnin:])
    d_c_s_p, d_c_s_m, d_c_au, d_c_al = data_cc.cusum_val(k=cusum_k, h=cusum_h, mu=data_ff_mean, 
                                                          sigma=np.sqrt(data_ff_var))
    alert_ind = __combine_alert_ind(d_c_au, d_c_al, burnin)
    first_true_alert_ind = __first_true_detected_alert_ind(alert_ind, true_cp) # find the index of first true alert
    # Append zeros for burn-in
    d_c_s_p = np.concatenate((np.zeros(burnin), d_c_s_p))
    d_c_s_m = np.concatenate((np.zeros(burnin), d_c_s_m))
    d_c_au = np.concatenate((np.zeros(burnin), d_c_au))
    d_c_al = np.concatenate((np.zeros(burnin), d_c_al))
    data_cp = data_cc.cusum_detect(k=cusum_k, h=cusum_h, mu=data_ff_mean, 
                                   sigma=np.sqrt(data_ff_var), burnin=burnin)
    if data_cp == first_true_alert_ind:
        first_true_alert_ind = None
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    # First graph: CUSUM Control Chart
    # axs[0].plot(-(cusum_h * np.sqrt(data_ff_var) * np.ones(len(d_c_s_p))), label='Lower limit')
    # axs[0].plot(cusum_h * np.sqrt(data_ff_var) * np.ones(len(d_c_s_p)), label='Upper limit')
    axs[0].plot(cusum_h * np.ones(len(d_c_s_p)), label='Bound limit') # new def for cusum
    axs[0].plot(d_c_s_p, label='$S_m^+$')
    axs[0].plot(d_c_s_m, label='$S_m^-$')
    axs[0].axvspan(0, burnin-1, facecolor='grey', alpha=0.25) # shadow for burnin
    axs[0].axvline(x=burnin-1, color='gray', linestyle=':', label="End of burn-in")
    if true_cp is not None:
        axs[0].axvline(x=true_cp, color='firebrick', linestyle='--', label="True CP")
    if data_cp is not None:
        axs[0].axvline(x=data_cp, color='skyblue', linestyle='--', label="Detected CP")
    if first_true_alert_ind is not None:
        axs[0].axvline(x=first_true_alert_ind, color='gold', linestyle=':', label="First True")
    axs[0].legend(loc='upper left')
    axs[0].set_title('CUSUM Control Chart')
    # Second graph: Alert values
    axs[1].plot(d_c_au, label='Upper Alert')
    axs[1].plot(d_c_al, label='Lower Alert')
    axs[1].axvspan(0, burnin-1, facecolor='grey', alpha=0.25)
    axs[1].axvline(x=burnin-1, color='gray', linestyle=':', label="End of burn-in")
    if true_cp is not None:
        axs[1].axvline(x=true_cp, color='firebrick', linestyle='--', label="True CP")
    if data_cp is not None:
        axs[1].axvline(x=data_cp, color='skyblue', linestyle='--', label="Detected CP")
    if first_true_alert_ind is not None:
        axs[1].axvline(x=first_true_alert_ind, color='gold', linestyle=':', label="First True")
    axs[1].legend(loc='upper left')
    axs[1].set_title('Alert Values')
    # Third graph: Data
    axs[2].plot(data, label='Data')
    axs[2].axvspan(0, burnin-1, facecolor='grey', alpha=0.25)
    axs[2].axvline(x=burnin-1, color='gray', linestyle=':', label="End of burn-in")
    if data_cp is not None:
        axs[2].axvline(x=data_cp, color='skyblue', linestyle='--', label="Detected CP")
    if true_cp is not None:
        axs[2].axvline(x=true_cp, color='firebrick', linestyle='--', label="True CP")
    if first_true_alert_ind is not None:
        axs[2].axvline(x=first_true_alert_ind, color='gold', linestyle=':', label="First True")
    axs[2].legend(loc='upper left')
    axs[2].set_title('Data')
    plt.show()


def generate_ewma_chart(data, burnin, ewma_rho, ewma_k, true_cp):
    """
    Generate an EWMA control chart with burn-in period highlighted and alert values from a given dataset.

    Parameters:
    data (numpy.array): Streaming data points.
    burnin (int): Number of initial data points to estimate the mean and variance.
    ewma_rho (float): Smoothing constant for EWMA control chart.
    ewma_k (float): Multiplier of the standard deviation for control limits.
    true_cp (int, optional): Actual point of change in data.

    Returns:
    None: This function outputs a (1,2) graph of EWMA control chart and alert values chart.
    """
    assert isinstance(data, np.ndarray), f"Input={data} must be a numpy array"
    assert isinstance(burnin, int) and burnin >=0, f"burnin ({burnin}) must be a non-negative integer"
    assert (true_cp is None or (isinstance(true_cp, int) and true_cp >=0)), f"true_cp ({true_cp}) must be a non-negative integer or None"
    assert isinstance(ewma_rho, float) and 0 < ewma_rho < 1, f"ewma_rho={ewma_rho} must be a float in the range (0, 1)"
    assert isinstance(ewma_k, (int, float)) and ewma_k > 0, f"ewma_k={ewma_k} must be a positive number"
    if burnin >= len(data):
        raise ValueError(f"Burnin period ({burnin}) must be less than the length of the data ({len(data)})")
    data_burnin_est = Normal_Mean_Var_Estimator(data[:burnin])
    data_ff_mean = data_burnin_est.sample_mean()
    data_ff_var = data_burnin_est.var_with_unknown_mean()
    data_cc = ControlChart(data[burnin:])
    d_e_s, d_e_v, d_e_au, d_e_al = data_cc.ewma_val(rho=ewma_rho, k=ewma_k, mu=data_ff_mean, 
                                                    sigma=np.sqrt(data_ff_var))
    data_e_cp = data_cc.ewma_detect(rho=ewma_rho, k=ewma_k, mu=data_ff_mean, 
                                    sigma=np.sqrt(data_ff_var), burnin=burnin)
    alert_ind = __combine_alert_ind(d_e_au, d_e_al, burnin)
    first_true_alert_ind = __first_true_detected_alert_ind(alert_ind, true_cp) # find the index of first true alert
    if data_e_cp == first_true_alert_ind:
        first_true_alert_ind = None
    d_e_au = np.concatenate((np.zeros(burnin), d_e_au)) # append 0 to burnin
    d_e_al = np.concatenate((np.zeros(burnin), d_e_al))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot stream data, Sn, mu, upper and lower alert
    axs[0].plot(data, label="stream data")
    axs[0].axvline(x=burnin, color='gray', linestyle=':', label="End of burn-in")
    axs[0].plot(np.ones(len(data))*data_ff_mean, label="$\hat{\mu}$")
    axs[0].plot(range(burnin, len(data)), d_e_s, label="$S_n$")
    axs[0].plot(range(burnin, len(data)), data_ff_mean + ewma_k * np.sqrt(d_e_v), label="Upper alert bound")
    axs[0].plot(range(burnin, len(data)), data_ff_mean - ewma_k * np.sqrt(d_e_v), label="Lower alert bound")
    if true_cp is not None:
        axs[0].axvline(x=true_cp, color='firebrick', linestyle='--', label="True CP")
    if data_e_cp is not None:
        axs[0].axvline(x=data_e_cp, color='skyblue', linestyle='--', label="Detected CP")
    if first_true_alert_ind is not None:
        axs[0].axvline(x=first_true_alert_ind, color='gold', linestyle=':', label="First True")
    axs[0].axvspan(0, burnin, alpha=0.25, color='gray')
    axs[0].legend(loc='upper left')
    axs[0].set_xlabel('Time stream')
    axs[0].set_ylabel('Value')
    axs[0].set_title('EWMA Control Chart')

    # Plot alert
    axs[1].axvline(x=burnin, color='gray', linestyle=':', label="End of burn-in")
    axs[1].axvspan(0, burnin, alpha=0.25, color='gray')
    axs[1].plot(d_e_au, label="Upper alert")
    axs[1].plot(d_e_al, label="Lower alert")
    if data_e_cp is not None:
        axs[1].axvline(x=data_e_cp, color='skyblue', linestyle='--', label="Detected CP")
    if true_cp is not None:
        axs[1].axvline(x=true_cp, color='firebrick', linestyle='--', label="True CP")
    if first_true_alert_ind is not None:
        axs[1].axvline(x=first_true_alert_ind, color='gold', linestyle=':', label="First True")
    axs[1].legend(loc='upper left')
    axs[1].set_xlabel('Time stream')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Alert value over stream')
    plt.show()