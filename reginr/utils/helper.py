import os
import pandas as pd
import numpy as np
from numpy import inf
from scipy.special import logit,gamma
from scipy.stats import weibull_min
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, normalize
from pyemd import emd_samples
import pickle


def load_data(dirpath, combined=True):
    """Load stem cell data
    
    Args:
        dirpath (str): path to directory containing e14_labels.csv and e14_labels.csv files
        combined (bool): combine e14 and r1 samples
    
    """
    filepath_e14 = os.path.join(dirpath, 'e14_labels.csv')
    filepath_r1 = os.path.join(dirpath, 'r1_labels.csv')
    raw_e14 = pd.read_csv(filepath_e14)
    raw_r1 = pd.read_csv(filepath_r1)
    
    di = {1:'ESC', 2:'EPI', 3:'NPC'}
    raw_e14['ct_label'] = raw_e14['ct_label'].map(di)
    raw_r1['ct_label'] = raw_r1['ct_label'].map(di)
    # e14
    e14 = pd.DataFrame(raw_e14['obs'].str.split('.').to_list(), columns=['strain', 'time', 'id'])
    e14['ct_label'] = raw_e14['ct_label']
    e14['time'] = e14['time'].map(lambda x: x.strip('h'))
    e14['time'] = e14['time'].astype(int)
    # r1
    r1 = pd.DataFrame(raw_r1['obs'].str.split('.').to_list(), columns=['strain', 'time', 'id'])
    r1['ct_label'] = raw_r1['ct_label']
    r1['time'] = r1['time'].map(lambda x: x.strip('h'))
    r1['time'] = r1['time'].astype(int)
    
    tsamples = e14['time'].unique()
    if combined:
        esc = []
        for t in tsamples:
            count = e14.loc[(e14['time']==t) & (e14['ct_label']=='ESC'), 'ct_label'].count()
            count += r1.loc[(r1['time']==t) & (r1['ct_label']=='ESC'), 'ct_label'].count()
            esc.append(count)
        
        epi = []
        for t in tsamples:
            count = e14.loc[(e14['time']==t) & (e14['ct_label']=='EPI'), 'ct_label'].count()
            count += r1.loc[(r1['time']==t) & (r1['ct_label']=='EPI'), 'ct_label'].count()
            epi.append(count)
            
        npc = []
        for t in tsamples:
            count = e14.loc[(e14['time']==t) & (e14['ct_label']=='NPC'), 'ct_label'].count()
            count += r1.loc[(r1['time']==t) & (r1['ct_label']=='NPC'), 'ct_label'].count()
            npc.append(count)
        
        d = {'Time':tsamples, 'ESC' : esc, 'EPI':epi, 'NPC':npc}
        count_df = pd.DataFrame(d)
        count_df['Source'] = 'experimental'
    
        return count_df
    
    else:
        esc_e14 = []
        esc_r1 = []
        for t in tsamples:
            esc_e14.append(e14.loc[(e14['time']==t) & (e14['ct_label']=='ESC'), 'ct_label'].count())
            esc_r1.append(r1.loc[(r1['time']==t) & (r1['ct_label']=='ESC'), 'ct_label'].count())
         
        epi_e14 = []
        epi_r1 = []
        for t in tsamples: 
            epi_e14.append(e14.loc[(e14['time']==t) & (e14['ct_label']=='EPI'), 'ct_label'].count())
            epi_r1.append(r1.loc[(r1['time']==t) & (r1['ct_label']=='EPI'), 'ct_label'].count())
         
        npc_e14 = []
        npc_r1 = []
        for t in tsamples:
            npc_e14.append(e14.loc[(e14['time']==t) & (e14['ct_label']=='NPC'), 'ct_label'].count())
            npc_r1.append(r1.loc[(r1['time']==t) & (r1['ct_label']=='NPC'), 'ct_label'].count())
         
                   
        d_e14 = {'Time':tsamples, 'ESC' : esc_e14, 'EPI':epi_e14, 'NPC':npc_e14}
        count_e14 = pd.DataFrame(d_e14)
        count_e14['Source'] = 'experimental'
        
        d_r1 = {'Time':tsamples, 'ESC' : esc_r1, 'EPI':epi_r1, 'NPC':npc_r1}
        count_r1 = pd.DataFrame(d_r1)
        count_r1['Source'] = 'experimental'
        
        return count_e14, count_r1

def calculate_bs_avg(bs_raw):
    """Takes sampled kmeans results and calculates mean"""
    states = bs_raw['state'].unique()
    times = bs_raw['time'].unique()
    lineages = ['E14', 'R1']
    data = {'time' : [], 'state' : [], 'value':[], 'L1':[]}
    for l in lineages:
        for t in times:
            for s in states:
                v = bs_raw[(bs_raw['state']==s) & (bs_raw['time']==t) & (bs_raw['L1']==l)]['value'].mean()
                data['time'].append(t)
                data['state'].append(s)
                data['value'].append(v)
                data['L1'].append(l)
                
    return pd.DataFrame(data)

def bs_to_wide(bs_avg, cell_line):
    """
    Args:
        cell_line (str): ['R1', 'E14']
    """
    bs_avg_r1 = bs_avg[bs_avg['L1']==cell_line]
    bs_avg_wide = bs_avg_r1.pivot(index='time', columns='state', values='value')
    bs_avg_wide = bs_avg_wide.reset_index(level=0)
    bs_avg_wide = bs_avg_wide.rename(columns={'time' : 'Time'}) # rename column otherwise error in rmse calc
    bs_avg_wide.columns.name = None
    return bs_avg_wide

def calculate_rmse(count_df, pop_df, avg=True, verbose=True, weights=None):
    """Input dataframes need to be in wide format
    
    Args:
        weights (list): [esc, epi, npc] float values on how to weigh rmse average
    
    """
    # change count_df values to float
    count_df['Time'] = count_df['Time'].astype('float')
    count_df['ESC'] = count_df['ESC'].astype('float')
    count_df['EPI'] = count_df['EPI'].astype('float')
    count_df['NPC'] = count_df['NPC'].astype('float')
    
    # merge to match shape
    merged = pd.merge_asof(pop_df, count_df, on='Time', direction='nearest', tolerance=0.2, suffixes=('_sim', '_exp')).dropna()
    
    y = merged.loc[:, ['ESC_exp','EPI_exp', 'NPC_exp']]
    yhat = merged.loc[:,['ESC_sim','EPI_sim', 'NPC_sim']]
    
    rmse = mean_squared_error(y, yhat, squared=False, multioutput='raw_values')
    if weights:
        rmse_avg = mean_squared_error(y, yhat, squared=False, multioutput=weights)
    else:
        rmse_avg = mean_squared_error(y, yhat, squared=False, multioutput='uniform_average')
    print('RMSE:\n ESC = {}\n EPI = {}\n NPC = {}\n Average = {}'.format(rmse[0], rmse[1], rmse[2], rmse_avg)) if verbose else None
    if avg:
        return rmse_avg
    else:
        return rmse

    
def calculate_emd(y_sim, alpha, r0):
    """Calculates Earth Movers distance between two distributions
    
    Takes wait-time distribution and calculates the EMD against the theoretical weibull distribution 
    given parameters at the start of the simulation
    
    Args:
        y_sim (list): (n_sim, n_wait_times) list of wait-times for each simulation, 
            for given reaction channel (esc, epi). Obtained from Channels.wait_times attribute
    
    Returns:
        EMD between simulation and true weibull distribution
    """
    y_sim = [item for sublist in y_sim for item in sublist] # flatten list
    y_sim = np.array(y_sim)
    n = 10000
    
    # params for generating theoretical samples (ground truth)
    k = alpha + 1 # shape
    beta = (alpha + 1) * (r0 * gamma((alpha + 2)/(alpha + 1)))**(alpha + 1)
    lam = np.power((alpha+1)/beta, 1/(alpha+1)) # scale
    # generate true samples
    y_true = weibull_min.rvs(k, loc=0, scale=lam, size=n)
    
    d = emd_samples(y_sim, y_true)
    
    return d
    
    
def calculate_emd_avg(y_sim, alpha, r0):
    """Calculates average Earth Movers distance between two distributions
    
    For each simulations, takes wait-time distribution and calculates the EMD against the theoretical weibull distribution 
    given parameters at the start of the simulation. Normalised by r0.
    
    Args:
        y_sim (list): (n_sim, n_wait_times) dimension list of wait-times for each simulation, 
            for given reaction channel (esc, epi). Obtained from Channels.wait_times attribute
    
    Returns:
        EMD averaged over all simulations normalised by r0
    """
    
    n_sim = len(y_sim)
    
    # params for generating theoretical samples (ground truth)
    k = alpha + 1 # shape
    beta = (alpha + 1) * (r0 * gamma((alpha + 2)/(alpha + 1)))**(alpha + 1)
    lam = np.power((alpha+1)/beta, 1/(alpha+1)) # scale
    n_samples = 100000
    
    
    emd_list = []
    for i in range(n_sim):
        y_hat = np.array(y_sim[i])
        y_true = weibull_min.rvs(k, loc=0, scale=lam, size=n_samples)
        emd_list.append(emd_samples(y_hat, y_true))
    
    return np.mean(emd_list) * r0
    
    
def arr_to_df(population, t_end, scaler, long=True):
    """Converts simulation array output to dataframe
    
    Each cell type averaged out over N simulations. 
    If scaler provided: Counts are then normalised using MinMaxScaler to allow comparison with experimental data.
    Otherwise uses l1 norm to normalise data
    
    """
    esc =population[:,:,0].mean(axis=0)
    epi =population[:,:,1].mean(axis=0)
    npc =population[:,:,2].mean(axis=0)
    pop_df = pd.DataFrame({'ESC':esc, 'EPI': epi, 'NPC':npc})
    time_points = np.linspace(0, t_end, population.shape[1])
    pop_df['Time'] =time_points
    pop_df['Source'] = 'simulated'
    
    # Normalise. If scaler not provided, create new one
    if scaler:
        if scaler.scale_.mean() == 1:
            print('Warning: wrong scaler used: {}'.format(scaler.scale_.mean()))
        pop_df[['ESC', 'EPI', 'NPC']] = scaler.transform(pop_df[['ESC', 'EPI', 'NPC']])
    else:
        #scaler = MinMaxScaler(feature_range=(0,1))
        #scaler = scaler.fit(pop_df[['ESC', 'EPI', 'NPC']])
        pop_df[['ESC', 'EPI', 'NPC']] = normalize(pop_df[['ESC', 'EPI', 'NPC']], norm='l1')

    
    
    
    if long:
        return pop_df.melt(id_vars=['Time', 'Source'], value_vars=['ESC', 'EPI', 'NPC'], var_name='Cell type', value_name='Count')
    else:
        return pop_df

def objective_fn(r0_esc, r0_epi, alpha_esc, alpha_epi):
    """
    Args:
        
        
    Returns:
        rmse (float): Average RMSE for all 3 cell types. Simulation vs experimental data.
    """
    bs_raw = pd.read_csv('./data/bootstraps10.csv')
    bs_avg = calculate_bs_avg(bs_raw)
    # remove poor datapoint: e14 75 esc
    idx = bs_avg[(bs_avg.state=='ESC') & (bs_avg.time==72) & (bs_avg.L1=='E14')].index
    bs_avg = bs_avg.drop(idx)
    # Logit transform
    a = bs_avg['value']
    y = logit_transform(a)
    bs_logit = bs_avg.copy()
    bs_logit['value'] = y
    # convert to wide for rmse calculation
    bs_logit_wide = bs_to_wide(bs_logit, cell_line = 'R1')
    
    # Define params
    channels = {
        'esc' : Channel('esc', r0=r0_esc, alpha=alpha_esc),
        'epi' : Channel('epi', r0=r0_epi, alpha=alpha_epi)
    }
    t_end = 170
    n_sim = 500
    timepoints = 100
    
    print("[INFO] starting simulation...")
    print("\tParams: r0_esc = {}, r0_epi = {}, alpha_esc = {}, alpha_epi = {}".format(r0_esc, r0_epi, alpha_esc, alpha_epi))
    population, channels_out, t2_error = simulate(t_end, n_sim, timepoints, channels, verbose=False)
    pop_df = arr_to_df(population, t_end, scaler=None, long=False)
    # Logit transform simulation output
    a = pop_df[['ESC', 'EPI', 'NPC']].values
    y = logit_transform(a)
    pop_logit = pop_df.copy()
    pop_logit[['ESC', 'EPI', 'NPC']] = y
    # Calculate rmse
    rmse = calculate_rmse(bs_avg_r1_wide, pop_logit, verbose=False)
    print("\tSimulation finished with RMSE: {}".format(rmse))
    return rmse

def logit_transform(a, t=5):
    """Apply logit function, setting a max threshold instead of +/- inf
    
    Args:
        a (np.array): array to transform
        t (float): max threshold for +/- inf values
        
    Returns:
        np.array of logit values
    """
    if type(a) is not np.ndarray:
        a = np.array(a)
    y = logit(a)
    # cap inf relative to max and min values - not implemented
    #ub = np.max(y[(y!=inf)&(y!=-inf)])
    #lb = np.min(y[(y!=inf)&(y!=-inf)])
    
    # replace inf values with threshold
    y[y == inf] = t
    y[y == -inf] = -t
    return y
    
def minmax_norm(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    if isinstance(data, pd.DataFrame):
        dfcopy = data.copy()
        dfcopy[['ESC', 'EPI', 'NPC']] = scaler.fit_transform(dfcopy[['ESC', 'EPI', 'NPC']])
        return dfcopy
    elif isinstance(data, np.ndarray):
        scaler = scaler.fit(data)
        return scaler.transform(data)
    else:
        raise ValueError('needs to be pd.DataFrame or np.ndarray')
        
def save_obj(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)

def load_obj(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)