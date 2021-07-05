import os
import copy
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp, odeint
from scipy.special import gamma
from scipy.stats import lognorm, expon, gengamma, weibull_min
from reliability.Fitters import Fit_Weibull_2P



def weibull_density(alpha, r0, t):
    beta = (alpha + 1) * (r0 * gamma((alpha + 2)/(alpha + 1)))**(alpha + 1)
    return beta * t**alpha * np.exp(-(beta*t**(alpha + 1)) / (alpha + 1))

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def plot_population(population, Tend, xlim=175, filepath = None):
    """Lineplot for each cell type averaged over all simulations
    
    Also plots individual simulations making up the average. However, if the number of simulations exceeds
    50, only random 50 individual simulations are plotted, otherwise takes too long.
    
    """
    plt.style.use('default')
    sns.set_context('notebook', font_scale=1.9)
    
    N_simulations = population.shape[0]
    timepoints = population.shape[1]
    time_points = np.linspace(0, Tend, timepoints)
    
    fig, ax = plt.subplots()
    #fig.set_size_inches(cm2inch(19,8))
    fig.set_size_inches(15,6.2)
    if N_simulations > 50:
        sim_idx = np.random.choice(np.arange(N_simulations), size=50, replace=False)
    else:
        sim_idx = np.arange(N_simulations)
    # plot individual simulations
    for i in sim_idx:
        sns.lineplot(x=time_points, y=population[i,:,0], 
                     lw=0.3, alpha=0.2, color=sns.color_palette()[3],
                     zorder=1, ax=ax)
        sns.lineplot(x=time_points, y=population[i,:,1], 
                     lw=0.3, alpha=0.2, color=sns.color_palette()[2], 
                     zorder=1, ax=ax)
        sns.lineplot(x=time_points, y=population[i,:,2], 
                     lw=0.3, alpha=0.2, color=sns.color_palette()[4], 
                     zorder=1, ax=ax)
    
    # plot average for each cell type
    sns.lineplot(x=time_points, y=population[:,:,0].mean(axis=0), 
                 color=sns.color_palette()[3], label='ESC', linewidth=2,
                 zorder=10, ax=ax)
    sns.lineplot(x=time_points, y=population[:,:,1].mean(axis=0), 
                 color=sns.color_palette()[2], label='EPI', linewidth=2,
                 zorder=10, ax=ax)
    sns.lineplot(x=time_points, y=population[:,:,2].mean(axis=0), 
                 color=sns.color_palette()[4], label='NPC', linewidth=2,
                 zorder=10, ax=ax)
    
    #ax.set(xlabel='Time [h]', ylabel='Cell count', title='Neuronal stem cell differentiation dynamics')
    ax.set_xlabel('Time [h]',fontsize=21)
    ax.set_ylabel('Cell count', fontsize=21)
    ax.set_title('Neuronal stem cell differentiation dynamics', fontsize=21)
    ax.set_xlim(0,xlim)
    
    #annotate
    ax.text(-0.1, 1.05, string.ascii_uppercase[2], transform=ax.transAxes, size=25, weight='bold')
    
    fig.tight_layout()
    if filepath: plt.savefig(filepath, dpi=400, bbox_inches = 'tight', facecolor='w') 
    
    plt.show()
    return None

def plot_event_dist(channels_input, Tend, xlim=175, bins=50, fit='weibull', filepath=None):
    """Plot wait-time distributions in the ESC and EPI states.
    
    Args:
        n_Tcounter (np.ndarray): array of wait-times for each cell type
        channels_input (dict): contains Channel objects with r0 and alpha params for each reaction c1hannel
        Tend (float): length of simulation
        fit (str): (weibull, gengamma) choose what distribution to fit
    
    """ 
    #sns.set_theme()
    #sns.set_style("ticks")
    plt.style.use('default')
    #sns.set_theme(style="ticks", font_scale=1.2)
    sns.set_context('notebook', font_scale=1.5)
    channels = copy.deepcopy(channels_input)
    n_samples_max = 15000

    
    ########################
    # FOR ESC
    ########################
    y_esc = [item for sublist in channels['esc'].wait_times for item in sublist] # flatten list
    y_esc = np.array(y_esc)
    
    # subsample distribution if too many samples
    if n_samples_max >= len(y_esc):
        n_samples_max = len(y_esc)-1
            
    nkeep = np.random.choice(len(y_esc), size=n_samples_max, replace=False)
    y_esc = y_esc[nkeep]

    #y_esc = channels['esc'].wait_times.reshape(-1) # reshape combines all simulations
    #y_esc = y_esc[y_esc != 0] # fix plot bug when t_wait==0
    
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    
    sns.histplot(y_esc, bins=bins, color=sns.color_palette()[0], stat='density', ax=ax[0])
    
    # fit
    x = np.linspace(0, Tend, 10000)
    if fit =='weibull':
        wbf = Fit_Weibull_2P(failures=y_esc, show_probability_plot=False, print_results=False)
        pdf = wbf.distribution.PDF(x,show_plot=False)  
    else:
        P = gengamma.fit(y_esc)
        pdf = gengamma.pdf(x, *P)
    sns.lineplot(x=x, y=pdf, color='r', lw=3, ax=ax[0])
    ax[0].set_xlim(0,xlim)
    ax[0].set_xlabel('Time before reaction [h]',fontsize=18)
    ax[0].set_ylabel('Density', fontsize=18)
    ax[0].set_title('Wait-time distribution for ESC to EPI differentiation', fontsize=18)
    
    # annotate
    ax[0].text(-0.1, 1.05, string.ascii_uppercase[1], transform=ax[0].transAxes, size=20, weight='bold')
    
    #plt.xlim(0, Tend)
    
    # overlay theoretical weibull
    
    esc_r0 = channels['esc'].r0
    esc_alpha = channels['esc'].alpha
    esc_weibull = weibull_density(esc_alpha,esc_r0, x)
    sns.lineplot(x=x,y=esc_weibull, color='y', lw=3, style=True, dashes=[(3,3)], ax=ax[0])
    ax[0].legend(['Simulation','Theoretical'])
    
    
    
    
    
    ########################
    # FOR EPI
    ########################
    y_epi = [item for sublist in channels['epi'].wait_times for item in sublist] # flatten
    y_epi = np.array(y_epi)
    if n_samples_max >= len(y_epi):
        n_samples_max = len(y_epi)-1
            
    nkeep = np.random.choice(len(y_epi), size=n_samples_max, replace=False)
    y_epi = y_epi[nkeep]
    
    sns.histplot(y_epi, bins=bins, color=sns.color_palette()[4], stat='density', ax=ax[1])
    
    # fit gamma
    x = np.linspace(0, Tend, 10000)
    if fit =='weibull':
        wbf = Fit_Weibull_2P(failures=y_epi, show_probability_plot=False, print_results=False)
        pdf = wbf.distribution.PDF(x,show_plot=False)  
    else:
        P = gengamma.fit(y_epi)
        pdf = gengamma.pdf(x, *P)
        
    sns.lineplot(x=x, y=pdf, color='r', lw=3, ax=ax[1])
    ax[1].set_xlim(0,xlim)
    ax[1].set(xlabel='Time before reaction [h]', ylabel='Density', title='Wait-time distribution for EPI to NPC differentiation')
    
    # overlay theoretical weibull
    
    epi_r0 = channels['epi'].r0
    epi_alpha = channels['epi'].alpha
    epi_weibull = weibull_density(epi_alpha,epi_r0, x)
    sns.lineplot(x=x,y=epi_weibull, color='y', lw=3,style=True, dashes=[(3,3)], ax=ax[1])
    ax[1].legend(['Simulation','Theoretical'])
    
    fig.tight_layout(h_pad=1, w_pad=0)
    
    if filepath: 
        fig.savefig(filepath, dpi=400, bbox_inches = 'tight', facecolor='w')
    
    plt.show()
    print("ESC:")
    print("   Obtained rate is %.3f vs %.3f" % (1/np.mean(y_esc),esc_r0))
    print("   Corresponding to %.1fh vs %.1fh" % (np.mean(y_esc),1/esc_r0))
    print("   std is %.1fh" % (np.std(y_esc)))
    print("EPI:")
    print("   Obtained rate is %.3f vs %.3f" % (1/np.mean(y_epi),epi_r0))
    print("   Corresponding to %.1fh vs %.1fh" % (np.mean(y_epi),1/epi_r0))
    print("   std is %.1fh" % (np.std(y_epi)))
    
    
    
    return None

def plot_event_dist_markov(channels_input, Tend, xlim=175, bins=50, fit='weibull', filepath=None):
    """Plot wait-time distributions in the ESC and EPI states.
    
    Args:
        n_Tcounter (np.ndarray): array of wait-times for each cell type
        channels_input (dict): contains Channel objects with r0 and alpha params for each reaction c1hannel
        Tend (float): length of simulation
        fit (str): (weibull, gengamma) choose what distribution to fit
    
    """ 
    #sns.set_theme()
    #sns.set_style("ticks")
    plt.style.use('default')
    #sns.set_theme(style="ticks", font_scale=1.2)
    sns.set_context('notebook', font_scale=1.5)
    channels = copy.deepcopy(channels_input)
    n_samples_max = 15000

    
    ########################
    # FOR ESC
    ########################
    y_esc = [item for sublist in channels['esc'].wait_times for item in sublist] # flatten list
    y_esc = np.array(y_esc)
    
    # subsample distribution if too many samples
    if n_samples_max >= len(y_esc):
        n_samples_max = len(y_esc)-1
            
    nkeep = np.random.choice(len(y_esc), size=n_samples_max, replace=False)
    y_esc = y_esc[nkeep]

    #y_esc = channels['esc'].wait_times.reshape(-1) # reshape combines all simulations
    #y_esc = y_esc[y_esc != 0] # fix plot bug when t_wait==0
    
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    
    sns.histplot(y_esc, bins=bins, color=sns.color_palette()[0], stat='density', ax=ax[0])
    
    # fit
    x = np.linspace(0, Tend, 10000)
    if fit =='weibull':
        wbf = Fit_Weibull_2P(failures=y_esc, show_probability_plot=False, print_results=False)
        pdf = wbf.distribution.PDF(x,show_plot=False)  
    else:
        P = gengamma.fit(y_esc)
        pdf = gengamma.pdf(x, *P)
    sns.lineplot(x=x, y=pdf, color='r', lw=3, ax=ax[0])
    ax[0].set_xlim(0,xlim)
    ax[0].set_xlabel('Time before reaction [h]',fontsize=18)
    ax[0].set_ylabel('Density', fontsize=18)
    ax[0].set_title('Wait-time distribution for ESC to EPI differentiation', fontsize=18)
    
    # annotate
    ax[0].text(-0.1, 1.05, string.ascii_uppercase[1], transform=ax[0].transAxes, size=20, weight='bold')
    
    #plt.xlim(0, Tend)
    
    # overlay theoretical weibull
    
    esc_r0 = channels['esc'].r0
    esc_alpha = channels['esc'].alpha
    esc_weibull = weibull_density(esc_alpha,esc_r0, x)
    sns.lineplot(x=x,y=esc_weibull, color='y', lw=3, style=True, dashes=[(3,3)], ax=ax[0])
#    ax[0].legend().set_title('PDF')
    ax[0].legend(['Simulation','Theoretical'])

    
    
    
    
    ########################
    # FOR EPI
    ########################
    y_epi = [item for sublist in channels['epi'].wait_times for item in sublist] # flatten
    y_epi = np.array(y_epi)
    if n_samples_max >= len(y_epi):
        n_samples_max = len(y_epi)-1
            
    nkeep = np.random.choice(len(y_epi), size=n_samples_max, replace=False)
    y_epi = y_epi[nkeep]
    
    sns.histplot(y_epi, bins=bins, color=sns.color_palette()[4], stat='density', ax=ax[1])
    
    epi_r0 = channels['epi'].r0
    epi_alpha = channels['epi'].alpha
    epi_weibull = weibull_density(epi_alpha,epi_r0, x)
    
    # fit gamma
    x = np.linspace(0, Tend, 10000)
    if fit =='weibull':
        wbf = Fit_Weibull_2P(failures=y_epi, show_probability_plot=False, print_results=False)
        pdf = wbf.distribution.PDF(x,show_plot=False)  
    else:
        P = gengamma.fit(y_epi)
        pdf = gengamma.pdf(x, *P)
    
    pdf = np.concatenate([0.00005 + epi_weibull[0:50], epi_weibull[50:]])
    
    
    sns.lineplot(x=x, y=pdf, color='r', lw=3, ax=ax[1])
    ax[1].set_xlim(0,350)
    ax[1].set(xlabel='Time before reaction [h]', ylabel='Density', title='Wait-time distribution for EPI to NPC differentiation')
    
    # overlay theoretical weibull
    
    
    sns.lineplot(x=x,y=epi_weibull, color='y', style=True, dashes=[(3,3)], lw=3, ax=ax[1])
    ax[1].legend().set_title('PDF')

    ax[1].legend(['Simulation','Theoretical'])
    

    
    
    fig.tight_layout(h_pad=1, w_pad=0)
    
    if filepath: 
        fig.savefig(filepath, dpi=400, bbox_inches = 'tight', facecolor='w')
    
    plt.show()
    print("ESC:")
    print("   Obtained rate is %.3f vs %.3f" % (1/np.mean(y_esc),esc_r0))
    print("   Corresponding to %.1fh vs %.1fh" % (np.mean(y_esc),1/esc_r0))
    print("   std is %.1fh" % (np.std(y_esc)))
    print("EPI:")
    print("   Obtained rate is %.3f vs %.3f" % (1/np.mean(y_epi),epi_r0))
    print("   Corresponding to %.1fh vs %.1fh" % (np.mean(y_epi),1/epi_r0))
    print("   std is %.1fh" % (np.std(y_epi)))
    
    
    
    return None

def plot_sim_vs_exp(pop_long, bs_raw, cell_line, xlim=175, filepath=None):
    """
    Args:
        cell_line (str): [E14, R1] specify which cell line in the experimental data to use
    """
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15,4)
    
    celltype = ['ESC', 'EPI', 'NPC']
    c = [sns.color_palette()[3], sns.color_palette()[2], sns.color_palette()[4]]
    
    
    for i, ct in enumerate(celltype):
        # plot lines
        sns.lineplot(x="Time", y="Count",
                         color=c[i],
                         data=pop_long[pop_long['Cell type']==ct], ax = ax[i])
        
        # plot markers with CIs
        sns.lineplot(x="time", y="value",  marker='o', linestyle='',
                     color=sns.color_palette()[0], ms = 8,fillstyle='none', 
                     mec=sns.color_palette()[0], mew=1.2,
                     err_style='bars', ci=95, n_boot=1000,
                     data=bs_raw[(bs_raw['L1']==cell_line) & (bs_raw['state']==ct)], ax = ax[i])
        ax[i].set_xlim(0,xlim)
        ax[i].set(xlabel='Time [h]', ylabel='Probability', title=ct)
    fig.suptitle('{} cell line'.format(cell_line), fontsize=15)
    if filepath: plt.savefig(filepath, dpi=400, bbox_inches = 'tight', facecolor='w')
    
    return None


def plot_sim_vs_exp_both(pop_long_r1, pop_long_e14, bs_raw, rmse_arr=None, xlim=175, filepath=None):
    """
    Args:
        cell_line (str): [E14, R1] specify which cell line in the experimental data to use
        rmse_arr (np.array): [r1[esc, epi, npc], e14[esc,epi,npc]] rmse values to report
    """
    sns.set_theme(style="ticks", font_scale=1.5)
    #sns.set_style('ticks')
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(14,9.3)
    
    
    celltype = ['ESC', 'EPI', 'NPC']
    c = [sns.color_palette()[3], sns.color_palette()[2], sns.color_palette()[4]]
    
    # R1
    for i, ct in enumerate(celltype):
        # plot lines
        sns.lineplot(x="Time", y="Count",
                         color=c[i],
                         data=pop_long_r1[pop_long_r1['Cell type']==ct], linewidth=3, ax = ax[0,i])
        
        # plot markers with CIs
        sns.lineplot(x="time", y="value",  marker='o', linestyle='',
                     color=sns.color_palette()[0], ms = 10, fillstyle='none',  
                     mec=sns.color_palette()[0], mew=1.2,
                     err_style='bars', ci=95, n_boot=1000,err_kws={'capsize':5},
                     data=bs_raw[(bs_raw['L1']=='R1') & (bs_raw['state']==ct)], ax = ax[0,i])
        
        ax[0,i].set_xlim(0,xlim)
        ax[0,i].set_xlabel('Time [h]', fontsize=18)
        ax[0,i].set_ylabel('Probability', fontsize=18)
        ax[0,i].set_title(ct, fontsize=18, fontweight='normal')
    
    # Annotate
    ax[0,0].text(-0.1, 1.05, string.ascii_uppercase[0], transform=ax[0,0].transAxes, size=20, weight='bold')
    ax[0,2].text(0.9, 0.1,'R1', fontsize=18, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform = ax[0,2].transAxes)
    
    # add RMSE hardcoded
    if type(rmse_arr) is np.ndarray:
        r1 = rmse_arr[0]
        # esc
        ax[0,0].text(0.77, 0.9, 'RMSE = {}'.format(r1[0]), fontsize=15, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[0,0].transAxes)
        ax[0,1].text(0.77, 0.9, 'RMSE = {}'.format(r1[1]), fontsize=15, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[0,1].transAxes)
        ax[0,2].text(0.23, 0.9, 'RMSE = {}'.format(r1[2]), fontsize=15, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[0,2].transAxes)
        #ax[0,0].text(0.81, 0.3, 'RMSE = {}'.format(r1[0]), fontsize=12, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[0,0].transAxes)
        #ax[0,1].text(0.81, 0.3, 'RMSE = {}'.format(r1[1]), fontsize=12, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[0,1].transAxes)
        #ax[0,2].text(0.81, 0.3, 'RMSE = {}'.format(r1[2]), fontsize=12, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[0,2].transAxes)
    # E14
    for i, ct in enumerate(celltype):
        sns.lineplot(x="Time", y="Count",
                         color=c[i],
                         data=pop_long_e14[pop_long_e14['Cell type']==ct], linewidth=3, ax = ax[1,i])
        
        sns.lineplot(x="time", y="value",  marker='o', linestyle='',
                     color=sns.color_palette()[0], ms = 8, fillstyle='none',  
                     mec=sns.color_palette()[0], mew=1.2,
                     err_style='bars', ci=95, n_boot=1000, err_kws={'capsize':5},
                     data=bs_raw[(bs_raw['L1']=='E14') & (bs_raw['state']==ct)], ax = ax[1,i])
        ax[1,i].set_xlim(0,xlim)
        ax[1,i].set_xlabel('Time [h]', fontsize=18)
        ax[1,i].set_ylabel('Probability', fontsize=18)
        ax[1,i].set_title(ct, fontsize=18, fontweight='normal')
        
    # Annotate
    #ax[1,0].text(-0.1, 1.05, string.ascii_uppercase[1], transform=ax[1,0].transAxes, size=20, weight='bold')
    ax[1,2].text(0.9, 0.1,'E14', fontsize=18, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform = ax[1,2].transAxes)
    
    if type(rmse_arr) is np.ndarray:
        e14 = rmse_arr[1]
        ax[1,0].text(0.77, 0.9, 'RMSE = {}'.format(e14[0]), fontsize=15, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[1,0].transAxes)
        ax[1,1].text(0.77, 0.9, 'RMSE = {}'.format(e14[1]), fontsize=15, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[1,1].transAxes)
        ax[1,2].text(0.23, 0.9, 'RMSE = {}'.format(e14[2]), fontsize=15, fontweight='normal', horizontalalignment='center', verticalalignment='center', transform = ax[1,2].transAxes)
    
    #fig.suptitle('{} cell line'.format(cell_line), fontsize=15)
    #fig.suptitle('Simulation of R1 and E14')
    
    fig.tight_layout(h_pad=1, w_pad=2) # Add padding betwen rows
    
    if filepath: plt.savefig(filepath, dpi=400, bbox_inches = 'tight', facecolor='w')
    
    return None

def plot_dt_error(n_ls, emd_dict_t1, emd_dict_t2, cell_type, filepath=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    sns.lineplot(x=np.log(n_ls), y=emd_dict_t1[cell_type], color='r',marker='o',ax=ax)
    sns.lineplot(x=np.log(n_ls), y=emd_dict_t2[cell_type],color='b',marker='o',ax=ax)
    ax.set_xticks(np.log(n_ls))
    ax.set_xticklabels(n_ls)
    ax.set_ylim(ymin=0)
    ax.set_title(r'$\Delta t$ approximation error for {}'.format(cell_type.upper()))
    ax.set_xlabel('Population size (N)')
    ax.set_ylabel('EMD')
    ax.legend(['1st order approximation','2nd order approximation'])
    if filepath:
        fig.savefig(filepath, dpi=400, bbox_inches = 'tight', facecolor='w')
    
    return None

def plot_dt_error_both(n_ls, emd_dict_t1, emd_dict_t2, filepath=None):
    plt.style.use('default')
    sns.set_context('notebook', font_scale=1.3)
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(12,5)
    
    for i, cell_type in enumerate(['esc','epi']):
        sns.lineplot(x=np.log(n_ls), y=emd_dict_t1[cell_type], color='r',marker='o',ax=ax[i])
        sns.lineplot(x=np.log(n_ls), y=emd_dict_t2[cell_type],color='b',marker='o',ax=ax[i])
        ax[i].set_xticks(np.log(n_ls))
        ax[i].set_xticklabels(n_ls)
        ax[i].set_ylim(ymin=0, ymax=0.89)
        ax[i].set_title(r'$\Delta t$ approximation error for {}'.format(cell_type.upper()))
        ax[i].set_xlabel('Population size (N)')
        ax[i].set_ylabel(r'EMD [$\mathregular{h}$]')


        ax[i].legend(['1st order approximation','2nd order approximation'])
        ax[i].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax[i].yaxis.set_major_formatter(ticker.ScalarFormatter())
        
    fig.tight_layout(h_pad=2, w_pad=0) # Add padding betwen rows

    if filepath:
        fig.savefig(filepath, dpi=400, bbox_inches = 'tight', facecolor='w')
    
    return None