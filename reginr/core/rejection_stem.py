import numpy as np
import pandas as pd
from operator import attrgetter
from numpy.random import Generator, PCG64
import copy
import sys
import uuid
import timeit
import time
import math
from scipy.special import gamma
from random import SystemRandom
random = SystemRandom()

#from rejection_gillespie.plot.utils_plot import plot_results



rng = Generator(PCG64())



class Cell():
    """Class to track individual cells and their properties
    
    Attributes:
        t_start (float): Timestamp at which the cell was initialised. Used for tracking 
            reaction waiting times (t_wait = t_current - t_start)
        
    
    """
    def __init__(self, t_start):
        # TODO init with tstart and channel_list, then for each channel create t_start attr
        self.t_start = t_start
        self.t_start_esc = self.t_start_epi = t_start

        self.tmax_esc = self.tmax_epi  = False
        self.id = gen_uuid()

class Channel():
    """
    Attributes:
        rmax (float): current upper bound for the reaction rate of the channel,
            determined by the cell with the highest r(t)=\beta * t^\alpha
        cell_id (str): id of the cell object with the highest rmax
    """
    # TODO use property at init to create a new entry for channel if not exists, then call dict in def sorted_channel_dict
    tstart_dict = {}
    
    def __init__(self, name, r0, alpha):
        self.name = name
        self.tstart_name = 't_start_' + name
        self.r0 = r0
        self.rmax = r0
        self.rmax_d1 = r0
        self.cell_id = ''
        self.alpha = alpha
        self.wait_times = []


    # TODO change to static or class method
    def _rate_t(self, t, rate, alpha):
        """Calculates the reaction rate r(t_i) = \beta * t_i^\alpha"""
        if alpha >= 0:
            beta = (alpha + 1) * (rate * gamma((alpha + 2)/(alpha + 1)))**(alpha + 1) 
            return beta * np.power(t,alpha)
        else:
            raise ValueError('Incorrect value of alpha: {}'.format(alpha))
    
    def _rate_t_d1(self, t, rate, alpha):
        """Calculates the first order derivative of rate r(t_i)"""
        beta = (alpha + 1) * (rate * gamma((alpha + 2)/(alpha + 1)))**(alpha + 1)
        if alpha >= 0:
            return alpha * beta * np.power(t, alpha - 1)
        else:
            raise ValueError('Incorrect value of alpha: {}'.format(alpha))



    
    def update_rmax(self, channel_sorted, cell_id_reacted, t):
        """Update rmax and rmax_d1 for given channel
        
        Update only takes place if the value at the top of the channel heap had been changed, i.e. 
        the last cell that reacted had the hightest waiting time, leading to highest r(t_wait)
        
        note: cell_id_reacted not necessary in current implementation but might be 
        useful in cases where one cell is involved in multiple reaction channels simultaneously
        
        """
        if (channel_sorted and t!=0): # t==0 only happens at the start of the simulation
            cmax_id, cell_tstart = next(iter(channel_sorted.items()))
            tmax = t - cell_tstart
            
            rmax_new =  self._rate_t(tmax, self.r0, self.alpha)
            if rmax_new > self.r0:
                self.rmax = rmax_new
                self.rmax_d1 = self._rate_t_d1(tmax, self.r0, self.alpha)
            else:
                self.rmax = self.r0
            self.cell_id = cmax_id
            
            

def gen_tstart_heap(c_list, channel_key):
    """Creates a sorted dictionary from a list of Cell objects for given channel. 

       Sorting based on tstart (time since last reaction) in ASCENDING order -> sorted(reverse=False)

    Args:
        c_list (list) : list of Cell objects
        channel_key (str) : key for which tstart attribute of the Cell object to select 
            (for cases where a cell is involved in multiple reaction channels simultaneously)

    Returns:
        Dictionary of Cell objects sorted by the tstart value for the respective channel, in ascending order.

    dict_sorted = {k:v for k, v in sorted(c_dict.items(),
                                          key=lambda x : attrgetter(channel_key)(x[1]),
                                          reverse=False)}
    """

    dict_tmp = {cell.id : getattr(cell, channel_key) for cell in c_list}
    dict_sorted = dict(sorted(dict_tmp.items(), key=lambda item: item[1], reverse=False))
    return dict_sorted

def gen_uuid():
    """Generate a 32char hex uuid to use as a key for each cell in the sorted dictionary

    random = SystemRandom() # globally defined

    """
    return uuid.UUID(int=random.getrandbits(128),version=4).hex

def initialise_gc(t_start, n_esc, n_epi, n_npc):
    """Initialises the Germinal Center cell populations as dictionary of lists

    Args:
        t_start (int): t_start instance attribute for the ~:class:`~rejection_gillespie.Cell` class
        n_** (int) : initial population size for each cell type

    Returns:
        Dict containing a list of cell instances for each cell type

    """
    # TODO: change to *args
    # use dictionary instead?
    list_esc = [Cell(t_start) for i in range(n_esc)]
    list_epi = [Cell(t_start) for i in range(n_epi)]
    list_npc = [Cell(t_start) for i in range(n_npc)]
    d = {'esc' : list_esc, 'epi' : list_epi, 'npc' : list_npc}
    return d

def compute_propensities(gc_data, channels, d_order=0):
    """Calculates the propensities for each reaction channel
    
    Args:
        gc_data (dict): Germinal center data. Each cell type is represented by a separate dict of Cell objects
        channels (dict): Dictionary of all reaction channels in the GC
        d_order (int): choose which order derivative to calculate (0 or 1)
    
    Returns:
        numpy array of propensities for each reaction channel
    """

    propensity = []
    if d_order == 0:
        propensity.append(len(gc_data['esc'])   * channels['esc'].rmax) # ESC -> EPI
        propensity.append(len(gc_data['epi'])   * channels['epi'].rmax) # EPI -> NPC
    elif d_order == 1:
        propensity.append(len(gc_data['esc'])   * channels['esc'].rmax_d1) # ESC -> EPI
        propensity.append(len(gc_data['epi'])   * channels['epi'].rmax_d1) # EPI -> NPC
    else:
        raise ValueError('Incorrect order of derivative inserted: {}'.format(d_order))
    return np.array(propensity)




def rate_t(t,rate,alpha):
    beta = (alpha + 1) * (rate * gamma((alpha + 2)/(alpha + 1)))**(alpha + 1)
    return np.power(t,alpha)*beta


def popcell(Cell_list, index):
    """
    exchange values of the element we want to delete with the last element, and then use pop()
    (popping the element i becomes O(1) rather than O(n))
    """

    Cell_last = copy.deepcopy(Cell_list[-1])
    Cell_list[-1] = Cell_list[index]
    Cell_list[index] = Cell_last
    popped_cell = Cell_list.pop()
    return popped_cell

def perform_reaction(mu, GCdata, tstart_heap_dict, channels, t, wt_counter):
    print_time = False
    """
    Perform the reaction mu and modify the GC data accordingly
    
    Note about complexity:
        
        - append() operation is O(1)
        - When pop() is called from the end, the operation is O(1), while calling pop() from anywhere else is O(n)
          due to memory realocation. One trick to gain speed is to exchange values of the element you want to delete 
          with the last element, and then use pop().
    """
    # lists containing cell populations of each cell type
    esc_popn = GCdata['esc']
    epi_popn = GCdata['epi']
    npc_popn = GCdata['npc']
    
    start = timeit.default_timer()
    
    cell_id_reacted = ""
            
    if mu == 0: # migrate ESC -> EPI
        indexC = rng.integers(0, len(esc_popn))
        selected_c = esc_popn[indexC]
        t_esc = t - selected_c.t_start_esc

        # TODO change to static or class method
        resc_max = channels['esc'].rmax
        resc = channels['esc'].r0
        a_esc = channels['esc'].alpha
        if rate_t(t_esc, resc, a_esc) >= resc_max*rng.uniform(0,1):
            # Remove esc cell from esc population
            CellESC = popcell(esc_popn, indexC)
            CellESC.t_start_esc = t
            CellESC.t_start_epi = t
            # Migrate esc cell to epi population
            epi_popn.append(CellESC)
            
            # append wait time of cell to corresponding channel
            wt_counter['esc'].append(t_esc)

            # pop cell from esc heap
            esc_sorted = tstart_heap_dict.get('esc')
            esc_sorted.pop(CellESC.id)
            
            # add to bottom of epi heap
            epi_sorted = tstart_heap_dict.get('epi')
            epi_sorted[CellESC.id] = t
            
            # update rmax for esc channel
            cell_id_reacted = CellESC.id

        
    elif mu == 1: # migrate EPI -> NPC
        indexC = rng.integers(0, len(epi_popn))
        selected_c = epi_popn[indexC]
        t_epi = t - selected_c.t_start_epi

        # TODO change to static or class method
        repi_max = channels['epi'].rmax
        repi = channels['epi'].r0
        a_epi = channels['epi'].alpha
        if rate_t(t_epi, repi, a_epi) >= repi_max*rng.uniform(0,1):
            CellEPI = popcell(epi_popn, indexC)
            CellEPI.t_start_esc = t
            CellEPI.t_start_epi = t
            npc_popn.append(CellEPI)

            # append wait time of cell to corresponding channel
            wt_counter['epi'].append(t_epi)
            
            # pop cell from epi heap
            epi_sorted = tstart_heap_dict.get('epi')
            epi_sorted.pop(CellEPI.id)
            
            # update rmax for epi channel
            cell_id_reacted = CellEPI.id
        
    else:
        print(" Warning, wrong reaction chosen for mu = %s" % mu)
        
    end = timeit.default_timer()
    t_elapsed = end - start
    
    if print_time: print("%.0d mu_s elapsed for reaction %s" % (t_elapsed*1e6,mu))
    
    return cell_id_reacted

def do_gillespie(t0, t_end, gc_data_init, channels_init, timepoints, order_approx):
    
    gc_data = copy.deepcopy(gc_data_init)
    channels = copy.deepcopy(channels_init)
    
    t = t0
    ti = 0
    population = np.zeros((timepoints+1,len(gc_data)))
    n_Tcounter = [[] for i in range(2)]
    wt_counter = {}
    for k in channels.keys():
        wt_counter[k] = []
        
    # Set up sorted heaps
    # For each channel create a dictionary of cells sorted by tstart
    # As of python 3.7, dictionaries are officialy ordered.    
    esc_popn, epi_popn, npc_popn = gc_data.values()
    esc_sorted = gen_tstart_heap(c_list = esc_popn, channel_key='t_start_esc')
    epi_sorted = gen_tstart_heap(c_list = epi_popn, channel_key='t_start_epi')
    channel_heap_sorted = {'esc' : esc_sorted, 'epi' : epi_sorted}
    
    # Track which cell had last reacted -> used for updating rmax
    cell_id_reacted = ""
    
    # Track dt error
    t2_error = []
    
    # check case where all alphas == 0 
    simplify_t1 = False
    for k in channels:
        if channels[k].alpha <= 0:
            simplify_t1 = True
    
    while t < t_end:

        # TODO use .get instead of .values()
        esc_popn, epi_popn, npc_popn = gc_data.values()
        
        esc = channels['esc']
        esc.update_rmax(esc_sorted, cell_id_reacted, t)
        epi = channels['epi']
        epi.update_rmax(epi_sorted, cell_id_reacted, t)
        
        propensities = compute_propensities(gc_data, channels, d_order = 0)
        a0 = np.sum(propensities)
        
        if order_approx == 2:
            d_propensities = compute_propensities(gc_data, channels, d_order = 1)
            d_a0 = np.sum(d_propensities)
        
        if t >= ti*t_end/timepoints:
            population[ti,:] = np.array([len(esc_popn),len(epi_popn), len(npc_popn)])
            ti += 1
        
        # If propensities are zero, quickly end the simulation
        if (a0 == 0 and t!=0):
            t += t_end/timepoints/1.2
            
        # If the number of cells is too high, quickly end the simulation
        elif sum([len(l) for l in gc_data.values()]) > 30000:
            t += t_end/timepoints/1.2
        
        else:
            # Time step
            u1 = rng.uniform(0,1)
            # 1st order approximation
            
            
            # 2nd order approximation
            
            
            if order_approx == 1:
                dt1 = np.log(1/u1)/a0
                t += dt1
            elif order_approx == 2:
                # if all reaction alphas == 0, calculation simplifies to exponential -> use 1st order
                if simplify_t1:
                    dt1 = np.log(1/u1)/a0
                    t+=dt1
                else:
                    dt2 = (- a0 + np.sqrt(np.power(a0, 2) - 2 * d_a0 * np.log(u1))) / (d_a0)
                    t += dt2
            else:
                raise ValueError('Wrong order approximation requested: [{}]. Can only input (int) 1 or 2 corresponding to 1st or 2nd order.'.format(order_approx))
            #t2_error.append([dt1, dt2]) # track \Delta t throughout simulation
            
            # Select which reaction should fire
            # update rmax with \Delta t
            esc.update_rmax(esc_sorted, cell_id_reacted, t)
            epi.update_rmax(epi_sorted, cell_id_reacted, t)
            # recalculate propensities
            propensities = compute_propensities(gc_data, channels, d_order = 0)
            a0 = np.sum(propensities)
            
            u2 = rng.uniform(0,1)
            mu = 0
            p_sum = 0.0
            while True:
                p_sum += propensities[mu]
                mu += 1
                if p_sum > u2 * a0:
                    break            
            mu = mu - 1
            
            # Perform reaction
            # cell_id_reacted currently not used
            cell_id_reacted = perform_reaction(mu,gc_data, channel_heap_sorted, channels, t, wt_counter)
                 
    # Fill out empty tail of population array by last value from simulation
    if ti < len(population):
        population[ti:] = population[ti-1]
    
    # Record final population state
    if t >= ti*t_end/timepoints:
            population[ti,:] = np.array([len(esc_popn),len(epi_popn), len(npc_popn)])
            ti += 1
    
    return population, wt_counter, np.array(t2_error)


def simulate(t_end, n_sim, timepoints, channels_init, n_esc=100, order_approx=1, verbose=True):
    """Run Rejection Gillespie simulation
    
    Args:
        order_approx(int): {1:'1st', 2:'2nd'} choose what order taylor approximation for \Delta t to use
    
    """
    channels = copy.deepcopy(channels_init)
    
    t0 = 0 # simulation always starts at 0
    
    # Initialise population counts (n_esc already set in function params)
    n_epi = 0
    n_npc = 0
    N_GC_init = [n_esc, n_epi, n_npc]

    gc_data_init = initialise_gc(t0, n_esc, n_epi, n_npc)
    
    population = np.empty((n_sim, timepoints+1, len(N_GC_init)))
    
    wt_dict = {}
    for k in channels:
        wt_dict[k] = []
    
    total_runtime = 0
    for i in range(n_sim):
        print('Running simulation n={}'.format(i)) if verbose else None
        start = time.time()
        
        population[i,:,:], wt_counter, t2_error = do_gillespie(t0, t_end, gc_data_init, channels, timepoints, order_approx)
        
        # append recorded wait-times of the simmulation
        for k, v in wt_counter.items():
            wt_dict[k].append(v)
        
        end = time.time()
        t_elapsed = end - start
        total_runtime += t_elapsed
    
        print("%.0ds elapsed for simulation %s" % (t_elapsed,i) ) if verbose else None
    
    
    for k in channels:
            channels[k].wait_times = wt_dict[k]
    
    print("Simulation finished in {:.2f} min ({:.0f}s)".format(total_runtime / 60, total_runtime) )if verbose else None
    return population, channels, t2_error


