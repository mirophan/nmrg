Non-Markovian Rejection Gillespie algorithm (NMRG)
==================================================
**Abstract** The Gillespie algorithm is commonly applied for simulating Poisson point processes, which follow an exponential waiting-time distribution, and are completely memoryless. However, stochastic processes governing biochemical interactions are empirically known to exhibit properties of memory, an inherently non-Markovian feature. The presence of such non-Markovian processes can significantly influence the course of a simulation.  In order to handle processes with memory, several extensions to the Gillespie algorithm have been proposed. However, these extensions are either limited by a high computational cost or are only applicable to a narrow selection of probability distributions found in biology. To challenge the aforementioned issues, a non-Markovian Rejection Gillespie algorithm is proposed in this thesis. The new algorithm is capable of generating simulations with non-exponential waiting-times, while remaining highly computationally efficient. It also incorporates the flexible Weibull distribution, allowing it to interpolate between a range of distributions relevant for biological processes. The proposed method was subsequently evaluated against an experimental dataset, for which it was able to correctly capture the underlying non-Markovian dynamics.

## Dependencies
`conda install -c conda-forge pandas numpy seaborn scipy pandas matplotlib ipython jupyterlab python=3.6`
`pip install dlib`

Note that Python 3.6 is forced in order to install `dlib` correctly. In addition, Python 3.6 is the earliest version compatible with this code, as it relies on the ordered implementation of the dictionary.

## Data availability
The mouse embryonic stem cell datased used for evaluating the model has been obtained from the following study by Stumpf and colleagues: [Stem Cell Differentiation as a Non-Markov Stochastic Process](https://www.sciencedirect.com/science/article/pii/S2405471217303423)

The data has been kindly communicated by the authors, and thus will not be uploaded to this repository. Please refer to the study above, should you wish to acquire the dataset.
