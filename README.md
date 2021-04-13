Rejection Gillespie for non-Markovian Reactions (REGINR)
==================================================
**Abstract.** The Gillespie algorithm is commonly applied for simulating memoryless processes that follow an exponential waiting-time. However, stochastic processes governing biological interactions, such as cell apoptosis and epidemic spreading, are empirically known to exhibit properties of memory, an inherently non-Markovian feature. The presence of such non-Markovian processes can significantly influence the outcome of a simulation. While several extensions to the Gillespie algorithm have been proposed, most of them suffer from either a high computational cost, or are only applicable to a narrow selection of probability distributions that do not match the experimentally observed biological data distributions. To tackle the aforementioned issues, we developed a Rejection Gillespie for non-Markovian Reactions (REGINR) that is capable of generating simulations with non-exponential waiting-times, while remaining an order of magnitude faster than alternative approaches. REGINR uses the Weibull distribution, which interpolates between the exponential, normal, and heavy-tailed distributions. We applied our algorithm to a mouse stem cell dataset with known non-Markovian dynamics and found it to faithfully recapitulate the underlying biological processes. We conclude that our algorithm is suitable for gaining insight into the role of molecular memory in stochastic models, as well as for accurately simulating real-world biological processes.

## Dependencies
```bash
conda install -c conda-forge pandas numpy seaborn scipy pandas matplotlib ipython jupyterlab python=3.6
pip install dlib
```

Note that Python 3.6 is forced in order to install `dlib` correctly. In addition, Python 3.6 is the earliest version compatible with this code, as it relies on the ordered implementation of the dictionary.

## Data availability
The mouse embryonic stem cell datased used for evaluating the model has been obtained from the following study by Stumpf and colleagues: [Stem Cell Differentiation as a Non-Markov Stochastic Process](https://www.sciencedirect.com/science/article/pii/S2405471217303423)

The data has been kindly communicated by the authors, and thus will not be uploaded to this repository. Please refer to the study above, should you wish to acquire the dataset.
