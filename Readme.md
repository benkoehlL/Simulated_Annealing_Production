# Description

This repository contains the code for the optimisation of a one-stage machine allocation and sequencing (job-shop) problem by the simulated annealing method.
I used the Metropolis-Hastings algorithm for the state update and Hamiltonian with symmetry-broken structure with respect to tardiness and earliness. 

## Installation Python Code

Instantiate a virtual environment in the current folder via: 
```
python3 -m venv venv
```
Now, install all requirements via
```
venv/bin/python -m pip install numpy tqdm 
```
You can run the script via
```
venv/bin/python anneal.py
```