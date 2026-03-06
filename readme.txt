========================================================================
Code Repository for: "Logic as a Hyperbolic Actuator: Evidence for 
VIP-Mediated Phase Transitions in Transformer Attention Manifolds"
Author: Matthew A. Pender
Version: 3.0
========================================================================

DESCRIPTION
This repository contains the Python scripts required to reproduce the 
topological phase transition experiments detailed in the accompanying 
manuscript. These scripts extract the internal attention matrices of a 
Transformer model (GPT-2) and evaluate them as strictly directed causal 
graphs to measure the geometric cost of logical reasoning. 

FILES INCLUDED
1. hyperbolic_scanner.py
2. hyperbolic_visualizer.py

DEPENDENCIES
To run these scripts, you will need a Python environment (3.8+) with the 
following packages installed:
- torch
- numpy
- networkx
- matplotlib
- transformers
- GraphRicciCurvature (pip install GraphRicciCurvature)

========================================================================
1. hyperbolic_scanner.py
========================================================================
This is the primary experimental script. It evaluates the macroscopic 
discrete Forman-Ricci curvature of the attention manifold across Layer 8.

Key Methodological Implementations:
- Strictly Length-Matched Prompts: Both the baseline and logic prompts 
  are exactly 13 tokens long after BPE tokenization to eliminate sequence 
  length as a confounding variable. 
- Sink Eradication: Index 0 is explicitly sliced from the matrix to 
  prevent the "Star Graph" artifact caused by attention sink dumping. 
- DAG Integrity: The graph is built as a strictly Directed Acyclic 
  Graph (DAG) by explicitly banning self-loops (j = i) and routing 
  information strictly from past tokens to present tokens. 
- Distance Inversion: Attention probabilities are converted to 
  geometric structural distances using the formula: w_dist = -log(p) + epsilon. 
  A microscopic epsilon (1e-5) is added to prevent zero-distance math 
  crashes in the solver when attention becomes 100% deterministic.


Execution: 
Run `python hyperbolic_scanner.py` from your terminal. 

Note on Math Solver Stability:
Depending on your hardware and NumPy backend, the Forman-Ricci solver 
can sometimes hang or crash due to OpenMP/MKL thread contention. If you 
experience infinite hanging during the "Processing" steps, restrict the 
math libraries to a single thread before running:

For Windows (PowerShell):
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python hyperbolic_scanner.py

For Mac/Linux:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python hyperbolic_scanner.py


Output: 
The script will print the topological values to the console and generate 
`fig_1.png`, mapping the structural phase transitions across all heads.

========================================================================
2. hyperbolic_visualizer.py
========================================================================
This script generates a physical 2D projection of the attention manifold 
for a specific head (default is Layer 8, Head 4) using the Kamada-Kawai 
force-directed layout. 

It uses the exact same 1% sparsity filter (p > 0.01) and -log(p) 
distance mathematics as the scanner. This visualizer physically 
demonstrates how the network sheds diffuse stochastic connections 
to form sparse, low-latency hierarchical trees under logical load. 

Execution:
Run `python hyperbolic_visualizer.py` from your terminal. 

Note: To switch between the Euclidean baseline visualization and the 
Hyperbolic logic visualization, simply comment/uncomment the desired 
`PROMPT` variable at the top of the script and run it again.