# Logic as a Hyperbolic Actuator
#### Evidence for Scale-Invariant Manifold Warping in Large Language Models

![manifold_corrected_results](manifold_corrected_results.png)

## Overview

This project characterizes the geometric transitions occurring within the transformer latent space during hierarchical synthesis. Utilizing a Local Pairwise Framework (mean step distance dˉxy​=0.0324), we identify a non-linear phase transition from near-Euclidean flatness into extreme hyperbolic regimes. This transition facilitates Geodesic Efficiency, allowing the model to navigate dense logical hierarchies without a collapse in semantic throughput.

## Core Discoveries

- Hyperbolic Phase Transition: Quantitative mapping of discrete Ollivier-Ricci curvature (κ) shifting from a near-flat baseline (κ≈−1.5) to deep hyperbolic states (κ≈−95.0) as logical gating density increases.

- Geodesic Efficiency: Discovery of a stable, positively modulated velocity profile (r=0.5636). This refutes the "Syntactic Wall" hypothesis by demonstrating that hyperbolic warping serves as a structural optimization for informational transport.

- Topological Separation: Clear causal separation between logical semantic gradients and stochastic null models, identifying hierarchical logic as a specific actuator for manifold warping.

- Structural Isomorphism: Establishes a functional link between biological inhibitory gating (SST-interneurons) and symbolic manifold modulation in artificial architectures.

## Methodology

Instead of global radial measurements which are prone to coordinate bias, this audit utilizes Incremental Concept Velocity (Vinc​) and local pairwise curvature. By measuring the geometric "torque" between adjacent logical states, we isolate the impact of hierarchical constraints on the manifold's intrinsic topology.

## Installation & Usage

Environment Setup:
```
pip install torch numpy POT pandas matplotlib scipy transformers tqdm
```
Generate Experimental Data:
```
python generate_experiment_plots.py
```

## Research Applications

- Mechanistic Interpretability: Visualizing how models "funnel" latent states into focused, high-curvature corridors during complex reasoning.

- AI Safety & Hallucination Research: Identifying geometric precursors to logical failure, where the manifold fails to transition into a sufficiently deep hyperbolic state.

- Neuromorphic Engineering: Applying Curvature Adaptation (CA) principles to energy-efficient hardware design.
