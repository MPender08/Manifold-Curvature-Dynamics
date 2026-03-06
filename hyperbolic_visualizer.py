"""
Logic as a Hyperbolic Actuator: Topological Visualizer (v5.1)
Maps the exact geometric distances (-log(p)) of the causal attention manifold.
"""
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIGURATION
MODEL_NAME = "gpt2"
HEAD_IDX = 4 
LAYER_IDX = 8 

PROMPT = "If Alice is older than Bob, then Alice is the oldest."
# PROMPT = "The tall man walked his big dog across the green city park."

def plot_topological_manifold():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attn = outputs.attentions[LAYER_IDX][0, HEAD_IDX].numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace('Ġ', ' ') for t in tokens]
    seq_len = len(clean_tokens)

    G = nx.DiGraph()
    
    # 1. Add nodes
    for i in range(1, seq_len):
        G.add_node(i, label=clean_tokens[i])

    # 2. Build the exact same rigorously filtered graph as v5.1 scanner
    for i in range(1, seq_len):
        row_probs = attn[i, 1:i]
        if len(row_probs) == 0:
            continue
            
        row_sum = np.sum(row_probs)
        if row_sum == 0:
            continue
            
        for j_idx, p_raw in enumerate(row_probs):
            p = p_raw / row_sum  # RENORMALIZE
            j = j_idx + 1
            
            if p > 0.01:  # 1% Sparsity Filter
                # TRUE GEOMETRIC DISTANCE
                dist = -np.log(p) + 1e-5
                # We store 'distance' for layout, and 'weight' for visual line thickness
                G.add_edge(j, i, weight=p, distance=dist)

    # 3. True Geometric Layout (Kamada-Kawai minimizes energy based on true path distances)
    # This fundamentally respects the -log(p) geometry we feed to Forman-Ricci.
    pos = nx.kamada_kawai_layout(G, weight='distance')

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw edges with thickness based on probability
    for u, v, d in G.edges(data=True):
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="->", color="crimson", 
                                  connectionstyle="arc3,rad=0.1", 
                                  alpha=min(d['weight'] * 3, 0.8), lw=d['weight'] * 5))

    # Draw nodes
    for i in range(1, seq_len):
        if i in pos:
            ax.scatter(pos[i][0], pos[i][1], s=800, color='white', edgecolors='navy', zorder=5, linewidths=1.5)
            ax.text(pos[i][0], pos[i][1], clean_tokens[i], 
                    ha='center', va='center', fontsize=9, fontweight='bold', zorder=6)

    plt.title(f"Attention Manifold Topology: Layer {LAYER_IDX}, Head {HEAD_IDX}\nEdges reflect normalized causal flow ($w_{{dist}} = -\log(p) + \epsilon$)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_topological_manifold()