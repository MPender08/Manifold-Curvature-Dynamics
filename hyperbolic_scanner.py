"""
Logic as a Hyperbolic Actuator (v5.1): The Causal Scanner
Method: Forman-Ricci Curvature on Directed Autoregressive Manifolds
"""
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from GraphRicciCurvature.FormanRicci import FormanRicci

# 1. CONFIGURATION
MODEL_NAME = "gpt2" 
LAYER_IDX = 8       # Deep reasoning layer

# 2. STRICTLY LENGTH-MATCHED PROMPTS (13 Tokens Each)
prompt_base = "The tall man walked his big dog across the green city park."
prompt_logic = "If Alice is older than Bob, then Alice is the oldest."

def measure_causal_curvature(model, tokenizer, text):
    """
    Extracts strictly causal (directed) attention graphs and computes 
    Forman-Ricci curvature to measure topological hierarchy across the LCC.
    """
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    attentions = outputs.attentions[LAYER_IDX][0].cpu().numpy()
    n_heads, seq_len, _ = attentions.shape
    
    # Print exact token lengths to ensure the confounder is eliminated
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    print(f"Sequence Length: {seq_len} tokens")
    
    head_curvatures = []
    
    for h in range(n_heads):
        attn_h = attentions[h]
        G = nx.DiGraph()
        
        # 3. BUILD THE DIRECTED ACYCLIC GRAPH (DAG)
        for i in range(1, seq_len):
            row_sum = np.sum(attn_h[i, 1:i]) 
            
            if row_sum == 0:
                continue
                
            for j in range(1, i):  
                p = attn_h[i, j] / row_sum  
                
                if p > 0.01: 
                    # THE FIX: Add a microscopic epsilon to prevent distance collapse
                    w_dist = -np.log(p) + 1e-5 
                    
                    G.add_edge(j, i, weight=w_dist)

        # 4. ISOLATE LCC & CALCULATE FORMAN-RICCI
        try:
            if len(G.nodes) > 0:
                lcc_nodes = max(nx.weakly_connected_components(G), key=len)
                G_lcc = G.subgraph(lcc_nodes).copy()
                
                frc = FormanRicci(G_lcc)
                frc.compute_ricci_curvature()
                
                ks = [d['formanCurvature'] for u, v, d in frc.G.edges(data=True)]
                avg_k = np.mean(ks) if ks else np.nan
                head_curvatures.append(avg_k)
            else:
                head_curvatures.append(np.nan)
                
        except Exception as e:
            print(f"Error computing curvature for Head {h}: {e}")
            head_curvatures.append(np.nan)

    return head_curvatures

def main():
    print(f"--- INITIALIZING {MODEL_NAME} (v5.1 Causal Architecture) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    print(f"\n--- Processing Baseline (Control) ---")
    curv_base = measure_causal_curvature(model, tokenizer, prompt_base)

    print(f"\n--- Processing Logic (Variable) ---")
    curv_logic = measure_causal_curvature(model, tokenizer, prompt_logic)

    # Visualization
    print("\n--- PLOTTING MACROSCOPIC PHASE TRANSITION ---")
    plt.figure(figsize=(12, 6))
    heads = range(len(curv_base))
    
    plt.plot(heads, curv_base, 'bo--', label='Baseline (Stochastic Flow)', alpha=0.6)
    plt.plot(heads, curv_logic, 'rs-', label='Logic (Hierarchical Flow)', alpha=0.8, linewidth=2)
    
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.title("Discrete Forman-Ricci Curvature ($\kappa$) of Causal Attention Manifolds (Layer 8)")
    plt.xlabel("Attention Head")
    plt.ylabel("Macroscopic Curvature ($\kappa$)")
    plt.xticks(heads)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("fig_1.png", dpi=300)
    print("Plot saved as fig_1.png")

if __name__ == "__main__":
    main()