import torch
import numpy as np
import ot
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist, cosine
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- MATHEMATICAL FRAMEWORK ---
class SemanticTopology:
    def __init__(self, model_name="gpt2"):
        print(f"Loading Manifold: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
        
        # Normalize the embedding matrix
        raw_embeds = self.model.transformer.wte.weight.detach()
        self.embed_matrix = torch.nn.functional.normalize(raw_embeds, p=2, dim=1).numpy()

    def get_metrics(self, text, top_k=50):
        inputs = self.tokenizer(text, return_tensors="pt")
        token_count = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = (top_probs / top_probs.sum()).numpy() 
            
            support_vectors = self.embed_matrix[top_indices.numpy()]
            
            # Eq 1: Center of Mass
            z_cm = np.average(support_vectors, axis=0, weights=top_probs)
            z_cm = z_cm / (np.linalg.norm(z_cm) + 1e-12)
            
            # Eq 3: Shannon Entropy (H)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
            
            return {
                "center": z_cm, 
                "probs": top_probs, 
                "support": support_vectors, 
                "entropy": entropy, 
                "tokens": token_count
            }

    def compute_kappa(self, state_ref, state_target):
        """Calculates Discrete Ricci Curvature (Eq 2)."""
        d_xy = cosine(state_ref['center'], state_target['center'])
        M = cdist(state_ref['support'], state_target['support'], metric='cosine')
        w1 = ot.emd2(state_ref['probs'], state_target['probs'], M)
        return 1 - (w1 / d_xy) if d_xy > 1e-7 else 0

# --- EXPERIMENT SETUP ---
def generate_gradient_prompts():
    """Generates an N-expansion across diverse thematic domains."""
    themes = ["The solar system", "A lonely cat", "Quantum mechanics", "Economic theory", "Baking a cake"]
    constraints = [
        "", # Level 0: Pure Associative
        "is a subject that", # Level 1: Low Gating
        "can be defined as the following:", # Level 2: Medium Gating
        "follows a strict hierarchical structure where", # Level 3: High Gating
        "must be analyzed through the formal logical lens of", # Level 4: Extreme Gating
        "in a first-order predicate logic syllogism, assuming P implies Q, then" # Level 5: Syntactic Wall
    ]
    
    gradient_list = []
    for theme in themes:
        for i, c in enumerate(constraints):
            gradient_list.append({"label": f"Level {i}", "prompt": f"{theme} {c}"})
    return gradient_list

def generate_scrambled_controls(standard_prompts, tokenizer, vocab_size):
    """
    Control Condition: 
    Extracts 'Level 5' prompts, measures length, and generates 
    random token sequences of identical length.
    """
    controls = []
    # Filter for the most complex prompts (Level 5) to match length complexity
    targets = [p for p in standard_prompts if "Level 5" in p['label']]
    
    print(f"Generating {len(targets)} control samples...")
    
    for item in targets:
        # Measure length of the logical prompt
        ids = tokenizer(item['prompt'], return_tensors='pt')['input_ids']
        length = ids.shape[1]
        
        # Generate random tokens (excluding special tokens roughly < 50256)
        rand_ids = torch.randint(0, vocab_size, (1, length))
        rand_text = tokenizer.decode(rand_ids[0], skip_special_tokens=True)
        
        controls.append({"label": "Control (Random)", "prompt": rand_text})
        
    return controls

# --- EXECUTION ---
if __name__ == "__main__":
    topo = SemanticTopology("gpt2")
    
    # Define Reference State (The "Origin")
    ref_state = topo.get_metrics("The") 
    
    # 1. Generate Experimental Data
    experiment_cases = generate_gradient_prompts()
    
    # 2. Generate Control Data (Null Model)
    control_cases = generate_scrambled_controls(experiment_cases, topo.tokenizer, topo.vocab_size)
    
    # Combine for processing
    all_cases = experiment_cases + control_cases

    print("\n[Processing N-Expansion Study...]")
    results = []
    
    for case in all_cases:
        state = topo.get_metrics(case['prompt'])
        kappa = topo.compute_kappa(ref_state, state)
        
        # Normalized Entropy Drop (Gamma)
        gamma = 1 - (state['entropy'] / ref_state['entropy'])
        
        # Velocity adjusted for length
        velocity = cosine(ref_state['center'], state['center']) / state['tokens']
        
        results.append({
            "Label": case['label'], 
            "Type": "Control" if "Control" in case['label'] else "Experiment",
            "Kappa": kappa, 
            "Velocity": velocity, 
            "Gating": gamma
        })

    df = pd.DataFrame(results)

    # --- VISUALIZATION: ACADEMIC GRADE PLOT ---
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

    # Split DataFrames
    df_exp = df[df['Type'] == 'Experiment']
    df_ctrl = df[df['Type'] == 'Control']

    # Plot 1: The Experimental Gradient (Colormapped circles)
    scatter = plt.scatter(df_exp['Kappa'], df_exp['Velocity'], c=df_exp['Gating'], 
                          cmap='viridis', s=120, alpha=0.8, edgecolors='black', linewidths=0.5,
                          label='Semantic Gradient')
    
    # Plot 2: The Random Controls (Gray X marks)
    plt.scatter(df_ctrl['Kappa'], df_ctrl['Velocity'], c='gray', marker='x', 
                s=100, linewidths=2, alpha=0.7, label='Null Model (Random)')

    cbar = plt.colorbar(scatter)
    # FIX: Added r'' to handle backslashes correctly
    cbar.set_label(r'Inhibitory Gating Density ($\gamma$)', fontsize=12)

    plt.title("Scale-Invariant Cognitive Dynamics: The Syntactic Wall", fontsize=14, fontweight='bold')
    
    # FIX: Added r"" to handle \rightarrow correctly
    plt.xlabel(r"Curvature ($\kappa$) [Deep Hyperbolic $\leftarrow \rightarrow$ Flat]", fontsize=12)
    plt.ylabel(r"Concept Velocity ($V_c$)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)

    # Add trendline for Complexity Penalty (only for Experiment data)
    z = np.polyfit(df_exp['Kappa'], df_exp['Velocity'], 1)
    p = np.poly1d(z)
    plt.plot(df_exp['Kappa'], p(df_exp['Kappa']), "r--", alpha=0.6, label=r"Complexity Penalty ($P_c$)")

    plt.legend(loc='upper right')

    # --- SAVE COMMANDS ---
    plt.savefig("syntactic_wall_controlled.pdf", bbox_inches='tight', format='pdf')
    plt.savefig("syntactic_wall_controlled.png", bbox_inches='tight', dpi=300)

    print("\n[Study Complete]")
    print(f"Experimental Correlation: {df_exp['Kappa'].corr(df_exp['Velocity']):.4f}")
    print("Files saved: 'syntactic_wall_controlled.pdf' and 'syntactic_wall_controlled.png'")
    
    plt.show()