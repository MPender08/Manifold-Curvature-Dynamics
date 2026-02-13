import torch
import numpy as np
import ot
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist, cosine
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- MATHEMATICAL FRAMEWORK ---
class SemanticTopology:
    def __init__(self, model_name="gpt2"):
        print(f"Loading Manifold: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
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
            top_probs = (top_probs / top_probs.sum()).numpy().astype(np.float64)
            support_vectors = self.embed_matrix[top_indices.numpy()]
            z_cm = np.average(support_vectors, axis=0, weights=top_probs)
            z_cm = z_cm / (np.linalg.norm(z_cm) + 1e-12)
            return {'center': z_cm, 'probs': top_probs, 'support': support_vectors, 'tokens': token_count}

    def compute_kappa(self, state_ref, state_target):
        d_xy = cosine(state_ref['center'], state_target['center'])
        if d_xy < 1e-7: return 0
        M = cdist(state_ref['support'], state_target['support'], metric='cosine').astype(np.float64)
        w1 = ot.emd2(state_ref['probs'], state_target['probs'], M)
        return 1 - (w1 / d_xy)

# --- EXPERIMENT SETUP ---
def generate_gradient_prompts():
    themes = ["Quantum mechanics", "Economic theory", "Baking a cake", "Ancient philosophy", "Jazz improvisation"]
    constraints = ["", "is a subject that", "can be defined as the following:", 
                   "follows a strict hierarchical structure where", 
                   "must be analyzed through the formal logical lens of", 
                   "in a first-order predicate logic syllogism, assuming P implies Q, then"]
    gradient_list = []
    for theme in themes:
        for i, c in enumerate(constraints):
            gradient_list.append({"theme": theme, "level": i, "prompt": f"{theme} {c}".strip()})
    return gradient_list

def generate_scrambled_controls(experiment_prompts, tokenizer, vocab_size):
    controls = []
    targets = [p for p in experiment_prompts if p['level'] == 5]
    for item in targets:
        length = tokenizer(item['prompt'], return_tensors='pt')['input_ids'].shape[1]
        rand_ids = torch.randint(0, vocab_size, (1, length))
        rand_text = tokenizer.decode(rand_ids[0], skip_special_tokens=True)
        controls.append({"theme": item['theme'], "level": 5, "prompt": rand_text, "type": "Control"})
    return controls

# --- EXECUTION ---
if __name__ == "__main__":
    topo = SemanticTopology("gpt2")
    ref_origin = topo.get_metrics("The")
    
    exp_cases = generate_gradient_prompts()
    ctrl_cases = generate_scrambled_controls(exp_cases, topo.tokenizer, topo.vocab_size)
    
    results = []
    print("\n[Running Hybrid Study...]")

    # Process Experimental Data (with last_state tracking for Incremental Velocity)
    for theme in tqdm(set(c['theme'] for c in exp_cases)):
        theme_steps = [c for c in exp_cases if c['theme'] == theme]
        last_state = None
        for step in theme_steps:
            curr_state = topo.get_metrics(step['prompt'])
            kappa = topo.compute_kappa(ref_origin, curr_state)
            
            # Incremental Velocity Calculation
            if last_state is not None:
                v_inc = cosine(last_state['center'], curr_state['center']) / (curr_state['tokens'] - last_state['tokens'])
            else:
                v_inc = cosine(ref_origin['center'], curr_state['center']) / curr_state['tokens']
                
            results.append({"Level": step['level'], "Type": "Experiment", "Kappa": kappa, "Velocity": v_inc})
            last_state = curr_state

    # Process Control Data
    for ctrl in tqdm(ctrl_cases):
        state = topo.get_metrics(ctrl['prompt'])
        kappa = topo.compute_kappa(ref_origin, state)
        v_inc = cosine(ref_origin['center'], state['center']) / state['tokens']
        results.append({"Level": 5, "Type": "Control", "Kappa": kappa, "Velocity": v_inc})

    df = pd.DataFrame(results)
    df_exp = df[df['Type'] == 'Experiment']
    df_ctrl = df[df['Type'] == 'Control']

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

    scatter = plt.scatter(df_exp['Kappa'], df_exp['Velocity'], c=df_exp['Level'], 
                          cmap='plasma', s=100, alpha=0.8, edgecolors='black', label='Semantic Gradient')
    plt.scatter(df_ctrl['Kappa'], df_ctrl['Velocity'], c='gray', marker='x', 
                s=100, linewidths=2, alpha=0.6, label='Null Model (Random)')

    plt.colorbar(scatter).set_label('Gating Density (Constraint Level)', fontsize=12)
    plt.title("The Syntactic Wall: Corrected Manifold Dynamics", fontsize=14, fontweight='bold')
    plt.xlabel(r"Curvature ($\kappa$) [Hyperbolic $\leftarrow \rightarrow$ Euclidean]", fontsize=12)
    plt.ylabel(r"Incremental Velocity ($V_{inc}$)", fontsize=12)
    
    # Complexity Penalty Trendline
    z = np.polyfit(df_exp['Kappa'], df_exp['Velocity'], 1)
    plt.plot(df_exp['Kappa'], np.poly1d(z)(df_exp['Kappa']), "r--", alpha=0.6, label=r"Complexity Penalty ($P_c$)")

    plt.legend()
    plt.savefig("syntactic_wall_final.pdf", bbox_inches='tight')
    plt.show()

    print(f"\nFinal Correlation (K vs Vinc): {df_exp['Kappa'].corr(df_exp['Velocity']):.4f}")