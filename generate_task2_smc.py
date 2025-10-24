# generate_task2_smc.py
"""Task 2 — Sequential Monte Carlo (SMC) helpers.
"""
from __future__ import annotations
import os
import math
import torch
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# The FastRewardCalculator is assumed to be in a file named api.py
# as per the problem description.
from api import FastRewardCalculator

def load_counts_and_reward(counts_dir: str, epsilon: float = 1e-9) -> FastRewardCalculator:
    """Initialize trigram-based reward calculator for Sequential Importance Sampling.
    
    Args:
        counts_dir: Directory path containing ngrams data with trigram_probs.pkl cache
        epsilon: Smoothing parameter - minimum probability for unseen trigrams (prevents log(0))
        
    Returns:
        FastRewardCalculator: Configured calculator for computing R(x) rewards
    """
    cache_file = os.path.join(counts_dir, "trigram_probs.pkl")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Error: The trigram cache file was not found at {cache_file}."
            "Please ensure the cache file exists in the specified directory."
        )
    return FastRewardCalculator(cache_file, epsilon=epsilon)


def load_model(model_name: str, hf_token: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, int]:
    """Load and configure Hugging Face model components for Sequential Importance Sampling.
    
    Args:
        model_name: Hugging Face model repository ID (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
        hf_token: Authentication token for accessing gated models
        device: Target device for model placement ("cuda:0", "cpu", etc.)
        
    Returns:
        Tuple containing:
            - tokenizer: Configured AutoTokenizer with proper padding token
            - model: AutoModelForCausalLM in evaluation mode on target device
            - eos_id: End-of-sequence token ID for generation termination
    """
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    # LLaMA models often don't have a pad token, which is needed for batching.
    # Set it to the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    eos_id = tokenizer.eos_token_id

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="auto" if "cuda" in device else None
    )
    
    if "cuda" not in device:
        model.to(device)

    # 3. Set to evaluation mode
    model.eval()

    return tokenizer, model, eos_id

def cal_intermediate_target_dist(
    reward_calc: FastRewardCalculator, 
    tokenizer: AutoTokenizer, 
    beta: float, 
    full_ids: List[int]
) -> float:
    """
    Calculates the *incremental* reward r_t for the current step t.
    
    r_t = -log(p_tri(x_t | x_{t-2}, x_{t-1}))
    
    The caller is responsible for multiplying by beta and exponentiating
    to get the incremental weight.

    Args:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: for ids→tokens conversion.
        beta: (Not used in this function, applied by caller)
        full_ids: current full context ids (prompt + generated so far), i.e., x_{1:t}

    Returns:
        float r_t >= 0. The incremental reward for step t.
    """
    # The reward is based on trigrams.
    # If we don't have at least 3 tokens, the reward is 0.
    if len(full_ids) < 3:
        return 0.0

    # Get the last three token *strings*
    t1, t2, t3 = tokenizer.convert_ids_to_tokens(full_ids[-3:])
    
    # Get log P(t3 | t1, t2) from the reward calculator's LM
    log_p_tri = reward_calc.token_lm.logp(t1, t2, t3)
    
    # The reward r_t is the negative log-probability
    r_t = -log_p_tri
    
    return r_t

@torch.no_grad()
def smc_for_prompt(
    tokenizer: Any,
    model: Any,
    reward_calc: Any,
    *,
    prefix: str,
    N: int,
    max_new_tokens: int,
    eos_id: int,
    beta: float,
    k: int,
) -> Dict:
    """
    Performs Sequential Monte Carlo (Algorithm 2) for a single prompt.
    
    Inputs:
      tokenizer, model: HF components from load_model.
      reward_calc: FastRewardCalculator.
      prefix: full prompt string fed to the model (instruction + space + prefix).
      N: number of particles (B in the algorithm).
      max_new_tokens: continuation budget (T_max in the algorithm).
      eos_id: stopping id.
      beta: reward scale.
      k: top-k for proposal.

    Outputs:
      {
        "samples": [ {"text": str, "weight": float}, ... ],  (N samples)
        "normalized_weights": [float, ...] (N normalized linear weights)
      }
    """
    device = model.device
    
    # 1. Initialize Particles (Line 1, k=0)
    prompt_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    T_prompt = prompt_ids.shape[1]
    
    # Repeat the prompt N times to create N particles
    particles = prompt_ids.repeat(N, 1) # Shape: [N, T_prompt]
    
    # Initialize cumulative log-weights. W_0 = 1, so log(W_0) = 0.
    log_weights = torch.zeros(N, device=device)
    
    # For efficient generation
    past_key_values = None
    
    # Keep track of which particles have not hit EOS
    is_alive = torch.ones(N, dtype=torch.bool, device=device)

    # Loop for t = 1 ... T_max (Algorithm Line 1)
    for t in range(max_new_tokens):
        is_final_step = (t == max_new_tokens - 1)
        
        # --- 1. Propose (Sample) (Algorithm Line 3) ---
        
        if t == 0:
            input_ids = particles
        else:
            input_ids = particles[:, -1].unsqueeze(-1) # Shape: [N, 1]

        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs.logits[:, -1, :] # Get logits for next token [N, V]
        past_key_values = outputs.past_key_values
        
        # Top-K sampling
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1) # [N, 1]
        
        next_token_ids = torch.gather(top_k_indices, -1, sampled_indices) # [N, 1]
        
        new_particles = torch.cat([particles, next_token_ids], dim=-1)
        
        # --- 2. Weight (Algorithm Line 4) ---
        # As derived, log(w_t) = beta * r_t
        
        log_inc_weights = torch.zeros(N, device=device)
        
        for i in range(N):
            if is_alive[i]:
                full_ids_list = new_particles[i].tolist()
                
                # Calculate incremental reward r_t
                r_t = cal_intermediate_target_dist(
                    reward_calc, tokenizer, beta, full_ids_list
                )
                
                # Store incremental log-weight
                log_inc_weights[i] = beta * r_t
        
        # Update cumulative log-weights: log(W_t) = log(W_{t-1}) + log(w_t)
        log_weights += log_inc_weights
        
        # --- 3. Handle EOS ---
        just_finished = (next_token_ids.squeeze() == eos_id)
        # Stop accumulating weight for particles that just finished
        # We apply the mask *before* resampling
        is_alive = is_alive & ~just_finished
        
        # --- 4. Resample (Algorithm Lines 6-11) ---
        
        if not is_final_step:
            # Normalize weights to get sampling distribution
            norm_weights = torch.softmax(log_weights, dim=0)
            
            try:
                ancestor_indices = torch.multinomial(
                    norm_weights, num_samples=N, replacement=True
                )
            except RuntimeError:
                # Handle weight collapse (all log_weights = -inf)
                print("Warning: Weight collapse. Resampling uniformly.")
                ancestor_indices = torch.randint(0, N, (N,), device=device)
            
            # Create new particle set by selecting ancestors
            particles = new_particles[ancestor_indices]
            
            # Reorder the KV cache to match the resampled particles
            past_key_values = model._reorder_cache(past_key_values, ancestor_indices)
            
            # Update 'is_alive' status from ancestors
            is_alive = is_alive[ancestor_indices]
            
            # --- THIS IS THE CORRECTION ---
            # In a standard SIR particle filter, weights are reset to uniform
            # after resampling. log(W_t) = log(1) = 0.
            log_weights.fill_(0.0)
            
        else:
            # This is the last step. Do not resample.
            # The final particles and weights are set.
            particles = new_particles
        
        # If all particles have hit EOS, we can stop early
        if not is_alive.any():
            break
            
    # --- 5. Return (Algorithm Line 13) ---
    
    # Normalize the final cumulative weights
    final_normalized_weights = torch.softmax(log_weights, dim=0).tolist()
    
    samples_list = []
    for i in range(N):
        # Get the generated part (x_{1:T_max})
        generated_ids = particles[i][T_prompt:].tolist()
        
        # Clean up any EOS tokens
        try:
            eos_idx = generated_ids.index(eos_id)
            generated_ids = generated_ids[:eos_idx]
        except ValueError:
            pass # No EOS token found, use the full sequence
                
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        samples_list.append({
            "text": text,
            "weight": final_normalized_weights[i] # Return normalized linear weight
        })

    return {
        "samples": samples_list,
        "normalized_weights": final_normalized_weights
    }
