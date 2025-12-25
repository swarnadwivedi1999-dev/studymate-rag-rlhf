# rlhf_loop.py
def optimize_prompt(base_prompt, reward):
    if reward < 0:
        return base_prompt + "\nPlease provide a clearer and more concise answer."
    elif reward >= 1.0:
        # Positive feedback - keep the prompt as is or add encouragement
        return base_prompt
    else:
        # Neutral feedback (0.5)
        return base_prompt