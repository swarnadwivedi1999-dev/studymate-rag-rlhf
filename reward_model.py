# reward_model.py
def compute_reward(rating):
    if rating >= 4:
        return 1.0
    elif rating == 3:
        return 0.5
    else:
        return -1.0