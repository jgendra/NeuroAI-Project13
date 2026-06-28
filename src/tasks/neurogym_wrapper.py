import neurogym as ngym
import torch
import numpy as np

# Hardcoded observation dictionaries for safety across Gymnasium versions
OB_DICTS = {
    'PerceptualDecisionMaking-v0': {
        'fixation': 0,
        'stimulus': [1, 2]
    },
    'ContextDecisionMaking-v0': {
        'fixation': 0,
        'stimulus1': [1, 2],
        'stimulus2': [3, 4],
        'context': [5, 6]
    }
}

def create_dataset_generator(task_name='PerceptualDecisionMaking-v0', batch_size=64, dt=20, seq_len=150):
    """
    Creates a NeuroGym environment ONCE and returns a continuous data generator.
    This prevents massive CPU overhead from recreating the environment every batch.
    """
    env_kwargs = {'dt': dt}
    
    # Note: In neurogym v1.0.8, ContextDecisionMaking-v0 naturally outputs 
    # the 7 required channels. We do not pass `use_expl_context=True` 
    # because it throws a TypeError in this specific version.
    
    env = ngym.make(task_name, **env_kwargs)
    dataset = ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)
    
    return dataset

def generate_single_trial(task_name, dt=20):
    """Helper function to generate a single clean trial for visualization."""
    env_kwargs = {'dt': dt}
    
    env = ngym.make(task_name, **env_kwargs)
    env.reset()
    
    # Use our safe, hardcoded dictionary instead of env.ob_dict
    ob_dict = OB_DICTS[task_name]
    
    observations = []
    actions = []
    
    # Run continuously until NeuroGym signals the end of the biological trial
    while True:
        action = env.action_space.sample() 
        
        # Handle Gymnasium vs Gym API update gracefully
        step_returns = env.step(action)
        
        if len(step_returns) == 5:
            obs, reward, terminated, truncated, info = step_returns
        else:
            obs, reward, done, info = step_returns
            
        observations.append(obs)
        actions.append(info['gt']) 
        
        # Break when NeuroGym signals a new trial starting
        if info.get('new_trial', False):
            break
        
    return np.array(observations), np.array(actions), ob_dict
