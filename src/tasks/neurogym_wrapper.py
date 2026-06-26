import neurogym as ngym
import torch
import numpy as np

try:
    from .mante_config import CONFIG, UNIFORM_COHS, MANTE_TEST_COHS
except ImportError:  # pragma: no cover - fallback for direct script execution
    from mante_config import CONFIG, UNIFORM_COHS, MANTE_TEST_COHS
# Safe observation maps (as you already did)
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


# ─────────────────────────────────────────────────────────────
# CONFIG-DRIVEN DATASET CREATION
# ─────────────────────────────────────────────────────────────

def create_dataset_generator(config):

    if isinstance(config, str):
        config = {"task": config}

    task_name = config.get("task_name", "ContextDecisionMaking-v0")
    batch_size = config.get("batch_size", 64)
    dt = config.get("dt", 20)
    seq_len = config.get("seq_len", 150)

    sigma = config.get("sigma", 1.0)
    timing = config.get("timing", None)
    use_expl_context = config.get("use_expl_context", True)

    #  FIX: correct coherence handling
    cohs = config.get("coh_levels", None)

    env_kwargs = {
        "dt": dt,
        "sigma": sigma,
    }

    if timing is not None:
        env_kwargs["timing"] = timing

    env = ngym.make(
        task_name,
        **env_kwargs,
        use_expl_context=use_expl_context
    )

    env.reset(seed=42)

    # ⚠️ SAFE ONLY IF TASK SUPPORTS IT
    if cohs is not None:
        try:
            env.unwrapped.cohs = np.array(cohs)
            print("Using coherence levels:", env.unwrapped.cohs[:5], "...")
        except AttributeError:
            print("Warning: task has no 'cohs' attribute")

    dataset = ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)

    dataset.config = config
    return dataset


# ─────────────────────────────────────────────────────────────
# SINGLE TRIAL GENERATOR (ALSO CONFIG-DRIVEN)
# ─────────────────────────────────────────────────────────────

def generate_single_trial(config):
    task_name = config.get("task_name", "PerceptualDecisionMaking-v0")
    dt = config.get("dt", 20)
    sigma = config.get("sigma", 1.0)
    timing = config.get("timing", None)
    use_expl_context = config.get("use_expl_context", True)

    env_kwargs = {
        "dt": dt,
        "sigma": sigma,
    }

    if timing is not None:
        env_kwargs["timing"] = timing

    env = ngym.make(
        task_name,
        **env_kwargs,
        use_expl_context=use_expl_context
    )

    env.reset()

    ob_dict = OB_DICTS[task_name]

    observations = []
    actions = []

    while True:
        action = env.action_space.sample()

        step_returns = env.step(action)

        if len(step_returns) == 5:
            obs, reward, terminated, truncated, info = step_returns
        else:
            obs, reward, done, info = step_returns

        observations.append(obs)
        actions.append(info['gt'])

        if info.get('new_trial', False):
            break

    return np.array(observations), np.array(actions), ob_dict