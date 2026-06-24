import numpy as np
import neurogym as ngym
import inspect

# create env exactly as your training script does
env = ngym.make("ContextDecisionMaking-v0")

print(inspect.getsource(env.unwrapped.__class__._new_trial))
print(type(env.unwrapped))
print(env.unwrapped.__class__.__module__)
print(env.unwrapped.__class__.__name__)
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
print("Name mapping:", getattr(env.observation_space, "name", None))

obs, info = env.reset()

print("\nInitial obs:", obs)
print("Info:", info)

# run one episode and log structure
obs_list = []

for t in range(200):
    action = 0  # always fixate (important for clean probing)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    obs_list.append(obs.copy())

    if done:
        break

obs_arr = np.array(obs_list)

print("\nTrajectory shape:", obs_arr.shape)
print("Mean per channel:", obs_arr.mean(axis=0))
print("Std per channel:", obs_arr.std(axis=0))

# print sample timesteps
print("\nFirst 5 timesteps:")
for i in range(5):
    print(i, obs_arr[i])