import time
import random
from typing import Any
import jax
import jax.numpy as jnp
import gymnax
import optax
from flax import nnx

from brll_core.algorithms.common.utils import parse_config
from brll_core.algorithms.common.logger import Logger

class PolicyNetwork(nnx.Module): 
    """Maps state -> action logits."""
    def __init__(self, din: int, dhidden: int, dout: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dhidden, rngs=rngs)
        self.linear2 = nnx.Linear(dhidden, dhidden, rngs=rngs)
        self.linear3 = nnx.Linear(dhidden, dout, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        return x

def compute_returns(rewards, gamma):
    def step(G, r):
        G = r + gamma * G
        return G, G
    _, returns = jax.lax.scan(step, 0.0, rewards, reverse=True)
    return returns

if __name__ == "__main__":
    config = parse_config()
    logger = Logger(config)

    # Extract hyperparameters from config layout
    env_name = config["env"]["make"]["id"].split("/")[-1]  # Handles "brll/CartPole-v1" -> "CartPole-v1"
    num_episodes = config["training"]["num_episodes"]
    gamma = config["training"]["gamma"]
    hidden_dim = config["network"]["actor"]["layers"][0]["params"]["out_features"] # or similar path mapping
    lr = config["training"]["optimiser"]["Adam"]["lr"]

    # Seed initialization
    seed = config["seed"]
    key = jax.random.PRNGKey(seed)
    
    # Initialization Gymnax Environment with Params
    env, env_params = gymnax.make(env_name)
    state_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).n
    max_episode_steps = env.num_steps

    # Initialize Model
    key, model_key = jax.random.split(key)
    policy = PolicyNetwork(din=state_dim, dout=action_dim, dhidden=hidden_dim, rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(policy, optax.adam(lr), wrt=nnx.Param)

    # Functional Episode Step definition
    def play_episode_step(carry, _):
        current_obs, current_state, current_key = carry
        
        # Policy Forward Pass
        logits = policy(current_obs)
        
        # Action Sampling
        current_key, subkey = jax.random.split(current_key)
        action = jax.random.categorical(subkey, logits)
        
        # Environment Step
        current_key, step_key = jax.random.split(current_key)
        next_obs, next_state, reward, done, _ = env.step(
            step_key, current_state, action, env_params
        )
        
        return (next_obs, next_state, current_key), (current_obs, action, reward, done)

    # Compile the episode rollout graph natively
    @jax.jit
    def run_full_episode(current_key):
        current_key, reset_key = jax.random.split(current_key)
        init_obs, init_state = env.reset(reset_key, env_params)
        
        # Use jax.lax.scan to execute the loop fully on the accelerator hardware
        _, (obs_history, actions_history, rewards_history, dones_history) = jax.lax.scan(
            play_episode_step,
            (init_obs, init_state, current_key),
            None,
            length=max_episode_steps
        )
        return obs_history, actions_history, rewards_history, dones_history

    # Optimization Step Compilation Function
    @jax.jit
    def train_step(policy, optimizer, states, actions, returns):
        def loss_fn(policy_model):
            logits = policy_model(states)
            log_probs_all = jax.nn.log_softmax(logits)
            log_probs = jnp.take_along_axis(log_probs_all, actions[:, None], axis=1).squeeze(1)
            return -(log_probs * returns).sum()

        loss, grads = nnx.value_and_grad(loss_fn)(policy)
        optimizer.update(policy, grads)
        return loss

    # Main Execution Iteration Engine
    for episode in range(num_episodes):
        key, episode_key = jax.random.split(key)
        
        # Rollout episode completely on GPU
        states, actions, rewards, dones = run_full_episode(episode_key)
        
        # Mask out rewards earned post-termination flag from fixed-length scan
        valid_mask = jnp.concatenate([jnp.array([True]), ~jnp.cumsum(dones)[:-1]])
        masked_rewards = rewards * valid_mask
        
        episode_return = jnp.sum(masked_rewards).item()
        returns = compute_returns(masked_rewards, gamma)

        # Optimization
        loss = train_step(policy, optimizer, states, actions, returns)

        if episode % config["training"]["eval_freq"] == 0:
            logger.log_metric("train/return_mean", episode_return, episode)
            logger.log_metric("objectives/policy_loss", loss.item(), episode)
            print(f"Episode {episode} | Return: {episode_return:.2f} | Loss: {loss:.3f}")

    logger.close()
