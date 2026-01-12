import gymnasium as gym
import numpy as np
import random
import dill
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import lbforaging #library for the Foraging envs

from iql import IQL
from cql import CQL

# ---------------- CONFIGURATIONS ----------------
CONFIG_IQL = {
    "seed": 0,
    "gamma": 0.95,
    "total_eps": 30000,
    "ep_length": 50,
    "eval_freq": 2000,
    "lr": 0.2,
    "init_epsilon": 0.9,
}

CONFIG_CQL = {
    "seed": 0,
    "gamma": 0.95,
    "total_eps": 30000, #we tried 10000, 15000 but results were low
    "ep_length": 50,
    "eval_freq": 2000,
    "lr": 0.5,          #a lot of problems with learning rate, needed higher
    "init_epsilon": 0.9,
}

def evaluate_agent(env, agent, eval_episodes=50):
    """
    Evaluates a trained agent over a fixed number of episodes
    using a greedy policy (epsilon = 0).
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    returns = []
    for _ in range(eval_episodes):
        obss, _ = env.reset()
        done = False
        ep_return = 0
        steps = 0

        while not done and steps < eval_episodes:
            actions = agent.act(obss)
            obss, rewards, done, _, _ = env.step(actions)
            ep_return += sum(rewards)
            steps += 1

        returns.append(ep_return)

    agent.epsilon = old_epsilon
    return np.mean(returns)


def plot_results(episode_returns, eval_returns, eval_episodes, agent_type, env_name, save_dir="plots"):
    """
    Plots training and evaluation returns.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    #window for smoothing
    window = 100
    smoothed = np.convolve(
        episode_returns,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(smoothed, label="Train Return (100 ep avg)")
    plt.plot(eval_episodes, eval_returns, "o-", label="Eval Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{agent_type} on {env_name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plot_path = os.path.join(
        save_dir,
        f"{agent_type}_{env_name.replace('-', '_')}_training.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close()


# ---------------- TRAIN ----------------
def train_agent(env_name, agent_type, config, save_dir="models"):
    """
    Trains an IQL or CQL agent on the specified lbf.
    """
    print(f"Training {agent_type} on {env_name}")

    env = gym.make(env_name)
    num_agents = env.n_agents
    action_spaces = [env.action_space[i] for i in range(num_agents)]

    #create agent calling the config parameters
    if agent_type == "IQL":
        agent = IQL(
            num_agents=num_agents,
            action_spaces=action_spaces,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=config["init_epsilon"],
        )
    else:
        agent = CQL(
            num_agents=num_agents,
            action_spaces=action_spaces,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=config["init_epsilon"],
        )

    episode_returns = []
    eval_returns = []
    eval_episodes_list = []
    
    for ep in tqdm(range(config["total_eps"]), desc=f"{agent_type} Training"):
        #here we schedule epsilon and learning rate
        agent.schedule_hyperparameters(ep, config["total_eps"])

        #minimum epsilon to ensure some exploration
        agent.epsilon = max(agent.epsilon, 0.05)

        #learning rate decay, this was added because I have a lot of problems with lr
        #chatgpt implemented this idea.
        if ep > 0.7 * config["total_eps"]:
            agent.learning_rate = max(
                0.05,
                agent.learning_rate * 0.999
            )

        obss, _ = env.reset()
        done = False
        steps = 0
        ep_return = 0
        #while loop for each episode
        while not done and steps < config["ep_length"]:
            actions = agent.act(obss)
            n_obss, rewards, done, truncated, _ = env.step(actions)
            #agent learns from the transition
            agent.learn(
                obss=[obss[i] for i in range(num_agents)],
                actions=actions,
                rewards=[rewards[i] for i in range(num_agents)],
                n_obss=[n_obss[i] for i in range(num_agents)],
                done=done or truncated,
            )

            obss = n_obss
            ep_return += sum(rewards)
            steps += 1

        episode_returns.append(ep_return)

        # ---- EVALUATION ----
        if (ep + 1) % config["eval_freq"] == 0:
            mean_eval = evaluate_agent(env, agent)
            eval_returns.append(mean_eval)
            eval_episodes_list.append(ep + 1)

            print(
                f"Episode {ep+1} | "
                f"Eval Return: {mean_eval:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"LR: {agent.learning_rate:.3f}"
            )

    # ---- SAVE MODEL ----
    #verify save directory exists
    os.makedirs(save_dir, exist_ok=True)
    #use of dill because with pickle library there were problems
    model_path = os.path.join(
        save_dir,
        f"{agent_type}_{env_name.replace('-', '_')}.pkl"
    )
    with open(model_path, "wb") as f:
        dill.dump(agent, f)

    print(f"Model saved to {model_path}")

    # ---- PLOT RESULTS ----
    plot_results(
        episode_returns,
        eval_returns,
        eval_episodes_list,
        agent_type,
        env_name,
    )

    env.close()
    return agent


# ---------------- MAIN ----------------
def main():
    #set seeds for reproducibility
    random.seed(0)
    np.random.seed(0)

    environments = [
        "Foraging-5x5-2p-1f-v3",
        "Foraging-5x5-2p-1f-coop-v3",
    ]

    agent_configs = {
        "IQL": CONFIG_IQL,
        "CQL": CONFIG_CQL,
    }
    #train agents for each environment and agent type
    for env_name in environments:
        for agent_type, config in agent_configs.items():
            train_agent(env_name, agent_type, config)


if __name__ == "__main__":
    main()
