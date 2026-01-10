import gymnasium as gym
import imageio
import numpy as np
import lbforaging

from cql import CQL

ENV_NAME = "Foraging-5x5-2p-1f-coop-v3"


def record_video(agent, filename="lbf_cql.mp4"):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    frames = []

    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.act(obs)
        obs, _, done, _, _ = env.step(action)
        frames.append(env.render())

    imageio.mimsave(filename, frames, fps=5)
    env.close()


if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    agent = CQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=0.99,
        learning_rate=0.1,
        epsilon=0.05,
    )


    record_video(agent)
