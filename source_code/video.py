"""
For this part, the generation of videos for the LBF environment presented
significant technical difficulties due to compatibility issues between Gymnasium, lbforaging, and common video backends.

As a result, an alternative solution based on Matplotlib rendering was adopted.
A substantial portion of this script was implemented using ChatGPT, 
and then I adapted, tested, and debugged apart to ensure correct
functionality and integration with the trained agents.

The purpose of this file is exclusively to visualize trained agent behaviour.
"""

import gymnasium as gym
import numpy as np
import dill
import os
from PIL import Image
import matplotlib.pyplot as plt
import lbforaging
import io


def load_agent(model_path):
    """
    Loads a trained agent

    :param model_path: Path to the saved agent file
    :return: Loaded agent instance
    """
    with open(model_path, "rb") as f:
        agent = dill.load(f)
    return agent


def generate_gif_matplotlib(
    env,
    agent,
    num_episodes=5,
    max_steps=50,
    save_path="agent.gif",
    fps=5,
):
    """
    Generates a GIF by rendering the environment using Matplotlib.

    This approach avoids relying on Gym's built-in render modes, which
    proved unstable for the Level-Based Foraging environments.
    """
    frames = []

    #switch agent to evaluation mode
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for episode in range(num_episodes):
        obss, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            #create Matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 8))

            #extract the grid state from the environment
            grid = env.unwrapped.field.copy()

            #initialize RGB image
            img = np.ones((*grid.shape, 3))

            #mark food locations in green
            food_mask = grid > 0
            img[food_mask] = [0, 1, 0]

            #draw agents (red for agent 1, blue for agent 2) uisng cubes
            for i, player in enumerate(env.unwrapped.players):
                y, x = player.position
                if i == 0:
                    img[y, x] = [1, 0, 0]
                else:
                    img[y, x] = [0, 0, 1]

            ax.imshow(img, interpolation="nearest")
            ax.set_title(f"Episode {episode + 1}, Step {steps + 1}")

            #draw grid lines
            ax.grid(True, which="both", color="black", linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            #legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, fc="red", label="Agent 1"),
                plt.Rectangle((0, 0), 1, 1, fc="blue", label="Agent 2"),
                plt.Rectangle((0, 0), 1, 1, fc="green", label="Food"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                bbox_to_anchor=(1.15, 1),
            )

            plt.tight_layout()

            #convert Matplotlib figure to image frame
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            frame = Image.open(buf).copy()
            frames.append(frame)
            buf.close()
            plt.close(fig)

            #agent selects actions and environment steps forward
            actions = agent.act(obss)
            obss, rewards, done, truncated, _ = env.step(actions)

            steps += 1

            if done or truncated:
                break

    agent.epsilon = old_epsilon

    #save frames as GIF
    if frames:
        duration = int(1000 / fps)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f"GIF saved to: {save_path}")
        print(f"Total frames: {len(frames)}")
    else:
        print("No frames were captured.")


def generate_all_gifs(
    models_dir="models",
    gifs_dir="gifs",
    num_episodes=3,
    fps=5,
):
    """
    Generates GIF visualizations for all trained models and environments.
    """
    os.makedirs(gifs_dir, exist_ok=True)

    environments = [
        "Foraging-5x5-2p-1f-v3",
        "Foraging-5x5-2p-1f-coop-v3",
    ]

    agent_types = ["IQL", "CQL"]

    for env_name in environments:
        for agent_type in agent_types:
            model_filename = f"{agent_type}_{env_name.replace('-', '_')}.pkl"
            model_path = os.path.join(models_dir, model_filename)

            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue

            print(f"\n{'=' * 60}")
            print(f"Generating GIF for {agent_type} on {env_name}")
            print(f"{'=' * 60}")

            #load trained agent
            agent = load_agent(model_path)

            #create environment without render_mode
            env = gym.make(env_name)

            gif_filename = f"{agent_type}_{env_name.replace('-', '_')}.gif"
            gif_path = os.path.join(gifs_dir, gif_filename)

            generate_gif_matplotlib(
                env=env,
                agent=agent,
                num_episodes=num_episodes,
                max_steps=50,
                save_path=gif_path,
                fps=fps,
            )

            env.close()


def main():
    """
    Main for generating all GIF visualizations.
    """
    generate_all_gifs(
        models_dir="models",
        gifs_dir="gifs",
        num_episodes=3,
        fps=5,
    )


if __name__ == "__main__":
    main()
