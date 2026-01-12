# PRACTICAL ACTIVITY 2

This repository contains implementations of **Independent Q-Learning (IQL)** and **Cooperative Q-Learning (CQL)** algorithms applied to multi-agent environments, including the Prisoner's Dilemma matrix game and Level-Based Foraging (LBF) tasks.

## 📋 Project Overview

This project explores multi-agent reinforcement learning in both competitive and cooperative settings:

1. **Prisoner's Dilemma**: A classic game theory problem testing cooperation vs. defection strategies
2. **Level-Based Foraging (LBF)**: Grid-world environments requiring coordination between agents to collect food

The goal is to analyze how independent learners (IQL and CQL) perform in different reward structures and coordination requirements.

## 🏗️ Project Structure

```
RL-PRA2/
├── source_code/
│   ├── gifs/                  # Generated visualizations
│   ├── plots/                 # Training curves and results
│   ├── models/                # Saved trained models (empty for memory issues in github)
│   ├── iql.py                 # Independent Q-Learning implementation
│   ├── cql.py                 # Cooperative Q-Learning implementation
│   ├── train_lbf.py           # Main training script for LBF environments
│   ├── train_main.py          # Training script for Prisoner's Dilemma
│   ├── video.py               # GIF generation for trained agents in LBF environments
│   ├── requirements.txt       # Python dependencies
│   └── matrix_game.py         # Prisoner's Dilemma environment
└── README.md                  # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.10.15 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Miguel231/RL-PRA2.git
cd RL-PRA2
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training Agents

#### Prisoner's Dilemma
Train IQL agents on the matrix game:

```bash
python train_main.py
```

#### Level-Based Foraging
Train IQL and CQL agents on both cooperative and standard foraging environments:

```bash
cd source_code
python train_lbf.py
```

This will:
- Train agents for 30,000 episodes
- Evaluate every 2,000 episodes
- Save models to `models/`
- Generate training plots in `plots/`

### Generating Visualizations
Create GIFs of trained agent behaviors:

```bash
python video.py
```

Generated GIFs will be saved in `gifs/` directory

## 📊 Environments

### Prisoner's Dilemma

Classic 2-player matrix game with payoff structure:

|               | Cooperate | Defect |
|---------------|-----------|--------|
| **Cooperate** | (-1, -1)  | (-5, 0)|
| **Defect**    | (0, -5)   | (-3, -3)|

### Level-Based Foraging

Two variants are included:

1. **Foraging-5x5-2p-1f-v3** (Standard)
   - 5x5 grid
   - 2 players
   - 1 food item
   - Agents can collect food individually if their level is sufficient

2. **Foraging-5x5-2p-1f-coop-v3** (Cooperative)
   - Same setup as standard
   - **Requires both agents to coordinate** to collect any food
   - Tests pure cooperation

## 🧠 Algorithms

### Independent Q-Learning (IQL)
- Each agent learns independently
- Uses ε-greedy exploration
- Standard Q-learning updates
- No explicit coordination mechanism

### Cooperative Q-Learning (CQL)
- Similar to IQL but designed for cooperative tasks
- Conservative Q-value updates
- Encourages coordination through reward structure
- Independent learning with cooperation incentives

## 📁 Saved Models

Trained models are saved as `.pkl` files in the `models/` directory:
- `IQL_Foraging_5x5_2p_1f_v3.pkl`
- `CQL_Foraging_5x5_2p_1f_v3.pkl`
- `IQL_Foraging_5x5_2p_1f_coop_v3.pkl`
- `CQL_Foraging_5x5_2p_1f_coop_v3.pkl`

## 🎥 Visualizations

The `video.py` script generates:
- Individual agent GIFs showing behavior in each environment

## 🔬 Experimental Setup

- **Training episodes**: 30,000 per agent
- **Episode length**: 50 steps maximum
- **Evaluation frequency**: Every 2,000 episodes
- **Evaluation episodes**: 50 episodes per evaluation
- **Random seed**: 0 (for reproducibility)

## 📚 Dependencies

Main libraries:
- `gymnasium`: RL environment interface
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `lbforaging`: Level-based foraging environments
- `pygame`: Rendering and visualization
- `Pillow (PIL)`: Image processing for GIFs
- `dill`: Model serialization
- `tqdm`: Progress bars

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

This is an academic project for UAB - Paradigms of Machine Learning course.

## 📄 License

This project is part of an academic assignment at Universitat Autònoma de Barcelona (UAB).

## 👥 Authors

- Miguel231 - [GitHub Profile](https://github.com/Miguel231)

## 🙏 Acknowledgments

- Level-Based Foraging environment: [lb-foraging](https://github.com/semitable/lb-foraging)
- Course: Paradigms of Machine Learning, UAB

**Note**: This project demonstrates fundamental concepts in multi-agent reinforcement learning and is intended for educational purposes.