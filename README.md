# Multi-Agent Reinforcement Learning Project

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
│   ├── iql.py                 # Independent Q-Learning implementation
│   ├── cql.py                 # Cooperative Q-Learning implementation
│   ├── train.py               # Main training script for LBF environments
│   ├── train_iql.py           # Training script for Prisoner's Dilemma
│   ├── video.py               # GIF generation for trained agents
│   └── pd_game.py             # Prisoner's Dilemma environment
├── models/                    # Saved trained models (.pkl files)
├── gifs/                      # Generated visualizations
├── plots/                     # Training curves and results
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

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

#### Level-Based Foraging
Train IQL and CQL agents on both cooperative and standard foraging environments:

```bash
cd source_code
python train.py
```

This will:
- Train agents for 30,000 episodes
- Evaluate every 2,000 episodes
- Save models to `models/`
- Generate training plots in `plots/`

#### Prisoner's Dilemma
Train IQL agents on the matrix game:

```bash
python train_iql.py
```

### Generating Visualizations

Create GIFs of trained agent behaviors:

```bash
python video.py
```

Generated GIFs will be saved in `gifs/` directory, including:
- Individual agent performances
- Side-by-side comparisons of IQL vs CQL

## 📊 Environments

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

### Prisoner's Dilemma

Classic 2-player matrix game with payoff structure:

|               | Cooperate | Defect |
|---------------|-----------|--------|
| **Cooperate** | (-1, -1)  | (-5, 0)|
| **Defect**    | (0, -5)   | (-3, -3)|

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

### Hyperparameters

**IQL Configuration:**
- Learning rate: 0.2
- Discount factor (γ): 0.95
- Initial ε: 0.9
- Evaluation ε: 0.05
- Episodes: 30,000

**CQL Configuration:**
- Learning rate: 0.5
- Discount factor (γ): 0.95
- Initial ε: 0.9
- Evaluation ε: 0.05
- Episodes: 30,000

## 📈 Results

### Key Findings

1. **Cooperative Environment**:
   - Both IQL and CQL achieve ~85-95% success rate
   - Agents learn effective coordination
   - Stable convergence around 15,000 episodes

2. **Standard Environment**:
   - High variance and instability
   - CQL: 15-55% success (with catastrophic forgetting)
   - IQL: 25-32% success (more stable but lower performance)
   - Neither converges to stable policies

3. **Prisoner's Dilemma**:
   - IQL converges to Nash Equilibrium (Defect, Defect)
   - Returns stabilize at -3.0
   - Demonstrates classic dilemma outcome

### Interpretation

The results demonstrate that independent learners excel in cooperative settings with aligned incentives but struggle significantly in mixed-motive or competitive scenarios. The non-stationary nature of multi-agent learning prevents stable convergence when agents have conflicting objectives.

## 📁 Saved Models

Trained models are saved as `.pkl` files in the `models/` directory:
- `IQL_Foraging_5x5_2p_1f_v3.pkl`
- `CQL_Foraging_5x5_2p_1f_v3.pkl`
- `IQL_Foraging_5x5_2p_1f_coop_v3.pkl`
- `CQL_Foraging_5x5_2p_1f_coop_v3.pkl`

Models can be loaded using:
```python
import dill
with open('models/IQL_Foraging_5x5_2p_1f_coop_v3.pkl', 'rb') as f:
    agent = dill.load(f)
```

## 🎥 Visualizations

The `video.py` script generates:
- Individual agent GIFs showing behavior in each environment
- Comparison GIFs showing IQL vs CQL side-by-side
- Original environment sprites and graphics

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

This is an academic project for UAB - Paradigms of Machine Learning course. For questions or issues, please open an issue on GitHub.

## 📄 License

This project is part of an academic assignment at Universitat Autònoma de Barcelona (UAB).

## 👥 Authors

- Miguel231 - [GitHub Profile](https://github.com/Miguel231)

## 🙏 Acknowledgments

- Level-Based Foraging environment: [lb-foraging](https://github.com/semitable/lb-foraging)
- Course: Paradigms of Machine Learning, UAB
- Professor: [Course Instructor Name]

## 📧 Contact

For questions or collaboration, please reach out through GitHub issues or pull requests.

---

**Note**: This project demonstrates fundamental concepts in multi-agent reinforcement learning and is intended for educational purposes.