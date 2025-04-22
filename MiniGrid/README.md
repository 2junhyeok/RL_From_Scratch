## SARSA, Q-Learing 학습에서 사용하는 탐색 정책의 차이 비교 분석

This project compares **SARSA** and **Q-Learning** algorithms in a minimal grid world using **MiniGrid-Empty-6x6-v0** environment. It includes Q-table visualization, training statistics, and reward analysis for better insight into learning dynamics.

<br>

## 📦 Features

- 🔁 Episodic training with SARSA or Q-Learning
- 📊 Smoothed reward and success rate visualization
- 🧠 Q-table extraction and CSV export
- 🗺️ Heatmap of Q-values for all state-action pairs

<br>

## 🧱 Environment

- **Python 3.8+**
- **MiniGrid (gym-minigrid)**
- **PyTorch (for optional extensions)**
- **Seaborn / Matplotlib / Pandas**

```bash
pip install gym gym-minigrid matplotlib seaborn pandas
```
<br>

## 🏁 How to Run

python experiment.py --model sarsa --episodes 1000 --env MiniGrid-Empty-6x6-v0
Arguments:

Argument	Description	Default
--model	Algorithm (sarsa, q_learning)	q_learning
--alpha	Learning rate	0.01
--gamma	Discount factor	0.9
--epsilon	Epsilon-greedy value	0.2
--episodes	Number of episodes	1000
--log-interval	Logging frequency	100
--save-path	Directory to save results	results
<br>

## 📂 Output

results/

training_results.png — training curves (Q-value mean, reward, success rate)

q_heatmap_all_states_actions.png — full Q-table heatmap

q_values.csv — raw Q-table in CSV format


<br>

## 📈 Example: Training Curve and Heatmap
<img src="MiniGrid/SarsaAndQLearning/experiments/results_09_S/q_heatmap_all_states_actions.png" width="450"/> <img src="MiniGrid/SarsaAndQLearning/experiments/results_09_S/training_results.png" width="450"/>

<br>

## 📁 Project Structure
.
├── experimentss/\
│   ├── experiment.py  \            
│   ├── results/\
│   └── ...       
├── models/
│   ├── SARSA.py               # SARSA algorithm\
│   └── Q_Learning.py          \
├── logs/                      # Output folder\
├── SarsaAndQLearning.ipynb\
└── README.md