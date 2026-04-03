# 🐍 Snake Game using Reinforcement Learning (Q-Learning)

## 📌 Project Overview
This project implements a Snake Game AI agent using Reinforcement Learning (Q-Learning).  
The agent learns to play the game by interacting with the environment and maximizing cumulative rewards.

The project also includes several improvements over basic implementations, such as reward shaping, loop prevention, and environment enhancements.

---

## 🎯 Objectives
- Implement a Snake game using Reinforcement Learning  
- Train an agent to learn optimal movement strategies  
- Improve learning efficiency and stability  
- Demonstrate practical application of Q-learning  

---

## 🧠 Reinforcement Learning Concepts

- **Agent**: Snake  
- **Environment**: Grid world with food and obstacles  
- **State**: Danger detection + food direction  
- **Actions**: Up, Down, Left, Right  

### 🎁 Reward Function
- +30 → Eat food  
- -20 → Collision  
- +5 → Move closer to food  
- -5 → Move away from food  

- **Algorithm**: Q-Learning  
- **Exploration Strategy**: Epsilon-Greedy  

---

## ⚙️ Features

### ✅ Core Features
- Q-learning based Snake agent  
- Real-time game visualization using Pygame  
- Score tracking  

### 🚀 Improvements Added
- Reward shaping for faster learning  
- Loop prevention mechanism  
- Safe food spawning  
- Obstacles in environment  
- Multiple difficulty levels  
- Training and gameplay separation  
- Gym-style environment structure  

### 💾 Additional Features
- Save and load trained model  
- Restart game functionality  
- Adjustable speed and difficulty  

---

## 🏗️ Project Structure
Snake-RL/
│── main.py
│── snake_q_model.pkl
│── README.md


---

## ▶️ How to Run

### 1. Install dependencies
pip install numpy pygame matplotlib


### 2. Run the project

python main.py


### 3. Choose option
- Train model  
- Load existing model  

---

## 🎮 Controls
- **R** → Restart game  
- Close window → Exit  

---

## 📊 Results
The agent learns to:
- Avoid collisions  
- Move toward food  
- Improve score over time  

Performance is evaluated using:
- Reward progression  
- Gameplay behavior  

---

## 📄 Base Reference

This project is based on Reinforcement Learning applied to Snake games.

### Base Research Idea
- Reinforcement Learning for Snake game agents  
- Q-learning based decision making  

---

## 🔄 Improvements Over Base Work

| Feature | Base Implementation | This Project |
|--------|-------------------|--------------|
| Reward | Simple | Reward shaping |
| Loop Handling | ❌ No | ✅ Yes |
| Food Spawn | Random | Safe spawning |
| Obstacles | ❌ No | ✅ Yes |
| Difficulty | ❌ No | ✅ Yes |
| Training | Combined | Separated |
| Structure | Basic | Modular (Gym-style) |

---

## 🧠 Key Learning Outcomes
- Understanding of Reinforcement Learning concepts  
- Implementation of Q-learning  
- Designing reward functions  
- Handling RL challenges (loops, exploration)  
- Building custom RL environments  

---

## 🎤 Viva Summary
> This project uses Q-learning to train a Snake agent using rewards and penalties. The agent learns optimal actions through interaction with the environment.

---

## 🚀 Future Work
- Implement Deep Q-Network (DQN)  
- Improve state representation  
- Add dynamic obstacles  
- Integrate with Gymnasium  

---

## 👨‍💻 Author
**Gokul Satheesh**
