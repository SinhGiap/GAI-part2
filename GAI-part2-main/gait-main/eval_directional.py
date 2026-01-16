"""
Evaluation script for the directional control mode.
Loads the best trained model and runs evaluation episodes with rendering.
"""

import os
import numpy as np
import pygame
from stable_baselines3 import PPO
from arena_env import ArenaEnv


def main():
    # load the best model
    model_path = "models/directional/best_model"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        print("Please train the model first using train_directional.py")
        return
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device="cpu")
    
    # create environment with rendering
    env = ArenaEnv(control_mode="directional", max_steps=2000)
    
    # run evaluation episodes
    num_episodes = 5
    episode_rewards = []
    episode_lengths = []
    episode_phases = []
    
    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("Close the window to end evaluation early.\n")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_phase = 1
        
        running = True
        while not done and running:
            # handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
            # predict action using the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # take step in environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            max_phase = max(max_phase, env.phase)
            
            # render the environment
            env.render()
        
        if not running:
            print("\nEvaluation interrupted by user.")
            break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_phases.append(max_phase)
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}, Max Phase = {max_phase}")
    
    env.close()
    
    # print statistics
    if episode_rewards:
        print("\n" + "=" * 60)
        print("EVALUATION STATISTICS")
        print("=" * 60)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Average max phase: {np.mean(episode_phases):.1f} ± {np.std(episode_phases):.1f}")
        print(f"Best episode reward: {np.max(episode_rewards):.2f}")
        print(f"Worst episode reward: {np.min(episode_rewards):.2f}")
        print("=" * 60)


if __name__ == "__main__":
    main()

