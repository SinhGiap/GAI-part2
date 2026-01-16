"""
Training script for the rotation control mode using PPO.
Trains for 500k timesteps and saves the model to models/rotation/.
"""

import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from arena_env import ArenaEnv


def main():
    # create environment
    env = ArenaEnv(control_mode="rotation", max_steps=2000)
    
    # create model save directory
    model_dir = "models/rotation"
    os.makedirs(model_dir, exist_ok=True)
    
    # create logs directory
    log_dir = "logs/rotation"
    os.makedirs(log_dir, exist_ok=True)
    
    # create evaluation environment
    eval_env = ArenaEnv(control_mode="rotation", max_steps=2000)
    
    # evaluation callback to save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # checkpoint callback to save periodic checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix="rotation_checkpoint"
    )
    
    # create PPO model with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # MUCH higher exploration (was 0.03)
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu"
    )
    
    print("Starting PPO training for rotation control...")
    print(f"Total timesteps: 600,000")
    print(f"Model will be saved to: {model_dir}")
    
    # train the model
    model.learn(
        total_timesteps=600000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # save final model
    final_model_path = os.path.join(model_dir, "rotation_final")
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
