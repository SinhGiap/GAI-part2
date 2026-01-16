import random
from arena_env import ArenaEnv


def demo(steps=600):
    env = ArenaEnv(control_mode="rotation", max_steps=steps)
    obs, _ = env.reset()
    for step in range(steps):
        for event in __import__('pygame').event.get():
            if event.type == __import__('pygame').QUIT:
                env.close()
                return
        # random actions (including shoot)
        if env.control_mode == 'rotation':
            action = random.choice([0,1,2,3,4])
        else:
            action = random.choice([0,1,2,3,4,5])
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            print(f"Episode ended at step {step+1}, resetting")
            obs, _ = env.reset()
    env.close()

if __name__ == '__main__':
    demo(steps=600)
