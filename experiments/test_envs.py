import jaxrl_m.envs
import gym
# Path: experiments/test_envs.py

def test_env(env_name):
    env = gym.make(env_name)
    env.reset()
    print("\n\nENVIRONMENT DETAILS")
    print("Env name:", env_name)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Modes:", env.get_num_modes())
    print("Is multimodal:", env.is_multimodal)
    

    #Testing modes randomization
    print("\n\nTesting modes randomization")
    for _ in range(10):
        env.reset()
        env.reset_mode()
        print("Mode:", env.mode)
        print("Target:", env.target)
            
    
    if env.is_multimodal:
        print("\n\nTesting multimodal env")
        for i in range(env.get_num_modes()):
            env.set_mode(i)
            print("Target:", env.target)
    
    print("\n\nTesting step")
    env.reset()
    for _ in range(100):
        env.render(mode="rgb_array")
        env.step(env.action_space.sample())
        
    env.close()
    print("Test passed!\n\n")

if __name__ == "__main__":
    envs = [
       "maze2d-twogoals-mode0-v0",
       "maze2d-twogoals-mode1-v0",
       "maze2d-twogoals-multimodal-v0",
    ]
    for env in envs:
        test_env(env)