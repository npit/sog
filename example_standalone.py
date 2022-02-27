"""
Example script of standalone environment utilization.
"""

from sog.sog import SOG
import logging

logging.basicConfig(level=logging.INFO)

preset = [
    {
        "name": "learner",
        "policy": "ppo"
    },
    {
        "name": "other_agent",
        "policy": "ppo"
    }
]
policy_paths = {
    "learner": "path/to/saved/policy/ppo.zip",
    "other_agent": "path/to/another/saved/policy/ppo.zip"
}
fn = SOG.get_agent_init_func(preset, policy_load_config=policy_paths)
env = SOG(agent_presets=preset, agent_initializer_funcs=fn)
state = env.reset()

done = False

while done != True:
    action = env.learning_agent.apply_policy(env)
    obs, rew, done, info = env.step(action) #take step using selected action
    env.render()
