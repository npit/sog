from dataclasses import dataclass

available_presets = {
    "honest2": [
        {
            "name": "honest1",
            "policy": "honest"
        },
        {
            "name": "honest2",
            "policy": "honest"
        }
    ],
    "random2": [
        {
            "name": "learner",
            "policy": "random"
        },
        {
            "name": "other_agent",
            "policy": "random"
        }

    ],
    "random3": [
        {
            "name": "learner",
            "policy": "random"
        },
        {
            "name": "agent_1",
            "policy": "random"
        },
        {
            "name": "agent_2",
            "policy": "random"
        }

    ],
    "ppo_honest": [
        {
            "name": "learner",
            "policy": "ppo"
        },
        {
            "name": "other_agent",
            "policy": "honest"
        }
    ],
    "ppo2": [
        {
            "name": "learner",
            "policy": "ppo"
        },
        {
            "name": "other_agent",
            "policy": "ppo"
        }
    ],

}