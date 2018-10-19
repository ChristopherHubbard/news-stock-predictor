import json

class ConfigManager():

    # Initialize the Config using the env setting
    def __init__(self, env):

        # Set the settings
        with open('config.json', 'r') as f:
            envs = json.load(f)

        # Set the config based on the env
        self.config = envs[env]