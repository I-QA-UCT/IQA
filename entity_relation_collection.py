import argparse
import textworld
import numpy as np

class RandomAgent(textworld.Agent):
    """ A random agent that selects randomly chosen admissable action. """

    def __init__ (self,seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed) 

    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()

    def act(self, state, reward, done):
        return self.rng.choice(state.admissible_commands)

class WalkthroughDone(NameError):
    pass

class WalkthroughAgent(textworld.Agent):
    """ Agent that follows a given list of commands"""

    def __init__(self, commands=None):
        self.commands = commands
    
    def reset(self, env):
        env.activate_state_tracking()
        env.display_command_during_render = True
        
        if self.commands is not None:
            self._commands = iter(self.commands)
            return  # Commands already specified.
        
        if not hasattr(env, "game"):
            raise NameError("WalkthroughAgent is only supported for generated games")

        self._commands = iter(env.game.quests[0].commands)

    
    def act(self, state, reward, done):
        try:
            action = next(self._commands)
        except StopIteration:
            raise WalkthroughDone()

        action = action.strip()
        return action
        


def run_agent(agent, game, output_file, steps,epochs):
    env = textworld.start(game)
    env.enable_extra_info('description')

    actions = set()
    for epoch in range(epochs):

        agent.reset(env)
        state = env.reset()
        
        reward = 0
        done =False

        for step in range (steps):
            command = agent.act(state, reward, done)

            output_file.write(state.description)
            output_file.write("Actions: " + str(state.admissible_commands) + '\n')
            actions.update(state.admissible_commands)
            output_file.write("Taken action:" + str(command))
            output_file.write('\n' + "---------" + '\n')
            state, reward, done = env.step(command)

            if done: 
                break

    env.close()
    return actions

def get_entity_relation(game_info):
    game = game_info[0]
    actions = set()
    output_file = open("./temp.txt",'w')

    # actions.update(run_agent(agent=WalkthroughAgent(),game=game,output_file=output_file,steps=1000,epochs=5))
    actions.update(run_agent(agent=RandomAgent(),game=game,output_file=output_file,steps=1000,epochs=5))

    output_file.close()