import argparse
import textworld
import numpy as np
import re
import requests
import json

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

# class WalkthroughDone(NameError):
#     pass

# class WalkthroughAgent(textworld.Agent):
#     """ Agent that follows a given list of commands"""

#     def __init__(self, commands=None):
#         self.commands = commands
    
#     def reset(self, env):
#         env.activate_state_tracking()
#         env.display_command_during_render = True
        
#         if self.commands is not None:
#             self._commands = iter(self.commands)
#             return  # Commands already specified.
        
#         if not hasattr(env, "game"):
#             raise NameError("WalkthroughAgent is only supported for generated games")

#         print(env.game.quests)
        # self._commands = iter(env.game.quests[0].commands)

    
    # def act(self, state, reward, done):
    #     try:
    #         action = next(self._commands)
    #     except StopIteration:
    #         raise WalkthroughDone()

    #     action = action.strip()
    #     return action
        


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


def openIE(sentence):
    url = "http://localhost:9000/"
    querystring = {
        "properties": "%7B%22annotators%22%3A%20%22openie%22%7D", #{"annotators": "openie"}
        "pipelineLanguage": "en"}
    response = requests.request("POST", url, data=sentence, params=querystring)
    response = json.JSONDecoder().decode(response.text)
    return response

def get_entity_relation(game_info):
    game = game_info[0]
    actions = set()
    output_file = open("./temp.txt",'w')

    # actions.update(run_agent(agent=WalkthroughAgent(),game=game,output_file=output_file,steps=1000,epochs=5))
    
    actions.update(run_agent(agent=RandomAgent(),game=game,output_file=output_file,steps=1000,epochs=5))

    output_file.close()


    output_file = open('./cleaned_temp.txt', 'w')
    with open('./temp.txt', 'r') as f:
        cur = []
        for line in f:
            if line != '---------' and "Actions:" not in str(line) and "Taken action:" not in str(line):
                cur.append(line)
            else:
                cur = [a.strip() for a in cur]
                cur = ' '.join(cur).strip().replace('\n', '').replace('---------', '')
                cur = re.sub("(?<=-\=).*?(?=\=-)", '', cur) #Look into easier way
                cur = cur.replace("-==-", '').strip()
                cur = '. '.join([a.strip() for a in cur.split('.')])
                output_file.write(cur + '\n')                
                cur = []

    output_file.close()

    input_file = open('./cleaned_temp.txt', 'r')

    entities = set()
    relations = set()

    sents = input_file.read()

    try: 
        list_sen = sents.split('\n')
        for sen in list_sen:
            for obv in openIE(sen)['sentences']:
                triple = obv['openie']
                for tr in triple:
                    h, r, t = tr['subject'], tr['relation'], tr['object']
                    entities.add(h)
                    entities.add(t)
                    relations.add(r)                
    except: 
        print("Error: OpenIE")

    act_out = open('./act2id.txt', 'w')
    act_out.write(str({k: i for i, k in enumerate(actions)}))
    act_out.close()

    ent_out = open('./entity2id.tsv', 'w')
    rel_out = open('./relation2id.tsv', 'w')

    for i, e in enumerate(entities):
        ent_out.write('_'.join(e.split()) + '\t' + str(i) + '\n')

    ent_out.close()
    for i, r in enumerate(relations):
        rel_out.write('_'.join(r.split()) + '\t' + str(i) + '\n')
    rel_out.close()

