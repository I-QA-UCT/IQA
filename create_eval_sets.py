import os
import shutil
import argparse

import gym
import json

import game_generator
from agent import Agent
from textworld.gym import register_game, make_batch2
import textworld

import numpy as np

request_infos = textworld.EnvInfos(description=True,
                                   inventory=True,
                                   verbs=True,
                                   location_names=True,
                                   location_nouns=True,
                                   location_adjs=True,
                                   object_names=True,
                                   object_nouns=True,
                                   object_adjs=True,
                                   facts=True,
                                   last_action=True,
                                   game=True,
                                   admissible_commands=True,
                                   extras=["object_locations", "object_attributes", "uuid"])


def create_evaluation_sets(agent, variant):

    question_types = ["location", "attribute", "existence"]
    random_map_types = [False, True]
    game_no = variant["games"]

    data = {}
    counter = 0

    for question_type in question_types:
        
        map_data = {}

        for random_map in random_map_types:

            map_type = "random_map" if random_map else "fixed_map"
            target_dest = "eval_set/" + question_type + "/" + map_type 
            
            games = game_generator.game_generator(
                path=".",
                random_map=random_map, 
                question_type=question_type,
                train_data_size=game_no, # number of games
            
            )

            games.sort()
            all_env_ids = [register_game(gamefile, request_infos=request_infos) for gamefile in games]
                        
            game_dict = {} 
            for episode, (game, env_id) in enumerate(zip(games, all_env_ids)):
                
                env_ids = make_batch2([env_id], parallel=True)

                env = gym.make(env_ids)
                env.seed(episode)

                obs, infos = env.reset()


                questions, answers, reward_helper_info = game_generator.generate_qa_pairs(infos, question_type=question_type, seed=episode)
                
                games_name = game.split(".")[1]
                game_dict[target_dest +  games_name + ".ulx"] = [{
                    "id" : counter,
                    "game_path" : target_dest +games_name + ".ulx",
                    "question" : questions[0],
                    "answer" : answers[0],
                    "entity" : reward_helper_info["_entities"][0],
                    "attribute" : None if not reward_helper_info["_attributes"] else reward_helper_info["_attributes"][0], 
                    }]
                counter += 1
            
            for game in games:

                prefix = game.split(".")
                
                for suffix in [".ulx", ".json", ".ni"]:
                    shutil.move(os.path.join("."+prefix[1]+suffix), target_dest)

            map_data[map_type] = game_dict

        data[question_type] = map_data
        
        

    with open("./eval_set/data.json", 'w') as f:
        json.dump(data, f)


    

if __name__ == '__main__':
    # Load in agent for config purposes
    agent = Agent()

    parser = argparse.ArgumentParser(description="Parameters for evalu")
    parser.add_argument("--games","-g",type=int,
                        default=1,)


    args = parser.parse_args()

    create_evaluation_sets(agent, variant=vars(args))


