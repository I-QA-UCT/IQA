import argparse
import torch
import pickle
import random
import time

import sys
sys.path.insert(0, './decision_transformer')


from model_qait import DecisionTransformer
from trainer_qait import process_input

import numpy as np
from textworld.generator import data
import generic
import reward_helper
import copy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from tqdm import tqdm
from os.path import join as pjoin
import gym
import textworld
from agent import Agent
from textworld.gym import register_games, make_batch
from query import process_facts

from collections import Counter
from itertools import islice

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


def evaluate(data_path : str, agent : Agent, variant : dict , model=None) -> tuple:
    """
    Evaluate an agent on a set of online TextWorld games for the QAit task.

    :param data_path: The working directory of the dataset.
    :param agent: The agent that will be interacting in TextWorld. Used also for config file.
    :param variant: command line arguments supplied to evaluate's main method parsed into a dictionary for accessing.
    :param model: external model that is passed in to evaluate function. Used mainly for decision transformer validation set.
    :returns: tuple of QA accuracy and sufficient information score of agent/model on test/validation set.
    :raises AssertionError: If a model is loaded in and has a different question type to the config file.
    :raises NotImplementedError:    (1) If a question type is not location, existence, or attribute. 
                                    (2) If the number of episodes to played is not between 1-500. 
                                    (3) If the specified reward type is not -1 OR greater than 0. 
    """
    eval_data_path = pjoin(data_path, agent.eval_data_path)

    decision_transformer = variant["decision_transformer"]
    qa_model = variant['qa_model']
    if decision_transformer:
        # Loading in decision transformer for action prediction.
        with open("decision_transformer/data/word_encodings.json") as word_encodings_data:
            word_encodings = json.load(word_encodings_data)
        
        if model is None:
            model = torch.load(f"{variant['model_dir']}/{variant['model']}.pt",map_location=torch.device('cpu'))
        

        model.eval()

        assert model.question_type == agent.question_type, f"Incorrect question_type loaded from agent's config. Expected '{model.question_type}' but found '{agent.question_type}'."


        # Add model to cuda (if compatible)
        device = "cuda" if agent.use_cuda else "cpu"
        model = model.to(device=device)

        # if a QA model is passed to the evaluate function, load it in.
        if qa_model is not None:
            qa_model = torch.load(f"{variant['model_dir']}/{variant['qa_model']}.pt",map_location=torch.device('cpu'))
            qa_model.eval()
            qa_model.device = device
            # if the QA model has a different question type to that of the config file, throw an assertion exception.
            assert qa_model.question_type == agent.question_type, f"Incorrect question_type loaded from agent's config. Expected '{model.question_type}' but found '{agent.question_type}'."

            qa_model = qa_model.to(device=device)

        # Create randon numpy number generator for sampling reward.
        np_rng = np.random.RandomState(agent.random_seed +  variant.get("iter_num",0))  # can be called without a seed

    with open(eval_data_path) as f:
        data = json.load(f)
    data = data[agent.question_type]
    data = data["random_map"] if agent.random_map else data["fixed_map"]

    print_qa_reward, print_sufficient_info_reward = [], []

    # if a seperate QA model is loaded in, create a list that will store its answer predictions.
    if qa_model:
        print_qa_reward_bert = []
    # The number of episodes to be evaluated upon.
    num_eval_episodes = agent.config["evaluate"]["eval_nb_episodes"]
    
    # Convert data to list so that it can be cut and/or shuffled (python 3.6 constraint)
    data = list(data.items())

    # Shuffles the order of evaluation.
    if agent.config["evaluate"]["shuffle"]:
        rng = random.Random(agent.config["general"]["random_seed"])
        rng.shuffle(data)
    
    # Cannot evaluate on 0 or more than 500 games. 
    if 1 <= num_eval_episodes <= 500:
        data = dict(data[:num_eval_episodes])
    else:
        raise NotImplementedError

    for game_path in tqdm(data, disable=agent.config["evaluate"]["silent"]):
        game_file_path = pjoin(data_path, game_path)
        assert os.path.exists(game_file_path), "Oh no! game path %s does not exist!" % game_file_path
        env_id = register_games([game_file_path], request_infos=request_infos)
        env_id = make_batch(env_id, batch_size=agent.eval_batch_size, parallel=True)
        env = gym.make(env_id)

        data_questions = [item["question"] for item in data[game_path]]
        data_answers = [item["answer"] for item in data[game_path]]
        data_entities = [item["entity"] for item in data[game_path]]
        if agent.question_type == "attribute":
            data_attributes = [item["attribute"] for item in data[game_path]]

        for q_no in range(len(data_questions)):
            questions = data_questions[q_no: q_no + 1]
            answers = data_answers[q_no: q_no + 1]
            reward_helper_info = {"_entities": data_entities[q_no: q_no + 1],
                                  "_answers": data_answers[q_no: q_no + 1]}
            if agent.question_type == "attribute":
                reward_helper_info["_attributes"] = data_attributes[q_no: q_no + 1]

            obs, infos = env.reset()
            batch_size = len(obs)
            agent.eval()
            agent.init(obs, infos)
            # get inputs
            commands, last_facts, init_facts = [], [], []
            commands_per_step, game_facts_cache = [], []
            for i in range(batch_size):
                commands.append("restart")
                last_facts.append(None)
                init_facts.append(None)
                game_facts_cache.append([])
                commands_per_step.append(["restart"])

            observation_strings, possible_words = agent.get_game_info_at_certain_step(obs, infos)
            observation_strings = [a + " <|> " + item for a, item in zip(commands, observation_strings)]
            input_quest, input_quest_char, _ = agent.get_agent_inputs(questions)

            transition_cache = []

            state_strings = agent.get_state_strings(infos)
            if agent.config["evaluate"]['initial_reward'] > 0: 
                initial_reward = agent.config["evaluate"]['initial_reward']
            elif agent.config["evaluate"]['initial_reward'] == -1:    
                initial_reward = np_rng.exponential(0.4)+1
            else:
                raise NotImplementedError

            rewards = [[initial_reward]] # batch x reward x timestep
            padded_state_history = []
            state_masks = []
            padded_command_history = []
            action_masks = []

            states_history = []
            for step_no in range(agent.eval_max_nb_steps_per_episode):
                # update answerer input
                for i in range(batch_size):
                    if agent.not_finished_yet[i] == 1:
                        agent.naozi.push_one(i, copy.copy(observation_strings[i]))
                    if agent.prev_step_is_still_interacting[i] == 1:
                        new_facts = process_facts(last_facts[i], infos["game"][i], infos["facts"][i], infos["last_action"][i], commands[i])
                        game_facts_cache[i].append(new_facts)  # info used in reward computing of existence question
                        last_facts[i] = new_facts
                        if step_no == 0:
                            init_facts[i] = copy.copy(new_facts)

                observation_strings_w_history = agent.naozi.get()
                input_observation, input_observation_char, _ =  agent.get_agent_inputs(observation_strings_w_history)

                # Batch size of 1
                if decision_transformer:
                    (processed_state, state_mask), (processed_command, action_mask) = process_input(
                        state=observation_strings_w_history[-1], 
                        question=questions[q_no],command=commands_per_step[0][-1], 
                        sequence_length=model.state_dim, 
                        word2id=word_encodings, 
                        pad_token="<pad>", 
                        tokenizer=None
                    )
                    
                    padded_state_history.append(processed_state)
                    padded_command_history.append(processed_command)
                    if state_mask and action_mask:
                        state_masks.append(state_mask)
                        action_masks.append(action_mask)
                    
                    states_history.append(observation_strings_w_history[-1])
                    
                    commands, replay_info, answer = agent.act_decision_transformer(padded_command_history,list(range(step_no+1)),obs, padded_state_history, rewards,model=model)

                    if "wait" in commands:
                        step_no = agent.eval_max_nb_steps_per_episode - 1

                else:
                    commands, replay_info = agent.act(obs, infos, input_observation, input_observation_char, input_quest, input_quest_char, possible_words, random=False)

                for i in range(batch_size):
                    commands_per_step[i].append(commands[i])

                replay_info = [observation_strings_w_history, questions, possible_words] + replay_info
                transition_cache.append(replay_info)

                obs, _, _, infos = env.step(commands)
                # possible words no not depend on history, because one can only interact with what is currently accessible
                observation_strings, possible_words = agent.get_game_info_at_certain_step(obs, infos)
                observation_strings = [a + " <|> " + item for a, item in zip(commands, observation_strings)]
                
                reward_helper_info["observation_before_finish"] = agent.naozi.get()

                # Reward needs to be given while the DT is interacting in TextWorld.
                # functions from reward_helper are edited to allow for intermediate 
                # rewards to be calculated and given.

                if agent.question_type == "location":
                    sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_location_during(reward_helper_info)
                elif agent.question_type == "attribute":
                    reward_helper_info["answers"] = answers
                    reward_helper_info["game_facts_per_step"] = game_facts_cache  # facts before and after issuing commands (we want to compare the differnce)
                    reward_helper_info["init_game_facts"] = init_facts
                    reward_helper_info["full_facts"] = infos["facts"]
                    reward_helper_info["commands_per_step"] = commands_per_step  # commands before and after issuing commands (we want to compare the differnce)
                    sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_attribute_during(reward_helper_info)
                elif agent.question_type == "existence":
                    reward_helper_info["answers"] = answers
                    reward_helper_info["game_facts_per_step"] = game_facts_cache  # facts before issuing command (we want to stop at correct state)
                    reward_helper_info["init_game_facts"] = init_facts
                    reward_helper_info["full_facts"] = infos["facts"]
                    sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_existence_during(reward_helper_info)

                state_strings = agent.get_state_strings(infos)
                rewards.append(sufficient_info_reward_np)
                rewards[-1][0] = max(rewards[-2][0] - rewards[-1][0],0)
                if (step_no == agent.eval_max_nb_steps_per_episode - 1 ) or (step_no > 0 and np.sum(generic.to_np(replay_info[-1])) == 0):
                    break

            # The agent has exhausted all steps, now answer question.
            answerer_input = agent.naozi.get()
            answerer_input_observation, answerer_input_observation_char, answerer_observation_ids =  agent.get_agent_inputs(answerer_input)

            if not decision_transformer:
                chosen_word_indices = agent.answer_question_act_greedy(answerer_input_observation, answerer_input_observation_char, answerer_observation_ids, input_quest, input_quest_char)  # batch
            
                chosen_word_indices_np = generic.to_np(chosen_word_indices)
                chosen_answers = [agent.word_vocab[item] for item in chosen_word_indices_np]
            else:
                # if we passed in a QA model, use it to answer question.
                if qa_model:

                    chosen_answers_bert = [agent.answer_question_transformer(states_history,questions[q_no],qa_model)]
                    qa_reward_np_bert = reward_helper.get_qa_reward(answers, chosen_answers_bert)
                # make the chosen answer the final answer prediction of the DT.
                chosen_answers = [answer]

            # rewards
            # qa reward
            qa_reward_np = reward_helper.get_qa_reward(answers, chosen_answers)

            # sufficient info rewards
            masks = [item[-1] for item in transition_cache]
            masks_np = [generic.to_np(item) for item in masks]
            # 1 1 0 0 0 --> 1 1 0 0 0 0
            game_finishing_mask = np.stack(masks_np + [np.zeros((batch_size,))], 0)  # game step+1 x batch size
            # 1 1 0 0 0 0 --> 0 1 0 0 0
            game_finishing_mask = game_finishing_mask[:-1, :] - game_finishing_mask[1:, :]  # game step x batch size
            
            if agent.question_type == "location":
                # sufficient info reward: location question
                reward_helper_info["observation_before_finish"] = answerer_input
                reward_helper_info["game_finishing_mask"] = game_finishing_mask
                sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_location(reward_helper_info)
            elif agent.question_type == "existence":
                # sufficient info reward: existence question
                reward_helper_info["observation_before_finish"] = answerer_input
                reward_helper_info["game_facts_per_step"] = game_facts_cache  # facts before issuing command (we want to stop at correct state)
                reward_helper_info["init_game_facts"] = init_facts
                reward_helper_info["full_facts"] = infos["facts"]
                reward_helper_info["answers"] = answers
                reward_helper_info["game_finishing_mask"] = game_finishing_mask
                sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_existence(reward_helper_info)
            elif agent.question_type == "attribute":
                # sufficient info reward: attribute question
                reward_helper_info["answers"] = answers
                reward_helper_info["game_facts_per_step"] = game_facts_cache  # facts before and after issuing commands (we want to compare the differnce)
                reward_helper_info["init_game_facts"] = init_facts
                reward_helper_info["full_facts"] = infos["facts"]
                reward_helper_info["commands_per_step"] = commands_per_step  # commands before and after issuing commands (we want to compare the differnce)
                reward_helper_info["game_finishing_mask"] = game_finishing_mask
                sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_attribute(reward_helper_info)
            else:
                raise NotImplementedError

            r_qa = np.mean(qa_reward_np)

            if qa_model:
                r_qa_bert = np.mean(qa_reward_np_bert)
                print_qa_reward_bert.append(r_qa_bert)

            r_sufficient_info = np.mean(np.sum(sufficient_info_reward_np, -1))
            print_qa_reward.append(r_qa)
            print_sufficient_info_reward.append(r_sufficient_info)

        env.close()

    if not agent.config["evaluate"]["silent"]:
        if not qa_model:
            print("===== Eval =====: qa acc: {:2.3f} | correct state: {:2.3f}".format(np.mean(print_qa_reward), np.mean(print_sufficient_info_reward)))    
        else:
            print(f"===== Eval =====: qa acc: {np.mean(print_qa_reward)} | bert qa acc {np.mean(print_qa_reward_bert)} | correct state: {np.mean(print_sufficient_info_reward)}")
    
    if not qa_model:
        return np.mean(print_qa_reward), np.mean(print_sufficient_info_reward), np.std(print_sufficient_info_reward)
    else:
        return np.mean(print_qa_reward), np.mean(print_qa_reward_bert), np.mean(print_sufficient_info_reward), np.std(print_sufficient_info_reward)

def evaluate_all(data_path, variant):
    """
    Evaluates multiple configurations and writes results/data to a logs file. 
    """
    max_train = {
        "fixed" : {"location" : 4.1 , "existence" : 3.8 , "attribute" : 3.73},
        "random" : {"location" : 4.1 , "existence" : 3.94 , "attribute" : 4.03}
    }

    question_types = ["location", "attribute", "existence"]
    random_map_types = [False, True]
    
    random_seeds = [42, 84, 168, 336, 672]

    initial_rewards = [1,2,3,4,5,-1,"max_train"]
    with open(f"{variant['model_out']}/eval_data.csv", "a") as out:
        
        print("question_type,random_map,rtg,bert_qa,dt_qa,suf_mean,suf_std,time,seed",file=out)

        for seed in random_seeds:

            for question_type in question_types:

                for random_map in random_map_types:
                    
                    map_type = "random_map" if random_map else "fixed_map"
                    variant["qa_model"] = f"{question_type}-500-{map_type}-qa-module"
                    variant["model"] = f"{question_type}-500-{map_type}"

                    for rtg in initial_rewards:
                        
                        agent = Agent()

                        if rtg == "max_train":
                            rtg = max_train["random" if random_map else "fixed"][question_type]
                        
                        agent.config["evaluate"]['initial_reward'] = rtg
                        agent.random_map = random_map
                        agent.question_type = question_type
                        agent.eval_folder = pjoin(agent.testset_path, question_type, (map_type))
                        agent.eval_data_path = pjoin(agent.testset_path, "data.json")
                        agent.random_seed = seed
                        np.random.seed(agent.random_seed)
                        torch.manual_seed(agent.random_seed)

                        start = time.time()
                        dt_qa, bert_qa, suf_mean, suf_std = evaluate(data_path, agent, variant)

                        end = time.time() - start

                        print(f"{question_type},{random_map},{rtg},{bert_qa},{dt_qa},{suf_mean},{suf_std},{end},{seed}",file=out)



            


if (__name__ == "__main__"):
    agent = Agent()

    output_dir, data_dir = ".", "."
    
    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt")
            agent.update_target_net()
        elif os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()
        else:
            print("Failed to load pretrained model... couldn't find the checkpoint file...")

    parser = argparse.ArgumentParser(description="Evaluate using decision transformer.")
    parser.add_argument("--decision_transformer","-dt",
                    default=True)
    parser.add_argument("--model","-m",
                    default="random_rollout")
    parser.add_argument("--qa_model","-qa",
                    default=None)
    parser.add_argument("--model_dir","-md",
                    default="decision_transformer/saved_models")
    parser.add_argument("--model_out","-mo",
                    default="decision_transformer/training_logs")
    parser.add_argument("--evaluate_all","-all", type=bool)
    args = vars(parser.parse_args())

    eval_all = args.get("evaluate_all",False)
    
    if eval_all:
        evaluate_all(data_path="./", variant=args)
    else:

        evaluate(agent=agent,data_path="./",variant=args)
