import argparse
import torch
import pickle

from decision_transformer import model_qait
from decision_transformer import trajectory_gpt2

import numpy as np
from textworld.generator import data
import generic
import reward_helper
import copy
import os
import json
from tqdm import tqdm
from os.path import join as pjoin
import gym
import textworld
from agent import Agent
from textworld.gym import register_games, make_batch
from query import process_facts

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


def evaluate(data_path, agent,variant):
    """
    Evaluate an agent on a test set
    """
    eval_data_path = pjoin(data_path, agent.eval_data_path)

    decision_transformer = variant["decision_transformer"]

    config = pickle.load(open(f"./{variant['model_dir']}/{variant['model']}_config.pkl", "rb"))

    print(config)
    model = model_qait.DecisionTransformer(
            state_dim=config["sentence_tensor_length"],
            act_dim = 3,
            max_length=config["K"],
            max_ep_len=50,
            hidden_size=config['embed_dim'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=4*config['embed_dim'],
            activation_function=config['activation_function'],
            n_positions=1024,
            resid_pdrop=config['dropout'],
            attn_pdrop=config['dropout'],
        )
    model.load_state_dict(torch.load(f"{variant['model_dir']}/{variant['model']}.pt"))
    model.eval()

    with open(eval_data_path) as f:
        data = json.load(f)
    data = data[agent.question_type]
    data = data["random_map"] if agent.random_map else data["fixed_map"]

    print_qa_reward, print_sufficient_info_reward = [], []
    for game_path in tqdm(data):
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
            counting_rewards = [agent.get_binarized_count(state_strings, update=True)] # batch x reward x timestep

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

                # Batch size of 1 for now
                if decision_transformer:
                    commands, replay_info = agent.act_decision_transformer(commands_per_step,[step_no]*batch_size,obs, observation_strings_w_history,questions,counting_rewards,model=model)
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
                
                state_strings = agent.get_state_strings(infos)
                counting_rewards.append(agent.get_binarized_count(state_strings, update=True))
                
                if (step_no == agent.eval_max_nb_steps_per_episode - 1 ) or (step_no > 0 and np.sum(generic.to_np(replay_info[-1])) == 0):
                    break

            # The agent has exhausted all steps, now answer question.
            answerer_input = agent.naozi.get()
            answerer_input_observation, answerer_input_observation_char, answerer_observation_ids =  agent.get_agent_inputs(answerer_input)

            chosen_word_indices = agent.answer_question_act_greedy(answerer_input_observation, answerer_input_observation_char, answerer_observation_ids, input_quest, input_quest_char)  # batch
            chosen_word_indices_np = generic.to_np(chosen_word_indices)
            chosen_answers = [agent.word_vocab[item] for item in chosen_word_indices_np]

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
            r_sufficient_info = np.mean(np.sum(sufficient_info_reward_np, -1))
            print_qa_reward.append(r_qa)
            print_sufficient_info_reward.append(r_sufficient_info)
        print("===== Eval =====: qa acc: {:2.3f} | correct state: {:2.3f}".format(np.mean(print_qa_reward), np.mean(print_sufficient_info_reward)))

        env.close()

    # print("===== Eval =====: qa acc: {:2.3f} | correct state: {:2.3f}".format(np.mean(print_qa_reward), np.mean(print_sufficient_info_reward)))
    return np.mean(print_qa_reward), np.mean(print_sufficient_info_reward)

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
    parser.add_argument("--model_dir","-md",
                    default="decision_transformer/saved_models")
    args = parser.parse_args()

    evaluate(agent=agent,data_path="./",variant=vars(args))
