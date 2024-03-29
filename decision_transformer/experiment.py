import os
import sys
sys.path.insert(0, './decision_transformer')
sys.path.append(".")

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb

import argparse
import pickle
import random
import sys
import json

from collections import deque

from evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from model_qait import DecisionTransformer, QuestionAnsweringModule
from trainer_qait import JsonDataset, SequenceTrainer, QuestionAnsweringTrainer

import evaluate
from agent import Agent

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def exponential_moving_average(curr, prev, alpha=0.7):
    return alpha*curr + (1-alpha)*prev

def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda') # cuda or cpu

    log_to_wandb = variant.get('log_to_wandb', False)
    
    random_map =  variant["random_map"]

    map_type = "random_map" if random_map else "fixed_map"

    env_name, dataset = variant['env'], map_type+ "/" + variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    bert_embeddings = True if variant['embed_type'] == "bert" else False 
    
    max_ep_len = variant["max_ep_length"]
    scale = 1.  # normalization for rewards/returns

    act_dim = 3 # act, mod, obj
    state_dim = variant["state_context_window"] # Number of tokens in the state string

    if bert_embeddings:
        act_dim += 7 # Make the action dimension 10 if bert is used
        state_dim += 20 # Add 20 tokens to the state embedding if bert is used (for subword)

    question_type = variant["question_type"]
    # load dataset
    dataset_filename = f'{dataset}.json'

    trajectories = JsonDataset(
        dataset_filename,
        state_dim,
        max_episodes=max_ep_len,
        use_bert=bert_embeddings,
        question_type=question_type,
    )

    # save all path information into separate lists
    states, traj_lens, returns,answers = [], [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, rtg, timesteps, ans, mask, state_mask, action_mask = [], [], [], [], [], [], [], [], []
        for i in range(batch_size):

            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))

            ans.append(traj['answer'][si:si + max_len].reshape(1, -1, 1))
            if 'state_mask' in traj and 'action_mask' in traj:
                state_mask.append(traj['state_mask'][si:si + max_len].reshape(1, -1, state_dim))
                action_mask.append(traj['action_mask'][si:si + max_len].reshape(1, -1, act_dim))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            ans[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), ans[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
            
            if state_mask and action_mask:
                state_mask[-1] =  np.concatenate([np.zeros((1, max_len - tlen, state_dim)), state_mask[-1]], axis=1)
                action_mask[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), action_mask[-1]], axis=1)

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.long, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.long, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        ans = torch.from_numpy(np.concatenate(ans, axis=0)).to(dtype=torch.long, device=device)
        
        if state_mask and action_mask:
            state_mask = torch.from_numpy(np.concatenate(state_mask, axis=0)).to(dtype=torch.long, device=device)
            action_mask = torch.from_numpy(np.concatenate(action_mask, axis=0)).to(dtype=torch.long, device=device)
        
        return s, a, r, rtg, timesteps, mask, ans, state_mask, action_mask

    def eval_episodes(model, iter_num=0) -> dict:
        """
        Function to be passed to Trainer for validation.
        :param mode: model to validated
        :iter_num: iteration of training loop
        :returns: dictionary of scores/accuracies of QAit evaluation
        """
        eval_qa_reward, eval_sufficient_info_reward = 0.0, 0.0
        # evaluate
        data_dir = "./"
        agent = Agent()
        
        # In the case of multiple experiments running that all use the config file, 
        # set the necessary values of the agent to that of the model.
        agent.question_type = question_type
        agent.random_map = random_map

        variant = {"decision_transformer" : True, "iter_num" : iter_num}

        eval_qa_reward, eval_sufficient_info_reward, eval_sufficient_info_reward_std = evaluate.evaluate(data_dir, agent, variant, model)

        return {
                "eval_qa_reward" : eval_qa_reward, 
                "eval_sufficient_info_reward" : eval_sufficient_info_reward ,
                "eval_sufficient_info_reward_std" : eval_sufficient_info_reward_std ,
            }

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim = act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            bert_embeddings= bert_embeddings,
            question_type=question_type,
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=torch.nn.CrossEntropyLoss(),
            eval_fns=[eval_episodes],
        )


    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
    
    print("========== Beginning Training ==========\n")
    min_loss = float("inf")
    with open(f"./decision_transformer/training_logs/{env_name}.json","w") as training_logs:

        for iter in range(variant['max_iters']):
            outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True, evaluate=(iter+1) % variant["eval_per_iter"]==0)
            
            
            if (iter+1) > variant["warmup_iterations"] and outputs["training/train_loss_mean"] <= min_loss:
                min_loss = outputs["training/train_loss_mean"]
                torch.save(model,f"{variant['model_out']}/{variant['env']}.pt")
            
            if log_to_wandb:
                wandb.log(outputs)

            outputs["iteration"] = iter+1
            print(json.dumps(outputs),file=training_logs)
            training_logs.flush()

def qa_experiment(
    exp_prefix,
    variant
):
    """
    Method for running QA model experiment.

    """
    device = variant.get('device', 'cuda') # cuda or cpu

    log_to_wandb = variant.get('log_to_wandb', False)

    random_map =  variant.get("random_map", False)
    map_type = "random_map" if random_map else "fixed_map"
    
    env_name, dataset = variant['env'],map_type +"/"+variant['dataset'] 
    model_type = variant['pretrained_model']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )

    model = QuestionAnsweringModule(
        vocab_size=variant["vocab_size"],
        hidden_size=variant["embed_dim"],
        context_window=variant["state_context_window"],
        attention_probs_dropout_prob=variant["dropout"],
        hidden_dropout_prob=variant["dropout"],
        question_type=variant["question_type"],
        gradient_checkpointing=True,
    )
    
    model = model.to(device=device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    

    decision_transformer = variant.get("load_dt", None)
    if decision_transformer is not None:
         decision_transformer = f"./{variant['model_out']}/{decision_transformer}.pt"

    trainer = QuestionAnsweringTrainer(
        model,
        optimizer,
        loss_fn, 
        batch_size=variant["batch_size"],
        num_workers=variant["num_workers"],
        dataset=dataset,
        decision_transformer=decision_transformer,
    )
    
    epochs = variant['max_iters']

    print("========== Beginning Training ==========\n")
    with open(f"./decision_transformer/training_logs/{env_name}.json","w") as training_logs:

        max_accuracy = 0
        for iter in range(epochs):
            outputs = trainer.train_iteration(iter_num=iter+1, print_logs=True)
            
            if outputs['evaluation/QA_accuracy'] >= max_accuracy:
                max_accuracy = outputs['evaluation/QA_accuracy']
                torch.save(model,f"{variant['model_out']}/{variant['env']}.pt")
            
            if log_to_wandb:
                wandb.log(outputs)
            
            outputs["iteration"] = iter+1
            print(json.dumps(outputs),file=training_logs)
            training_logs.flush()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='random_rollout')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--K', type=int, default=5) # context window of DT
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, qa for question-anwering model
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--activation_function', type=str, default='tanh')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool)
    parser.add_argument('--state_context_window', '-state', type=int, default=170) # size of context window for qa, or size of each state for dt
    parser.add_argument('--vocab_size', '-vocab', type=int, default=1654)
    parser.add_argument('--embed_type', type=str, default="normal")
    parser.add_argument('--model_out', type=str, default="./decision_transformer/saved_models")
    parser.add_argument('--eval_per_iter', type=int, default=1) # num of iterations before evaluating 
    parser.add_argument('--num_workers', type=int, default=2) # workers for dataloader
    parser.add_argument('--pretrained_model' ,type=str, default="longformer")
    parser.add_argument('--question_type', '-qt' ,type=str, default="location")
    parser.add_argument('--random_map', '-mt' ,type=bool) # map type: random or fixed
    parser.add_argument('--warmup_iterations' ,type=int, default=500) # how many iterations to wait before starting to save model
    parser.add_argument('--max_ep_length', "-len" ,type=int, default=50)
    parser.add_argument('--load_dt', "-load" ,type=str) # will DT be used in training QA model

    args = vars(parser.parse_args())

    model_type = args["model_type"]

    if model_type == "dt":
        experiment('iqa-experiment', variant=args)
    elif model_type == "qa":
        qa_experiment('iqa-experiment-qa', variant=args)
    else:
        raise NotImplementedError