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

from evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from model_qait import DecisionTransformer
from trainer_qait import JsonDataset, SequenceTrainer 

import evaluate
from agent import Agent

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda') # cuda or cpu

    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    # valid_env_names = set(['random_rollout','dqn_loc'])
    bert_embeddings = True if variant['embed_type'] == "bert" else False 
    # if env_name in valid_env_names:
    max_ep_len = 50
    env_targets = [3600, 1800]  # evaluation conditioning targets
    scale = 10.  # normalization for rewards/returns
    # else:
    #     raise NotImplementedError
    act_dim = 3 # act, mod, obj
    state_dim = variant["sentence_tensor_length"] # Number of tokens in the state string

    if bert_embeddings:
        act_dim += 7 # Make the action dimension 10 if bert is used
        state_dim += 20 # Add 20 tokens to the state embedding if bert is used (for subword)

    import os
    # print("\n".join(os.listdir()))

    # load dataset
    dataset_filename = f'{dataset}.json'
    trajectories = JsonDataset(dataset_filename,state_dim,max_episodes=max_ep_len,use_bert=bert_embeddings)

    # save all path information into separate lists
    mode = "normal"
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
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            ans[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), ans[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            # Add in masks from trajectory object?
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

    def eval_episodes(model):
        eval_qa_reward, eval_sufficient_info_reward = 0.0, 0.0
        # evaluate
        data_dir = "./"
        agent = Agent()

        variant = {"decision_transformer" : True}

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
        # wandb.watch(model)  # wandb has some bug
    print("========== Beginning Training ==========\n")
    best_sufficient_info_score = float("-inf")
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True, evaluate=(iter+1) % variant["eval_per_iter"]==0)
        
        if 'evaluation/eval_sufficient_info_reward' in outputs and outputs['evaluation/eval_sufficient_info_reward'] > best_sufficient_info_score:
            best_sufficient_info_score = outputs['evaluation/eval_sufficient_info_reward']
            torch.save(model,f"{variant['model_out']}/{variant['env']}.pt")
        
        if log_to_wandb:
            wandb.log(outputs)


    # qa = "_qa" if variant["answer_question"] else ""
    # print()
    # torch.save(model.state_dict(), f"./decision_transformer/saved_models/{variant['env']}{qa}.pt")
    # with open(f"./decision_transformer/saved_models/{variant['env']}{qa}_config.pkl", "wb") as config:
    #     pickle.dump(variant, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='random_rollout')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
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
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--sentence_tensor_length', '-sent', type=int, default=170)
    parser.add_argument('--vocab_size', '-vocab', type=int, default=1654)
    parser.add_argument('--answer_question', '-qa', type=bool, default=False)
    parser.add_argument('--embed_type', type=str, default="normal")
    parser.add_argument('--model_out', type=str, default="./decision_transformer/saved_models")
    parser.add_argument('--eval_per_iter', type=int, default=100)


    args = parser.parse_args()

    experiment('iqa-experiment', variant=vars(args))
