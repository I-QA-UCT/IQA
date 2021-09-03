import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_qait import DecisionTransformer, Trajectory, QuestionAnsweringModule

from transformers import BertTokenizer

from collections import defaultdict, deque
from itertools import islice

import time
import json
import random

RELATIVE_PATH = "decision_transformer/data/"

WORD_ENCODINGS = RELATIVE_PATH + "word_encodings.json"

def process_input(state, question, command, sequence_length, word2id, pad_token, tokenizer):

    def command_to_id(command):
        attn_mask = None
        if tokenizer is None:
            command = command.replace("[PAD]",pad_token).split()
            if len(command) == 1:
                act, mod, obj = *command, pad_token, pad_token
            elif len(command) == 2:
                act, obj, mod  = *command, "</s>"
            else:
                act, mod, obj = command

            if mod == pad_token:
                mod = "</s>"
            command_ids = [word2id[act],word2id[mod],word2id[obj]]
        else:
            encoded = tokenizer.encode_plus(
                text=command,  # the sentence to be encoded
                add_special_tokens=False,  # Add [CLS] and [SEP]
                max_length = 10,  # maximum length of a command
                pad_to_max_length=True,  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                truncation=True # Truncates sentence if it's longer than maxi length
            )
            command_ids = encoded['input_ids']
            attn_mask = encoded['attention_mask']
        
        return command_ids, attn_mask

    def state_question_to_id(state, question, seq_len):
        attn_mask=None
        if tokenizer is not None:
            state = "[CLS] " + " ".join(state.replace("<|>","").split())  + " [SEP] " + question +  " [SEP]"
            
            encoded = tokenizer.encode_plus(
                text=state,  # the sentence to be encoded
                add_special_tokens=False,  # Add [CLS] and [SEP]
                max_length = seq_len,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                truncation=True # Truncates sentence if it's longer than maxi length
            )
            state_question_ids = encoded['input_ids']
            attn_mask = encoded['attention_mask']
        else:
            state = state + " <|> " +question

        # If word doesn't exist - return <unk>
            state_question_ids = [word2id.get(word,1) for word in state.split()]

            diff = len(state_question_ids) - seq_len

            if diff > 0:
                del state_question_ids[-(diff+1):-1]
            elif diff < 0:
                state_question_ids.extend(abs(diff)*[0])

        return state_question_ids, attn_mask

    return state_question_to_id(state,question,sequence_length), command_to_id(command)

class JsonDataset(Dataset):

    def __init__(
        self, 
        dataset,
        sentence_length=200, 
        max_episodes=50, 
        use_bert=False, 
        question_type="location",
    ):      

        self.sentence_length = sentence_length
        self.max_episodes = max_episodes
        
        self.question_type = question_type

        self.tz = None if not use_bert else BertTokenizer.from_pretrained('bert-base-uncased')
        self.trajectories = self.load(RELATIVE_PATH + dataset, WORD_ENCODINGS)
        
        # Shuffle data
        # random.seed(42)
        # random.shuffle(self.trajectories)
        # correct_trajectories_ind = set([i for i in range(len(self.trajectories)) if self.trajectories[i]["total_reward"].item() >= 1])
        # incorrct_traj_len =  round(len(correct_trajectories_ind)/correct_traj_prop * (1-correct_traj_prop))
        # incorrect_trajectories_ind = set(islice((i for i in range(len(self.trajectories)) if i not in correct_trajectories_ind),round(incorrct_traj_len)))
        # self.trajectories = [self.trajectories[i] for i in correct_trajectories_ind | incorrect_trajectories_ind]

    def __getitem__(self,index):
        return self.trajectories[index]
    
    def __len__(self):
        return len(self.trajectories)

    def __iter__(self):
        for trajectory in self.trajectories:
            yield trajectory

    def load(self, offline_rl_data_filename, word_encodings_filename):
        trajectories = []
        with open(offline_rl_data_filename) as offline_rl_data, open(word_encodings_filename) as word_encodings_data:
            

            word_encodings = json.load(word_encodings_data)
            commands = ["action","modifier","object"]

            PAD_tag = "[PAD]" if self.tz is not None else "<pad>"

            for episode_no,sample_entry in enumerate(offline_rl_data):
                
                episode = json.loads(sample_entry)

                # reward = episode["total_reward"]
                
                trajectory = Trajectory()
                
                # Terminals is a list of booleans of length {episode_max} where the first X 
                # elements are False indicatying the trajectory has not completed afterwhich
                # the list is True from X-1 until {episode_max-1} 
                completed_terminals = self.max_episodes - len(episode["steps"])
                trajectory["terminals"] = [False]*len(episode["steps"]) + [True]*completed_terminals

                if self.question_type in ["existence","attribute"]:
                    answer = int(episode["answer"]) # If existence or attribute Q, make the answer a 0 or 1
                elif self.question_type == "location":
                    answer = word_encodings[episode["answer"]]
                else:
                    raise NotImplementedError

                for game_step in episode["steps"]:
                    
                    # game_step["state"].replace("<s>","").replace("</s>","").replace("<|>","")

                    # Get the action, modifier, object triple
                    command = game_step["command"]
                    act, mod, obj = command["action"], command["modifier"], command["object"]
                    
                    # Timestep
                    timestep = game_step['step']

                    # Get reward and add it to the negative total
                    # Thus, when all steps complete reward should = 0
                    reward = game_step["reward"]

                    (observations, state_mask), (actions, action_mask) = process_input(state=game_step["state"], question=episode["question"], command=" ".join([act,mod,obj]), sequence_length=self.sentence_length, word2id=word_encodings,pad_token=PAD_tag,tokenizer=self.tz)
                    trajectory.add({"rewards" : reward, "observations" :  observations , "timesteps" : timestep, "actions" : actions,"answer" : answer})
                    # If bert embeddings are used, add the masks to the trajectory.
                    if state_mask and action_mask:
                        trajectory.add({"state_mask" : state_mask, "action_mask" : action_mask,})
                
                trajectories.append(trajectory)

        return trajectories

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, evaluate=True):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for step in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        if evaluate:
            eval_start = time.time()

            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model, iter_num=iter_num)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v
            
            logs['time/evaluation'] = time.time() - eval_start

        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds, answer_pred = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

class QuestionAnsweringDataLoader(Dataset):
    
    def __init__(
        self,
        dataset,
        context_window=180,
        question_type="location",
        model_type="bert"
    ):
        offline_rl_data_filename = RELATIVE_PATH + dataset + ".json"
        word_encodings_filename = WORD_ENCODINGS

        self.question_type = question_type
        self.context_window = context_window

        with open(offline_rl_data_filename) as offline_rl_data, open(word_encodings_filename) as word_encodings_data:
            
            self.dataset = []
            word_encodings = json.load(word_encodings_data)
            self.vocab_size = len(word_encodings)


            for episode_no,sample_entry in enumerate(offline_rl_data):

                episode = json.loads(sample_entry)
                if episode["reward"][-1] < 1.0:
                    continue
                
                if self.question_type in ["existence","attribute"]:
                    answer = int(episode["answer"]) # If existence or attribute Q, make the answer a 0 or 1
                elif self.question_type == "location":
                    answer = word_encodings[episode["answer"]]
                else:
                    raise NotImplementedError
                

                if model_type == "bert":
                    # This algorithm appends each state string to a deque starting from the
                    # last state observed to the first state observed. The aim of such a function
                    # is to create a context string combining the last state observed with the maximum
                    # amount of previous state strings that the context_window allows for. 
                    cleaned_states = deque()
                    for game_step in reversed(episode["steps"]):
                        cleaned_state = game_step["state"].replace("<s>","").replace("</s>","").replace("<|>","").replace("<pad>","").split()
                        if len(cleaned_states) + len(cleaned_state) < self.context_window*0.9:
                            cleaned_states.extendleft(reversed(cleaned_state))
                        else:
                            cleaned_states.extendleft(reversed(cleaned_state[int(-(len(cleaned_states) + len(cleaned_state) - self.context_window*0.9)):]))
                            break

                    text_prompt = " ".join(cleaned_states)
                    
                    self.dataset.append((episode["question"], text_prompt, answer))
                elif model_type == "longformer":
                    cleaned_states = []
                    for game_step in episode["steps"]:
                        cleaned_state = game_step["state"].replace("<|>","").replace("<pad>","")
                        cleaned_states.append(" ".join(cleaned_state.split()))
                    
                    # If the number of tokens is greater than 4050 then pop from the middle
                    # until the no. of states is less than or equal to 4050.
                    if len(cleaned_states) > 4050:
                        while len(cleaned_states) > 4050:
                            cleaned_states.pop((len(cleaned_states)-1)//2)

                    self.dataset.append((episode["question"] ," ".join(cleaned_states), answer))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    
class QuestionAnsweringTrainer(Trainer):

    def __init__(
        self,
        model, 
        optimizer,
        loss_fn,
        batch_size=4,
        get_batch=None,
        dataset="dqn_loc.json",
        num_workers=1
    ):
        super().__init__(model, optimizer, batch_size, get_batch, loss_fn)
        
        qa_dataset = QuestionAnsweringDataLoader(dataset, model.context_window, model.question_type, model.pretrained_model)
        train_size = round(len(qa_dataset)*0.8)
        eval_size = len(qa_dataset) - train_size
        self.train_subset, self.val_subset = torch.utils.data.random_split(
                qa_dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(model.context_window))
        
        self.dataloader = DataLoader(dataset=self.train_subset, shuffle=True, batch_size=batch_size)
        self.val_loader = DataLoader(dataset=self.val_subset, shuffle=False, batch_size=batch_size)
        
        
    def train_step(self):
        total_losses = []
        hits = 0
        for batch in self.dataloader:

            questions, prompts, answers = batch

            output = self.model.forward([p +" "+ q for q,p in zip(questions,prompts)])
            answers_tensor = torch.tensor(answers,device=self.model.device)
            loss = self.loss_fn(output,answers_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            total_losses.append(loss.detach().cpu().item()) 
            hits += (torch.argmax(output,dim=1) == answers_tensor).sum().detach()
        return total_losses, hits.cpu().item()/len(self.train_subset)


    def train_iteration(self, num_steps=0, iter_num=0, print_logs=False):


        logs = dict()

        train_start = time.time()

        self.model.train()

        train_losses, train_accuracy = self.train_step()
        if self.scheduler is not None:
            self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        hits = 0
        for batch in self.val_loader:

            questions, prompts, answers = batch
            output = self.model.forward([p +" "+ q for q,p in zip(questions,prompts)])
            answers_tensor = torch.tensor(answers,device=self.model.device)
            hits += (torch.argmax(output,dim=1) == answers_tensor).sum().detach()

        logs["evaluation/QA_accuracy"] = hits.cpu().item()/len(self.val_subset)

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/QA_accuracy'] = train_accuracy

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs



class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, rtg, timesteps, attention_mask, answer_targets, state_mask, action_mask = self.get_batch(self.batch_size)

        vocab_size = self.model.vocab_size
        
        answer_vocab_size = vocab_size if self.model.question_type == "location" else 2 # Attribute or existence

        command_target = torch.clone(actions)
        action_target,modifier_target,object_target = [command_target[:,:,i] for i in range(3)]

        action_preds, modifier_preds, object_preds, answer_preds = self.model.forward(
        states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, state_mask=state_mask, action_mask=action_mask)
        
        answer_preds = answer_preds.reshape(-1, answer_vocab_size)[attention_mask.reshape(-1) > 0]
        answer_targets = answer_targets.reshape(-1)[attention_mask.reshape(-1) > 0]

        action_preds = action_preds.reshape(-1, vocab_size)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]

        modifier_preds = modifier_preds.reshape(-1, vocab_size)[attention_mask.reshape(-1) > 0]
        modifier_target = modifier_target.reshape(-1)[attention_mask.reshape(-1) > 0]

        object_preds = object_preds.reshape(-1, vocab_size)[attention_mask.reshape(-1) > 0]
        object_target = object_target.reshape(-1)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target) + self.loss_fn(modifier_preds, modifier_target) + self.loss_fn(object_preds, object_target) + self.loss_fn(answer_preds, answer_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()
    