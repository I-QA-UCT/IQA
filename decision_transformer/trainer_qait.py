import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_qait import DecisionTransformer, Trajectory, QuestionAnsweringBert

from transformers import BertTokenizer

from collections import defaultdict

import time
import json

RELATIVE_PATH = "decision_transformer/data/"

WORD_ENCODINGS = RELATIVE_PATH + "word_encodings.json"

class JsonDataset(Dataset):

    def __init__(self,offline_rl_data_filename,sentence_length=200,max_episodes=50):      
        self.sentence_length = sentence_length
        self.max_episodes = max_episodes
        self.tz = BertTokenizer.from_pretrained('bert-base-uncased')#,unk_token="<unk>",sep_token="<|>",pad_token="<pad>",bos_token="<s>",eos_token="</s>")

        self.trajectories = self.load(RELATIVE_PATH + offline_rl_data_filename, WORD_ENCODINGS)

    def pad_input(self,sentence, seq_len):

        diff = len(sentence) - seq_len

        if diff > 0:
            del sentence[-(diff+1):-1]
        elif diff < 0:
            sentence.extend(abs(diff)*[0])

        return sentence 

    def __getitem__(self,index):
        return self.trajectories[index]
    
    def __len__(self):
        return len(self.trajectories)

    def __iter__(self):
        for trajectory in self.trajectories:
            yield trajectory

    def load(self,offline_rl_data_filename,word_encodings_filename):
        trajectories = []
        with open(offline_rl_data_filename) as offline_rl_data,open(word_encodings_filename) as word_encodings_data:
            

            word_encodings = defaultdict(lambda: 1,json.load(word_encodings_data))
            commands = ["action","modifier","object"]

            EOS_tag = "[SEP]"#word_encodings["</s>"]
            PAD_tag = "[PAD]"#word_encodings["<pad>"]

            for episode_no,sample_entry in enumerate(offline_rl_data):
                
                episode = json.loads(sample_entry)

                reward = episode["total_reward"]
                
                trajectory = Trajectory()
                
                # Terminals is a list of booleans of length {episode_max} where the first X 
                # elements are False indicatying the trajectory has not completed afterwhich
                # the list is True from X-1 until {episode_max-1} 
                completed_terminals = self.max_episodes - len(episode["steps"])
                trajectory["terminals"] = [False]*len(episode["steps"]) + [True]*completed_terminals
                
                trajectory["mask"] = episode["mask"]
                trajectory["answer"] = self.tz(episode["answer"],padding='max_length', truncation=True, max_length = 1)
                print(trajectory["answer"])

                for game_step in episode["steps"]:
                    
                    game_step["state"].replace("<s>","[CLS]").replace("</s>","[SEP]").replace("<|>","[SEP]")

                    # Get the action, modifier, object triple 
                    act, mod, obj = [game_step["command"][command].replace("</s>","[PAD]").replace("<pad>","[PAD]") for command in commands]

                    if mod == PAD_tag and obj != PAD_tag:
                        mod,obj = obj, mod

                    # Timestep
                    timestep = game_step['step']

                    # Get reward and add it to the negative total
                    # Thus, when all steps complete reward should = 0
                    reward -= game_step["reward"]

                    # trajectory.add({"rewards" : reward, "observations" : self.pad_input([word_encodings[word] for word in game_step["state"].split()],self.sentence_length),
                    #  "timesteps" : timestep, "actions" : [word_encodings[act], word_encodings[mod],word_encodings[obj]]})
                    tokenized_state = self.tz(game_step["state"],padding='max_length', truncation=True, max_length = self.sentence_length)
                    tokenized_action = self.tz(" ".join([act,mod,obj]), add_special_tokens=False,padding='max_length',truncation=True, max_length = 3)
                    trajectory.add({"rewards" : reward, "observations" :  tokenized_state["input_ids"],
                     "timesteps" : timestep, "actions" : tokenized_action["input_ids"],
                     #"token_type_ids" : tokenized_state["token_type_ids"],
                     })

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

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

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

        eval_start = time.time()

        # self.model.eval()
        # for eval_fn in self.eval_fns:
        #     outputs = eval_fn(self.model)
        #     for k, v in outputs.items():
        #         logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
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

    def __init__(self,data):
        self.offline_rl_data_filename = RELATIVE_PATH + data
        self.word_encodings_filename = WORD_ENCODINGS

        with open(self.offline_rl_data_filename) as offline_rl_data:
            
            self.dataset = []

            prompts, questions, answers = [], [], []

            for episode_no,sample_entry in enumerate(offline_rl_data):

                episode = json.loads(sample_entry)
                questions.append(episode["question"])
                answers.append(episode["answer"])

                prompt = []
                for game_step in episode["steps"]:
                    prompt.append(game_step["state"].replace("<s>","").replace("</s>","").replace("<|>",""))
                
                self.dataset.append((" ".join(prompt), episode["question"], episode["answer"]))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

        
    def __iter__(self):

        with open(self.offline_rl_data_filename) as offline_rl_data,open(self.word_encodings_filename) as word_encodings_data:
            
            encodings = json.load(word_encodings_data)
            vocab_size = len(encodings)
            choices = list(encodings.keys())


            prompts, questions, answers = [], [], []

            for episode_no,sample_entry in enumerate(offline_rl_data):

                episode = json.loads(sample_entry)
                questions.append(episode["question"])
                answers.append(episode["answer"])

                prompt = []
                for game_step in episode["steps"]:
                    prompt.append(game_step["state"].replace("<s>","").replace("</s>","").replace("<|>",""))
                
                # prompts.append(" ".join(prompt))
                # if episode_no % self.batch_size == 0:
                yield " ".join(prompt), choices, episode["question"], episode["answer"]

                # yield prompts, choices, questions, answers
                    # prompts, questions, answers = [], [], []

class QuestionAnsweringTrainer:

    def __init__(self,model, optimizer, epochs=100, steps_per_epoch=10, batch_size=4,data="dqn_loc.json" ,**kwargs):
        self.dataset = QuestionAnsweringDataLoader(data)
        self.dataloader = DataLoader(self.dataset,batch_size=batch_size,shuffle=False, **kwargs)
        self.model = model
        self.optimizer = optimizer

        self.epochs = epochs
        self.steps = steps_per_epoch

        self.choices = ["kitchen", "pantry", "livingroom", "bathroom", "bedroom",
                  "backyard", "garden", "shed", "driveway", "street",
                  "corridor", "supermarket"]
        self.choice2id = {v:i for i,v in enumerate(self.choices)}

    
    def train_step(self):
        loss = 0
        for batch in self.dataloader:

            prompts, questions, answers = batch
            output, loss = self.model.forward(prompts, self.choices, questions, [self.choice2id[answer] for answer in answers])

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
        
            loss += loss.detach().cpu().item()
        
        return loss

    def train(self):
        self.model.train()
        train_losses = []
        for epoch in range(self.epochs):
            train_loss = self.train_step()
            print(f"Epoch {epoch+1}/{self.epochs}: Train loss = {train_loss}")
            train_losses.append(train_loss)
            # if self.scheduler is not None:
            #     self.scheduler.step()



class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, rtg, timesteps, attention_mask, answer_targets, game_mask = self.get_batch(self.batch_size)

        vocab_size = self.model.vocab_size

        if self.model.answer_question:
            answer_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask)
            
            answer_preds = answer_preds.reshape(-1, vocab_size)[attention_mask.reshape(-1) > 0]
            answer_targets = answer_targets.reshape(-1)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(answer_preds[:,-1],answer_targets[:,-1])
        else:
            command_target = torch.clone(actions)
            action_target,modifier_target,object_target = [command_target[:,:,i] for i in range(3)]

            action_preds,modifier_preds,object_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask)

            action_preds = action_preds.reshape(-1, vocab_size)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]

            modifier_preds = modifier_preds.reshape(-1, vocab_size)[attention_mask.reshape(-1) > 0]
            modifier_target = modifier_target.reshape(-1)[attention_mask.reshape(-1) > 0]

            object_preds = object_preds.reshape(-1, vocab_size)[attention_mask.reshape(-1) > 0]
            object_target = object_target.reshape(-1)[attention_mask.reshape(-1) > 0]
            
            loss = self.loss_fn(action_preds,action_target) + self.loss_fn(modifier_preds,modifier_target) + self.loss_fn(object_preds,object_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()

if __name__ == "__main__":
    model = QuestionAnsweringBert()
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )
    trainer = QuestionAnsweringTrainer(model,optimizer, batch_size=1, num_workers=1, data="random_rollouts.json",)

    trainer.train()