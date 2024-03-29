import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_qait import DecisionTransformer, Trajectory, QuestionAnsweringModule
from tqdm import tqdm
from transformers import BertTokenizer

from collections import defaultdict, deque
from itertools import islice

import time
import json
import random

import re

RELATIVE_PATH = "decision_transformer/data/"

WORD_ENCODINGS = RELATIVE_PATH + "word_encodings.json"

def process_input(state, question, command, sequence_length, word2id, pad_token, tokenizer):
    """
    Input processing function. Takes in textual data and pads, truncuates, or tokenises it.

    :param state: state to be tokenised
    :param question: question that must be answered using the states. Concatenated to the end of state-strings.
    :param command: the text command or action 
    :param sequence_length: the maximum length of the state-string sequence. Used for tokenising
    :param word2id: dictionary for converting words to ids
    :param pad_token: token used for padding
    :param tokenizer: used for tokenising state-strings. Can be none if BERT is not used for QA.
    :returns: a tuple of tokenised states and actions. If BERT is used, can also return attention masks.
    """

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
    """
    Dataset object used for storing all trajectory objects. 

    :param dataset: name of dataset to be used for training
    :param sentence_length: length of the state-string context window used for action generation and answer prediction
    :param max_episodes: the maximum length of a trajectory
    :param use_bert: determine whether to use BERT for tokenising
    :param question_type: question type of data being loaded in
    """
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

            # if there is no tokenizer, use `<pad>` else use BERT tokeniser's `[PAD]` token for padding.
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
    """
    Trainer parent class.

    :param model: model to be trained
    :param optimizer: optimizer to be used for training. Weights and decay determined before passing.
    :param loss_fn: loss function to be used
    :param batch_size: batch size
    :param get_batch: method used for processing batches of data into correct format.
    :param scheduler: scheduler to be used when training
    :eval_fns: set of functions used to evaluate model at specific timesteps.
    """
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
        """
        Method for running the DT on a single batch and calculating loss.

        :returns: loss of DT when attempting to predict state, reward, and action
        """
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
    """
    Dataloader object used to store state-strings when training QA model. 

    :param dataset: name of dataset to be used for training
    :param context_window: length of the state-string context window used for question answering
    :param model_type: the type of QA model to be trained. Either 'longformer' or 'bert'
    :param use_bert: determine whether to use BERT for tokenising
    :param question_type: question type of data being loaded in
    :param decision_transformer: Decision Transformer model used for determining training set (see paper).

    """
    def __init__(
        self,
        dataset,
        context_window=180,
        question_type="location",
        model_type="bert",
        decision_transformer = None,
    ):
        offline_rl_data_filename = RELATIVE_PATH + dataset + ".json"
        word_encodings_filename = WORD_ENCODINGS
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.question_type = question_type
        self.context_window = context_window
        rng = random.Random(42)
        balancer = 0 # used to count the number of successful "Yes" trajectories in order to balance out presence of "Yes" and "No" trajectories.

        if self.question_type == "existence":
            pattern = re.compile(r"is there any (.*?) in the world \?")

        if decision_transformer is not None:
            model = torch.load(decision_transformer,map_location=device)
            model.eval()


        with open(offline_rl_data_filename) as offline_rl_data, open(word_encodings_filename) as word_encodings_data:
            
            self.dataset = []
            word_encodings = json.load(word_encodings_data)
            id2word = {key:word for word,key in word_encodings.items()}
            self.vocab_size = len(word_encodings)


            for episode_no,sample_entry in enumerate(tqdm(offline_rl_data)):

                episode = json.loads(sample_entry)
                # If a DT is passed in, use final states it predicts to stop in as context window for QA
                if model:
                    
                    if self.question_type in ["existence","attribute"]:
                        answer = int(episode["answer"])
                    elif self.question_type == "location":
                        answer = word_encodings[episode["answer"]]

                    states, actions, rtg, question = [], [], [[episode["total_reward"]]], episode["question"] 
                    for step_no,game_step in enumerate(episode["steps"]):
                        command = game_step["command"]
                        state, reward, action = game_step["state"], game_step["reward"], " ".join([command["action"],command["modifier"],command["object"]])
                        (processed_state, state_mask), (processed_command, action_mask) = process_input(
                            state=state, 
                            question=question,
                            command=action, 
                            sequence_length=model.state_dim, 
                            word2id=word_encodings, 
                            pad_token="<pad>", 
                            tokenizer=None
                        )
                        
                        states.append(processed_state)
                        actions.append(processed_command)
                        
                        act, mod, obj, _ = model.get_command(states, actions, rtg, list(range(step_no+1)), None, None, device=device)
                        
                        act, mod, obj = id2word[act.cpu().item()], id2word[mod.cpu().item()] ,id2word[obj.cpu().item()]

                        rtg.append([game_step["reward"]])
                        rtg[-1][0] = max(rtg[-2][0] - rtg[-1][0],0)

                        if "wait" in [act, mod, obj]:
                            cleaned_states = deque()
                            for step in reversed(episode["steps"][:step_no+1]):
                                cleaned_state = step["state"].replace("<s>","").replace("</s>","").replace("<|>","").replace("<pad>","").split()
                                if len(cleaned_states) + len(cleaned_state) < self.context_window*0.9:
                                    cleaned_states.extendleft(reversed(cleaned_state))
                                else:
                                    cleaned_states.extendleft(reversed(cleaned_state[int(-(len(cleaned_states) + len(cleaned_state) - self.context_window*0.9)):]))
                                    break
                            text_prompt = " ".join(cleaned_states)
                            self.dataset.append((question, text_prompt, answer))
                # If a DT is not passed in, heuristically select state-strings to train QA model on.
                else:
                    if self.question_type in ["existence","attribute"]:
                        if self.question_type == "existence":
                            entity = pattern.search(episode["question"])
                            episode["entity"] = entity.groups(0)[0]
                        answer = int(episode["answer"])
                    elif self.question_type == "location":
                        if episode["steps"][-1]["reward"] != 1.0:
                            continue

                        answer = word_encodings[episode["answer"]]
                    else:
                        raise NotImplementedError
                    

                    if model_type == "bert":

                        cleaned_states = deque()

                        # This algorithm checks for if an entity being asked about exists in the set of states of the trajectory.
                        # If so, those states that neighour the state containing the entity are added to either end of a deque
                        # until the maximum context length is reached. Thereafter, we prune either end of this pumped out state string
                        # to make sure it fits within specified context window.
                        # This allows for states to be added to the QA trainer that contain the entity in question.
                        
                        if self.question_type == "attribute" or (self.question_type == "existence" and answer == 1):
                            for i,step in enumerate(episode["steps"]):
                                if episode["entity"] in step["state"]:
                                    balancer +=1 # used to gurantee equal distr. of answer types.
                                    mid = i
                                    l, r = mid-1, mid+1

                                    cleaned_states.extend(episode["steps"][mid]["state"].replace("<s>","").replace("</s>","").replace("<|>","").replace("<pad>","").split()) 

                                    while (r < len(episode["steps"]) or l >= 0) and len(cleaned_states) < self.context_window:
                                        
                                        if r < len(episode["steps"]):
                                            cleaned_state_r = episode["steps"][r]["state"].replace("<s>","").replace("</s>","").replace("<|>","").replace("<pad>","").split()
                                            cleaned_states.extend(cleaned_state_r)
                                            r+=1

                                        if l >= 0:
                                            cleaned_state_l = episode["steps"][l]["state"].replace("<s>","").replace("</s>","").replace("<|>","").replace("<pad>","").split()
                                            cleaned_states.extendleft(reversed(cleaned_state_l))
                                            l-=1
                                    
                                    while len(cleaned_states) >= self.context_window*0.9:
                                            if cleaned_states[0] != episode["entity"]:
                                                cleaned_states.popleft()
                                            if cleaned_states[-1] != episode["entity"]:
                                                cleaned_states.pop()
                                            
                                            if cleaned_states[-1] == episode["entity"] and cleaned_states[0] == episode["entity"]:
                                                break
                                    break
                        # This algorithm appends each state string to a deque starting from the
                        # last state observed to the first state observed. The aim of such a function
                        # is to create a context string combining the last state observed with the maximum
                        # amount of previous state strings that the context_window allows for. 
                        
                        elif self.question_type == "location" or (self.question_type == "existence" and answer == 0 and balancer > 0):
                            
                            for game_step in reversed(episode["steps"]):
                                cleaned_state = game_step["state"].replace("<s>","").replace("</s>","").replace("<|>","").replace("<pad>","").split()
                                if len(cleaned_states) + len(cleaned_state) < self.context_window*0.9:
                                    cleaned_states.extendleft(reversed(cleaned_state))
                                else:
                                    cleaned_states.extendleft(reversed(cleaned_state[int(-(len(cleaned_states) + len(cleaned_state) - self.context_window*0.9)):]))
                                    break
                            balancer -= 1
                        if cleaned_states:
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
    """
    Trainer class for QA model. Training set can either be determined using Decision Transformer (1) or heuristically (2).
    
    (1) When using the DT we feed it data and allow for it to determine when to stop (by issuing the 'wait' command). The last 512 tokens observed by the DT are then used as training data for the QA model. 
    (2) A set of heuristics allows for us to determine which states to use for training that differ for each question-type.
    
    For the purposes of our paper, we used the Decision Transformer method.

    :param model: QA model to be trained
    :param optimizer: optimizer to be used for training. Weights and decay determined before passing.
    :param loss_fn: loss function to be used
    :param batch_size: batch size
    :param dataset: name of the dataset with which to load data from
    :param num_workers: number of workers used for dataloader
    :param decision_transformer: Decision Transformer model used for determining training set (see paper).
    :param get_batch: method used for processing batches of data into correct format.

    """
    def __init__(
        self,
        model : QuestionAnsweringModule, 
        optimizer,
        loss_fn : function,
        batch_size=4,
        get_batch=None,
        dataset="dqn_loc.json",
        num_workers=1,
        decision_transformer = None,
    ):
        super().__init__(model, optimizer, batch_size, get_batch, loss_fn)
        
        qa_dataset = QuestionAnsweringDataLoader(
            dataset,
            model.context_window, 
            model.question_type, 
            model.pretrained_model,
            decision_transformer
        )
        
        # calculates the training and validation set sizes.
        # context_window size used as random seed for splitting.
        train_size = round(len(qa_dataset)*0.8)
        val_size = len(qa_dataset) - train_size
        self.train_subset, self.val_subset = torch.utils.data.random_split(
                qa_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(model.context_window))

        self.dataloader = DataLoader(dataset=self.train_subset, shuffle=True, batch_size=batch_size)
        self.val_loader = DataLoader(dataset=self.val_subset, shuffle=False, batch_size=batch_size)
        
        
    def train_step(self):
        total_losses = []
        hits = 0
        for batch in self.dataloader:

            questions, prompts, answers = batch

            output = self.model.forward([p +" [SEP] "+ q for q,p in zip(questions,prompts)])
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
            output = self.model.forward([p +" [SEP] "+ q for q,p in zip(questions,prompts)])
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
    """
    Trainer class for training Decision Transformer.

    :returns: loss of action, modifier, and object prediction as well as answer

    """
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
    