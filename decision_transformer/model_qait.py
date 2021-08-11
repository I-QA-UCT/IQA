import numpy as np
import torch
import torch.nn as nn

from transformers import GPT2Config
from trajectory_gpt2 import GPT2Model

from collections import defaultdict


class DecisionTransformer(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            vocab_size = 1654,
            max_length=None,
            max_ep_len=50,
            action_tanh=True,
            **kwargs
    ):
        super(DecisionTransformer,self).__init__()

        
        self.hidden_size = hidden_size
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.max_ep_len = max_ep_len
        self.vocab_size = vocab_size
        self.max_length = max_length

        config = GPT2Config(
            vocab_size=self.vocab_size,
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)

        self.word_embedding = torch.nn.Embedding(self.vocab_size,hidden_size)
        # When done try embeddig actions seperately

        # Single GRU for action and state
        self.encoder = torch.nn.GRU(hidden_size,hidden_size,batch_first=True)


        self.embed_ln = nn.LayerNorm(hidden_size)
        
        self.predict_answer = nn.Sequential(
            *([nn.Linear(hidden_size, self.vocab_size)]+ ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.vocab_size)]+ ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_modifer = nn.Sequential(
            *([nn.Linear(hidden_size, self.vocab_size)] + ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_object = nn.Sequential(
            *([nn.Linear(hidden_size, self.vocab_size)] + ([nn.Tanh()] if action_tanh else []))
        )
        
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        # word masks - possible words for decoder and then use masked softmax to get actual distribution

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # states will be a tensor of word ids of the state i.e batch x sequence (number of words in observation) 
        # actions will be a tensor of word ids of the state i.e batch x 3 

        # embed each modality with a different

        # Embded state with GRU
        encoded_states = []
        state_word_embeddings = self.word_embedding(states)
        for batch in range(len(state_word_embeddings)):
            encoded_state,_ = self.encoder(state_word_embeddings[batch])
            encoded_states.append(encoded_state[:,-1,:])
        
        encoded_state=torch.stack(encoded_states)

        # Embded action with GRU
        encoded_actions = []
        action_word_embeddings = self.word_embedding(actions)
        for batch in range(len(action_word_embeddings)):
            encoded_action,_ = self.encoder(action_word_embeddings[batch])
            encoded_actions.append(encoded_action[:,-1,:])
        
        encoded_action=torch.stack(encoded_actions)

        state_embeddings = encoded_state 
        action_embeddings = encoded_action
        returns_embeddings = self.embed_return(returns_to_go) 
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings 
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        

    
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:,1])  # predict next action given state
        modifier_preds = self.predict_modifer(x[:,1])  # predict next action given state
        object_preds = self.predict_object(x[:,1])  # predict next action given state
        answer_pred = self.predict_answer(x[:,1]) # predict answer

        return action_preds, modifier_preds, object_preds, answer_pred

    def get_command(self, states, actions, returns_to_go, timesteps, **kwargs):
        
        # print(states, actions, returns_to_go, timesteps,sep="\n")
        
        states = torch.Tensor(states).reshape(1, -1, self.state_dim)
        actions = torch.Tensor(actions).reshape(1, -1, self.act_dim)
        returns_to_go = torch.Tensor(returns_to_go).reshape(1, -1, 1)
        timesteps = torch.Tensor(timesteps).reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.long)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.long)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds, modifier_preds, object_preds, answer_pred = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return torch.argmax(action_preds[0,-1]), torch.argmax(modifier_preds[0,-1]), torch.argmax(object_preds[0,-1]), torch.argmax(answer_pred[0,-1])

class Trajectory(object):

    def __init__(self):
        self.trajectory = defaultdict(list)

    def add(self,sequence):
        for key, arg in sequence.items():
            self.trajectory[key].append(arg)

    def __contains__(self, key):
        if key in self.trajectory: 
            return True
    
    def __getitem__(self,key):
        return torch.FloatTensor(self.trajectory[key])

    def __setitem__(self,key,item):
        self.trajectory[key] = item

    def __len__(self):
        return len(self.trajectory)

    def __str__(self):
        return "\n".join([f"{key}: {self.trajectory[key]} ({type(self.trajectory[key])})" for key in self.trajectory])
    