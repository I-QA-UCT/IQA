import numpy as np
import torch
import torch.nn as nn

from transformers import BertTokenizerFast, BertModel, LongformerModel, LongformerTokenizerFast
from transformers import GPT2Config
from trajectory_gpt2 import GPT2Model


from collections import defaultdict, deque

class DecisionTransformer(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    
    :param state_dim: dimension of state or length of state-string to be used as context for action generation
    :param act_dim: action dimension.

    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=50,
            action_tanh=True,
            answer_question = False,
            vocab_size = 1654,
            bert_embeddings = True,
            question_type = "location",
            **kwargs
    ):
        super(DecisionTransformer,self).__init__()

        
        self.hidden_size = hidden_size
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.max_ep_len = max_ep_len
        self.vocab_size = vocab_size if not bert_embeddings else 30522
        self.max_length = max_length

        self.answer_question = answer_question
        self.question_type = question_type

        self.bert_embeddings = bert_embeddings

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

        # Single GRU for action and state.
        # If BERT is used for embedding, set the hidden size to 768 (default BERT size) else, 
        # use the passed in hidden size.
        self.encoder = torch.nn.GRU(768 if self.bert_embeddings else hidden_size,hidden_size,batch_first=True)
        
        # BERT for encoding
        self.bert = BertModel.from_pretrained('bert-base-uncased') #,unk_token="<unk>",sep_token="<|>",pad_token="<pad>",bos_token="<s>",eos_token="</s>")

        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # If location type question, answer prediction will be calculated over the entire vocab.
        if self.question_type == "location":
            self.predict_answer = nn.Sequential(
                *([nn.Linear(hidden_size, self.vocab_size)])
            )
        # If existence or attribute, answer prediction will be of size 2, either 1 (yes) or 0 (no).
        elif self.question_type in ["existence", "attribute"]:
            self.predict_answer = nn.Sequential(
                *([nn.Linear(hidden_size, 2)])
            )
        else:
            raise NotImplementedError
        
        # action, modifier, and object prediction heads. Predictions to be softmaxed over the entire vocab.
        self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.vocab_size)])
            )
        
        self.predict_modifer = nn.Sequential(
            *([nn.Linear(hidden_size, self.vocab_size)])
        )

        self.predict_object = nn.Sequential(
            *([nn.Linear(hidden_size, self.vocab_size)])
        )
        
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, state_mask=None, action_mask=None, device="cuda") -> tuple:
        """
        Forward propagation method for Decision Transformer. Takes in a tensor of states, actions, rewards, returns_to_go
        and timesteps for action generation. Attention masks are also passed in.

        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        # word masks - possible words for decoder and then use masked softmax to get actual distribution

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)

        # states will be a tensor of word ids of the state i.e batch x sequence (number of words in observation) 
        # actions will be a tensor of word ids of the state i.e batch x 3 

        # embed each modality with a different
        # Embded state with GRU
        encoded_states = []
        
        # Either embed state tokens with BERT or use seperate word_embedding layer
        if self.bert_embeddings:
            state_word_embeddings = []
            for batch in range(len(states[0])):
                embedded_state = self.bert(input_ids=states[:,batch,:],attention_mask=state_mask[:,batch,:])
                state_word_embeddings.append(embedded_state["last_hidden_state"])
            state_word_embeddings = torch.stack(state_word_embeddings,dim=1)
        else:
            state_word_embeddings = self.word_embedding(states)

        for batch in range(len(state_word_embeddings)):
            encoded_state,_ = self.encoder(state_word_embeddings[batch])
            encoded_states.append(encoded_state[:,-1,:])
        
        encoded_state=torch.stack(encoded_states)
        
        # Either embed action tokens with BERT or use seperate word_embedding layer
        encoded_actions = []
        if self.bert_embeddings:
            action_word_embeddings = []
            for batch in range(len(actions[0])):
                embedded_action = self.bert(input_ids=actions[:,batch,:],attention_mask=action_mask[:,batch,:])
                action_word_embeddings.append(embedded_action["last_hidden_state"])
            action_word_embeddings = torch.stack(action_word_embeddings,dim=1)
        else:
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

        return self.predict(x)

    def predict(self, encoded_sequence) -> tuple:
        
        """
        Function that takes in the GRU encoded sequence and returns predictions. 
        :param encoded_sequence: Transformer output for a particular timestep t.
        :returns: tuple consisting of embedding vectors for the action, modifier, and object of command as well the predicted answer.
        """
        encoded_returns, encoded_states, encoded_actions = encoded_sequence[:,0],encoded_sequence[:,1],encoded_sequence[:,2]
        
        answer_preds = self.predict_answer(encoded_states) # predict answer given state
        action_preds = self.predict_action(encoded_states)  # predict next action given state
        modifier_preds = self.predict_modifer(encoded_states)  # predict next modifier given state
        object_preds = self.predict_object(encoded_states)  # predict next object given state

        return action_preds, modifier_preds, object_preds, answer_preds


    def get_command(self, states, actions, returns_to_go, timesteps, state_mask, action_mask, device="cpu",**kwargs):
        """
        Uses the Decision Transformer model to predict an action, modifier, object triple as well as an answer prediction. 

        :param states: list of tokenised states for each timestep t
        :param actions: list of tokenised actions for each timestep t
        :param returns_to_go: list of returns_to_go for each timestep t
        :param timesteps: list of timesteps for positional embedding
        :param state_mask: attention mask for states
        :param action_mask: attention mask for actions
        :param device: the device with which the tensors will be stored (`cuda` or `cpu`)
        :param `**kwargs`: key word arguments used for DT's forward method.
        """
        if state_mask and action_mask:
            state_masks = torch.Tensor(state_mask).reshape(1, -1, self.state_dim).long().to(dtype=torch.long, device=device)
            action_masks = torch.Tensor(action_mask).reshape(1, -1, self.act_dim).long().to(dtype=torch.long, device=device)
        else:
            state_masks = None
            action_masks = None
        states = torch.Tensor(states).reshape(1, -1, self.state_dim).long().to(dtype=torch.long, device=device)
        actions = torch.Tensor(actions).reshape(1, -1, self.act_dim).long().to(dtype=torch.long, device=device)
        returns_to_go = torch.Tensor(returns_to_go).reshape(1, -1, 1).to(device=device)
        timesteps = torch.Tensor(timesteps).reshape(1, -1).long().to(dtype=torch.long, device=device)

        attention_mask = None
            
        action_preds, modifier_preds, object_preds, answer_pred = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=None, state_mask=state_masks, action_mask=action_masks, device=device, **kwargs)

        # As discussed in the paper, word for actions, modifiers, and objects are sampled from a probability
        # distribution as opposed to the index corresponding to the highest logit being returned. 
        softmax = nn.Softmax(dim=0)
        action_dist = torch.distributions.categorical.Categorical(softmax(action_preds[-1,-1]))
        modifier_dist = torch.distributions.categorical.Categorical(softmax(modifier_preds[-1,-1]))
        object_dist = torch.distributions.categorical.Categorical(softmax(object_preds[-1,-1]))
        
        return action_dist.sample(), modifier_dist.sample(), object_dist.sample(), torch.argmax(answer_pred[-1,-1])
        
class QuestionAnsweringModule(nn.Module):
    """
    Question answering module for QAit task

    :param vocab_size: size of the vocab 
    :param hidden_size: embedding dimension
    :param context_window: the number of tokens to be used as context for question-answering
    :param question_type: the type of question being answered
    :param pretrained_model: name of pretrained model to be used for QA ('bert' or 'longformer')
    :param `**kwargs`: key word arguments for longformer
    :raises NotImplementedError: if a question_type is not equal to attribute, location, or existence
    """


    def __init__(
        self, 
        vocab_size, 
        hidden_size=64,
        context_window=200,
        question_type="location", 
        pretrained_model="bert",
        **kwargs
    ):
        super(QuestionAnsweringModule,self).__init__()
        
        self.context_window = context_window
        self.question_type = question_type
        self.pretrained_model = pretrained_model

        if self.pretrained_model == "bert":
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        elif self.pretrained_model == "longformer":
            max_length = 4096
            self.model = LongformerModel.from_pretrained(
                'allenai/longformer-base-4096', 
                output_hidden_states=True, 
                attention_window = context_window,
                max_length = max_length,
                num_labels=vocab_size,
                **kwargs,
            )
            
            self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = max_length, **kwargs)

        self.hidden_size = hidden_size

        self.context_window = context_window
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # if question type is location, set vocab size to pre-determined size
        if self.question_type == "location":
            self.vocab_size = vocab_size
        # else if it is attribute or existence, set vocab_size to 2
        elif self.question_type in ["attribute", "existence"]:
            self.vocab_size = 2
        else:
            raise NotImplementedError

        self.out = nn.Linear(self.model.config.hidden_size, self.vocab_size)
    
    def forward(self, prompt_questions):
        """
        Forward method for QA model.

        :param prompt_questions: list of questions and text pairs used for question-answering.  
        """

        encoding = self.tokenizer(
            prompt_questions, 
            max_length=self.context_window, 
            truncation=True,padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )

        output = self.model(input_ids=encoding['input_ids'].to(device=self.device),
                attention_mask=encoding['attention_mask'].to(device=self.device))
        
        return self.out(output["pooler_output"].to(device=self.device))
    
    def predict(self, states, question):
        """
        Answer prediction method for QA model. 
        Cleans the state-string of all unnessesary symbols and truncates it to 90% of the context_window.
        This is due to certain words being broken up into multiple tokens when using the BERT tokeniser
        nessecitating the context window not be filled up.

        :param states: list of state-strings
        :param question: question that needs to be answered using list of state-strings as context.
        :returns: index from vocab of expected answer.
        """

        # This algorithm appends each state-string to a deque starting from the
        # last state observed to the first state observed. The aim of such a function
        # is to create a context string combining the last state observed with the maximum
        # amount of previous state strings that the context_window allows for.
        for state in reversed(states):
            cleaned_state = state.replace("<s>","").replace("</s>","").replace("<|>","").replace("<pad>","").split()
            if len(cleaned_states) + len(cleaned_state) < self.context_window*0.9:
                cleaned_states.extendleft(reversed(cleaned_state))
            else:
                cleaned_states.extendleft(reversed(cleaned_state[int(-(len(cleaned_states) + len(cleaned_state) - self.context_window*0.9)):]))
                break
        text_prompt = " ".join(cleaned_states)

        answer_preds = self.forward(text_prompt +" [SEP] "+question)
        return torch.argmax(answer_preds[-1])

class Trajectory(object):
    """
    Trajectory object. Used to store information about offline reinforcement learning trajectories.
    """

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
    

