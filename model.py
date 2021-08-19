import logging
import os
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ReLU, ModuleList
from torch_geometric.nn.conv import GATConv
from typing import Optional, Callable, List
from torch_geometric.typing import Adj
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from layers import Embedding, GATlayer, MergeEmbeddings, EncoderBlock, CQAttention, AnswerPointer, masked_softmax, NoisyLinear, Transformer


logger = logging.getLogger(__name__)


class DQN(torch.nn.Module):
    model_name = 'dqn'

    def __init__(self, config, word_vocab, char_vocab, answer_type="pointing", generate_length=3):
        super(DQN, self).__init__()
        self.config = config
        self.word_vocab = word_vocab
        self.word_vocab_size = len(word_vocab)
        self.char_vocab = char_vocab
        self.char_vocab_size = len(char_vocab)
        self.generate_length = generate_length
        self.answer_type = answer_type
        self.read_config()
        self._def_layers()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        """
        Load config file and set model parameters
        """
        # model config
        model_config = self.config['model']

        # word
        self.use_pretrained_embedding = model_config['use_pretrained_embedding']
        self.word_embedding_size = model_config['word_embedding_size']
        self.word_embedding_trainable = model_config['word_embedding_trainable']
        self.pretrained_embedding_path = "crawl-300d-2M.vec.h5"
        # char
        self.char_embedding_size = model_config['char_embedding_size']
        self.char_embedding_trainable = model_config['char_embedding_trainable']
        self.embedding_dropout = model_config['embedding_dropout']

        self.encoder_layers = model_config['encoder_layers']
        self.encoder_conv_num = model_config['encoder_conv_num']
        self.aggregation_layers = model_config['aggregation_layers']
        self.aggregation_conv_num = model_config['aggregation_conv_num']
        self.block_hidden_dim = model_config['block_hidden_dim']
        self.gat_out_dim = self.config['gat']['out_features']
        self.n_heads = model_config['n_heads']
        self.block_dropout = model_config['block_dropout']
        self.attention_dropout = model_config['attention_dropout']
        self.action_scorer_hidden_dim = model_config['action_scorer_hidden_dim']
        self.question_answerer_hidden_dim = model_config['question_answerer_hidden_dim']

        # distributional RL
        self.use_distributional = self.config['distributional']['enable']
        self.atoms = self.config['distributional']['atoms']
        self.v_min = self.config['distributional']['v_min']
        self.v_max = self.config['distributional']['v_max']

        # dueling networks
        self.dueling_networks = self.config['dueling_networks']
        self.noisy_net = self.config['epsilon_greedy']['noisy_net']

    def _def_layers(self):
        """
        Create the layers of the DQN
        """

        # word embeddings
        if self.use_pretrained_embedding:
            self.word_embedding = Embedding(embedding_size=self.word_embedding_size,
                                            vocab_size=self.word_vocab_size,
                                            id2word=self.word_vocab,
                                            dropout_rate=self.embedding_dropout,
                                            load_pretrained=True,
                                            trainable=self.word_embedding_trainable,
                                            embedding_oov_init="random",
                                            pretrained_embedding_path=self.pretrained_embedding_path)
        else:
            self.word_embedding = Embedding(embedding_size=self.word_embedding_size,
                                            vocab_size=self.word_vocab_size,
                                            trainable=self.word_embedding_trainable,
                                            dropout_rate=self.embedding_dropout)

        # char embeddings
        self.char_embedding = Embedding(embedding_size=self.char_embedding_size,
                                        vocab_size=self.char_vocab_size,
                                        trainable=self.char_embedding_trainable,
                                        dropout_rate=self.embedding_dropout)

        self.merge_embeddings = MergeEmbeddings(block_hidden_dim=self.block_hidden_dim, word_emb_dim=self.word_embedding_size, char_emb_dim=self.char_embedding_size, dropout=self.embedding_dropout)

        self.encoders = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=7, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        self.context_question_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)

        self.context_question_attention_resizer = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)

        self.aggregators = torch.nn.ModuleList([EncoderBlock(conv_num=self.aggregation_conv_num, ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim,
                                                                n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.aggregation_layers)])

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear
        self.action_scorer_shared_linear = linear_function(self.block_hidden_dim+self.gat_out_dim, self.action_scorer_hidden_dim)

        if self.use_distributional:
            if self.dueling_networks:
                action_scorer_output_size = self.atoms
                action_scorer_advantage_output_size = self.word_vocab_size * self.atoms
            else:
                action_scorer_output_size = self.word_vocab_size * self.atoms
        else:
            if self.dueling_networks:
                action_scorer_output_size = 1
                action_scorer_advantage_output_size = self.word_vocab_size
            else:
                action_scorer_output_size = self.word_vocab_size

        action_scorers = []
        for _ in range(self.generate_length):
            action_scorers.append(linear_function(self.action_scorer_hidden_dim, action_scorer_output_size))
        self.action_scorers = torch.nn.ModuleList(action_scorers)

        if self.dueling_networks:
            action_scorers_advantage = []
            for _ in range(self.generate_length):
                action_scorers_advantage.append(linear_function(self.action_scorer_hidden_dim, action_scorer_advantage_output_size))
            self.action_scorers_advantage = torch.nn.ModuleList(action_scorers_advantage)

        self.answer_pointer = AnswerPointer(block_hidden_dim=self.block_hidden_dim, noisy_net=self.noisy_net)

        if self.answer_type in ["2 way"]:
            self.question_answerer_output_1 = linear_function(self.block_hidden_dim, self.question_answerer_hidden_dim)
            self.question_answerer_output_2 = linear_function(self.question_answerer_hidden_dim, 2)

    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask):
        # node encoding: batch x num_node x hid
        # node mask: batch x num_node

        #Option one: rewrite context_question_attention
        #Option two: concat doc_encoding and GAT_encoding
        #Option three: Input GAT into Encoder block
        #Option four: Create another encoder block that takes output of M0 and concat
        #Option five: Concat output of GAT with M0

        X = self.context_question_attention(doc_encodings, q_encodings, doc_mask, q_mask)
        M0 = self.context_question_attention_resizer(X)
        M0 = F.dropout(M0, p=self.block_dropout, training=self.training)
        square_mask = torch.bmm(doc_mask.unsqueeze(-1), doc_mask.unsqueeze(1))  # batch x time x time
        for i in range(self.aggregation_layers):
            M0 = self.aggregators[i](M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)      
        return M0

    def representation_generator(self, _input_words, _input_chars):
        embeddings, mask = self.word_embedding(_input_words)  # batch x time x emb
        char_embeddings, _ = self.char_embedding(_input_chars)  # batch x time x nchar x emb
        merged_embeddings = self.merge_embeddings(embeddings, char_embeddings, mask)  # batch x time x emb
        square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x time x time
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoders[i](merged_embeddings, mask, square_mask, i * (self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc

        return encoding_sequence, mask

    def action_scorer(self, state_representation_sequence, word_masks, gat_out):
        
        state_representation, _ = torch.max(state_representation_sequence, 1)

        state_representation = torch.cat((state_representation, gat_out), dim=-1)

        hidden = self.action_scorer_shared_linear(state_representation)  # batch x hid
        hidden = torch.relu(hidden)  # batch x hid
    
        #Option: Concat to Hidden or concat to state_rep

        action_ranks = []
        for i in range(self.generate_length):
            a_rank = self.action_scorers[i](hidden)  # batch x n_vocab, or batch x n_vocab*atoms
            if self.use_distributional:
                if self.dueling_networks:
                    a_rank_advantage = self.action_scorers_advantage[i](hidden)  # advantage stream
                    a_rank = a_rank.view(-1, 1, self.atoms)
                    a_rank_advantage = a_rank_advantage.view(-1, self.word_vocab_size, self.atoms)
                    a_rank_advantage = a_rank_advantage * word_masks[i].unsqueeze(-1)
                    q = a_rank + a_rank_advantage - a_rank_advantage.mean(1, keepdim=True)  # combine streams
                else:
                    q = a_rank.view(-1, self.word_vocab_size, self.atoms)  # batch x n_vocab x atoms
                q = masked_softmax(q, word_masks[i].unsqueeze(-1), axis=-1)  # batch x n_vocab x atoms
            else:
                if self.dueling_networks:
                    a_rank_advantage = self.action_scorers_advantage[i](hidden)  # advantage stream, batch x vocab
                    a_rank_advantage = a_rank_advantage * word_masks[i]
                    q = a_rank + a_rank_advantage - a_rank_advantage.mean(1, keepdim=True)  # combine streams  # batch x vocab
                else:
                    q = a_rank   #batch x vocab
                q = q * word_masks[i]
            action_ranks.append(q)
        return action_ranks

    def answer_question(self, matching_representation_sequence, doc_mask):
        """
        Answer question based on representation
        :return prediction distribution.
        """
        square_mask = torch.bmm(doc_mask.unsqueeze(-1), doc_mask.unsqueeze(1))  # batch x time x time
        M0 = matching_representation_sequence
        M1 = M0
        for i in range(self.aggregation_layers):
             M0 = self.aggregators[i](M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        M2 = M0
        pred = self.answer_pointer(M1, M2, doc_mask)  # batch x time
        # pred_distribution: batch x time
        pred_distribution = masked_softmax(pred, m=doc_mask, axis=-1)  # 
        if self.answer_type == "pointing":
            return pred_distribution

        z = torch.bmm(pred_distribution.view(pred_distribution.size(0), 1, pred_distribution.size(1)), M2)  # batch x 1 x inp
        z = z.view(z.size(0), -1)  # batch x inp
        hidden = self.question_answerer_output_1(z)  # batch x hid
        hidden = torch.relu(hidden)  # batch x hid
        pred = self.question_answerer_output_2(hidden)  # batch x out
        pred = masked_softmax(pred, axis=-1)
        return pred

    def reset_noise(self):
        if self.noisy_net:
            self.action_scorer_shared_linear.reset_noise()
            for i in range(len(self.action_scorers)):
                self.action_scorers[i].reset_noise()
            self.answer_pointer.zero_noise()
            if self.answer_type in ["2 way"]:
                self.question_answerer_output_1.zero_noise()
                self.question_answerer_output_2.zero_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.action_scorer_shared_linear.zero_noise()
            for i in range(len(self.action_scorers)):
                self.action_scorers[i].zero_noise()
            self.answer_pointer.zero_noise()
            if self.answer_type in ["2 way"]:
                self.question_answerer_output_1.zero_noise()
                self.question_answerer_output_2.zero_noise()

# class GAT(torch.nn.Module):

#     def __init__(self, num_features, num_hidden, num_class, dropout, alpha, num_heads):
#         super(GAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [GATlayer(num_features, num_hidden, dropout, alpha, concat=False) for i in range (num_heads)]

#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
        
#         self.out_attention = GATlayer(num_hidden * num_heads, num_class, dropout, alpha, concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([attention(x, adj) for attention in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#
#          return x    

class GAT(torch.nn.Module):

    def __init__(self, num_features, num_hidden, num_class, num_heads, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = GATConv(in_channels=num_features, out_channels=num_hidden, heads=num_heads, concat=True, negative_slope=alpha, dropout=dropout)
        self.out_attention = GATConv(in_channels=num_hidden * num_heads, out_channels=num_class, heads=1, concat=False, negative_slope=alpha, dropout=dropout) #TODO Investigate heads=1
    
    def forward(self, graph_rep):
        x = graph_rep.x
        adj = graph_rep.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attentions(x, adj)
        # x = F.elu(x) #Optional nonlinearity function
        x = F.dropout(x, self.dropout, training=self.training)
        return x
        # x = self.out_attention(x, adj)
        # return F.elu(x) #For multi head attention

    #TODO: Add multi-head attention to config 

class StateNetwork(torch.nn.Module):
    # def __init__(self, action_set, params, embeddings=None):
    def __init__(self, params, embeddings=None):
        super(StateNetwork,self).__init__()
        self.params = params
        # self.action_set = action_set
        self.use_cuda = params['use_cuda']
        self.use_bert = params['use_bert']
        self.bert_size = params['bert_size']
        if self.use_bert:
            
            if self.bert_size == 'tiny':
                features = 128
            elif self.bert_size == 'mini':
                features = 256
            elif self.bert_size == 'base':
                features = 768
            else:
                features = 512 #Small or medium

            self.GAT = GAT(num_features=features, num_hidden=params['num_hidden'], num_class=params['out_features'], num_heads=params['num_heads'], dropout=params['dropout'], alpha=params['alpha'])
            self.transformer = Transformer(hidden_size=params['num_hidden'], num_layers=params['transformer']['num_layers'], transformer_heads = params['transformer']['transformer_heads'], dropout=params['transformer']['dropout'])
        else:
            self.GAT = GAT(num_features=params['gat_emb_size'], num_hidden=params['gat_hidden_size'], num_class=params['gat_out_size'], dropout=params['dropout_ratio'], alpha=params['alpha'], num_heads=params['gat_num_heads'])
            if params['qa_init']:
                self.pretrained_embeds = torch.nn.Embedding.from_pretrained(embeddings, freeze=False)
            else:
                self.pretrained_embeds = embeddings.new_tensor(embeddings.data)
            self.vocab_kge, self.vocab = self.load_files()
            self.init_state_ent_emb()
            self.fc1 = torch.nn.Linear(self.state_ent_emb.weight.size()[0] * params['gat_hidden_size'] * 1, params['gat_out_size']) #TODO:Dynamic sizing here
                  
    def init_state_ent_emb(self):
        embeddings = torch.zeros((len(self.vocab_kge), self.params['embedding_size']))
        for i in range(len(self.vocab_kge)):
            graph_node_text = self.vocab_kge[i].split('_') #Text in OpenIE extractioned entities
            graph_node_ids = []
            for w in graph_node_text:
                if w in self.vocab.keys(): 
                    if self.vocab[w] < len(self.vocab) - 2:
                        graph_node_ids.append(self.vocab[w])
                    else:
                        graph_node_ids.append(1)
                else:
                    graph_node_ids.append(1)

            if self.use_cuda:
                graph_node_ids = torch.LongTensor(graph_node_ids).cuda()
            else:
                graph_node_ids = torch.LongTensor(graph_node_ids)

            cur_embeds = self.pretrained_embeds(graph_node_ids)

            cur_embeds = cur_embeds.mean(dim=0)
            embeddings[i, :] = cur_embeds
        self.state_ent_emb = torch.nn.Embedding.from_pretrained(embeddings, freeze=False)

    def load_files(self):

        entities = {}

        #TODO: Investigate initialze_double/entity2id.tsv
        with open("entity2id.tsv", 'r') as file_output:
            for line in file_output:
                entity, entity_id = line.split('\t')
                entities[int(entity_id.strip())] = entity.strip()

        vocab = {}
        i = 0
        with open('vocabularies/word_vocab.txt', 'r') as file_output:
            for i, line in enumerate(file_output):
                vocab[line.strip()] = i

        return entities, vocab

    def forward(self, graph_rep, batch_num_nodes):
        
        if self.use_bert:
            x = self.GAT(graph_rep)

            count = 0
            graph_list = []
            for num in batch_num_nodes:
                graph = x[count:count+num,:]
                graph_list.append(graph)
                count += num

            batch, masks = self.transformer.pad(graph_list)
            out = self.transformer.encoder(src=batch, src_key_padding_mask=masks).permute(1,0,2)
            out = out[:,-1,:]

            return out
        else:
            _, adj = graph_rep
            if len(adj.size()) == 2:
                adj = adj.unsqueeze(0)
            batch_size = len(adj)
            x = self.GAT(self.state_ent_emb.weight, adj).view(batch_size, -1)
            out = self.fc1(x)
        return out