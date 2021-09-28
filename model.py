import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv

from layers import Embedding, MergeEmbeddings, EncoderBlock, CQAttention, AnswerPointer, masked_softmax, NoisyLinear, Transformer


logger = logging.getLogger(__name__)


class DQN(torch.nn.Module):
    model_name = 'dqn'

    def __init__(self, config, word_vocab, char_vocab, device, answer_type="pointing", generate_length=3):
        super(DQN, self).__init__()
        self.config = config
        self.word_vocab = word_vocab
        self.word_vocab_size = len(word_vocab)
        self.char_vocab = char_vocab
        self.char_vocab_size = len(char_vocab)
        self.device = device
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
        self.gat_out_features = self.config['gat']['out_features']
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
        params = self.config['gat']
        self.gat = GATContainer(params=params,  device=self.device)
        
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
        self.action_scorer_shared_linear = linear_function(self.block_hidden_dim+self.gat_out_features, self.action_scorer_hidden_dim)

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

        self.answer_pointer = AnswerPointer(block_hidden_dim=self.block_hidden_dim, gat_out_features = self.gat_out_features, noisy_net=self.noisy_net)

        if self.answer_type in ["2 way"]:
            self.question_answerer_output_1 = linear_function(self.block_hidden_dim, self.question_answerer_hidden_dim)
            self.question_answerer_output_2 = linear_function(self.question_answerer_hidden_dim, 2)

    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask):
        # node encoding: batch x num_node x hid
        # node mask: batch x num_node

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

    def answer_question(self, matching_representation_sequence, doc_mask, gat_out):
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
        pred = self.answer_pointer(M1, M2, doc_mask,gat_out)  # batch x time
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

class GAT(torch.nn.Module):
    """
    The actual Graph Attention Network class.
    """
    def __init__(self, num_features, num_hidden, num_class, num_heads, dropout, alpha, device):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.device = device
        self.attentions = GATConv(in_channels=num_features, out_channels=num_hidden, heads=num_heads, concat=True, negative_slope=alpha, dropout=dropout).to(self.device) #The Graph Attention Layer
        
    def forward(self, graph_rep):
        """
        Forward call for the GAT. Inputs the graph state representation into the Graph Attention Layer.
        :param graph_rep:  Data object containing entities' vector representation and a list of edge indices.
        :return x: The transformed set of output features of each node in the entire graph.
        """
        x = graph_rep.x
        adj = graph_rep.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attentions(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class GATContainer(torch.nn.Module):
    """
    The GATContainer class contains both the Graph Attention Network and Transformer Encoder and utilises them to produce a fixed length graph representation at each time step.
    """
    def __init__(self, params, device):
        super(GATContainer,self).__init__()
        self.params = params
        self.device = device
        self.bert_size = params['bert_size']
        
        if self.bert_size == 'tiny':
            features = 128
        elif self.bert_size == 'mini':
            features = 256
        elif self.bert_size == 'base':
            features = 768
        else:
            features = 512 #Small or medium

        #Initialise GAT and transformer encoder
        self.gat = GAT(num_features=features, num_hidden=params['num_hidden'], num_class=params['out_features'], num_heads=params['num_heads'], dropout=params['dropout'], alpha=params['alpha'], device=self.device)
        self.transformer = Transformer(hidden_size=params['out_features'], num_layers=params['transformer']['num_layers'], transformer_heads = params['transformer']['transformer_heads'], dropout=params['transformer']['dropout'], device=self.device)

    def forward(self, graph_rep, batch_num_nodes):
        """
        Forward call for the GAT container. Inputs the graph state representation into the GAT and summarises the outputs using the transformer.
        :param graph_rep: Data object containing entities' vector representation and a list of edge indices.
        :param batch_num_nodes: List containing the number of entities in each graph in the batch.
        :return out: The final output of the transformer, which is the summarised GAT output.
        """
        x = self.gat(graph_rep)

        #Break batch up into its appropriate graphs
        count = 0
        graph_list = []
        for num in batch_num_nodes:
            graph = x[count:count+num,:]
            graph_list.append(graph)
            count += num

        batch, masks = self.transformer.pad(vecs=graph_list, device=self.device) #pad items in the batch to be of equal lengh for transformer
        out = self.transformer.encoder(src=batch, src_key_padding_mask=masks).permute(1,0,2)
        out = out[:,-1,:]

        return out
    
        