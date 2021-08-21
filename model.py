import logging
import os
import numpy as np
import torch.optim
import torch
import torch.nn.functional as F

from layers import Embedding, MergeEmbeddings, EncoderBlock, CQAttention, AnswerPointer, masked_softmax, NoisyLinear

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

        self.dqn_conditioning = self.config['dqn_conditioning']
        self.icm = self.config['icm']['enable']

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

        self.merge_embeddings = MergeEmbeddings(block_hidden_dim=self.block_hidden_dim, word_emb_dim=self.word_embedding_size,
                                                char_emb_dim=self.char_embedding_size, dropout=self.embedding_dropout)

        self.encoders = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=7,
                                                          block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        self.context_question_attention = CQAttention(
            block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)

        self.context_question_attention_resizer = torch.nn.Linear(
            self.block_hidden_dim * 4, self.block_hidden_dim)

        self.aggregators = torch.nn.ModuleList([EncoderBlock(conv_num=self.aggregation_conv_num, ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim,
                                                             n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.aggregation_layers)])

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear
        self.action_scorer_shared_linear = linear_function(
            self.block_hidden_dim, self.action_scorer_hidden_dim)

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
        for i in range(self.generate_length):
            if not self.dqn_conditioning:
                action_scorers.append(linear_function(
                    self.action_scorer_hidden_dim, action_scorer_output_size))
            else:
                action_scorers.append(linear_function(
                    self.action_scorer_hidden_dim+i*action_scorer_output_size, action_scorer_output_size))
        self.action_scorers = torch.nn.ModuleList(action_scorers)

        if self.dueling_networks:
            action_scorers_advantage = []
            for _ in range(self.generate_length):
                action_scorers_advantage.append(linear_function(
                    self.action_scorer_hidden_dim, action_scorer_advantage_output_size))
            self.action_scorers_advantage = torch.nn.ModuleList(
                action_scorers_advantage)

        self.answer_pointer = AnswerPointer(
            block_hidden_dim=self.block_hidden_dim, noisy_net=self.noisy_net)

        if self.answer_type in ["2 way"]:
            self.question_answerer_output_1 = linear_function(
                self.block_hidden_dim, self.question_answerer_hidden_dim)
            self.question_answerer_output_2 = linear_function(
                self.question_answerer_hidden_dim, 2)

        if self.icm:
            self.curiosity_module = ICM(self.config,self.block_hidden_dim,3*self.word_embedding_size,self.word_vocab_size)

    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask):
        # node encoding: batch x num_node x hid
        # node mask: batch x num_node
        X = self.context_question_attention(
            doc_encodings, q_encodings, doc_mask, q_mask)
        M0 = self.context_question_attention_resizer(X)
        M0 = F.dropout(M0, p=self.block_dropout, training=self.training)
        square_mask = torch.bmm(doc_mask.unsqueeze(-1),
                                doc_mask.unsqueeze(1))  # batch x time x time
        for i in range(self.aggregation_layers):
            M0 = self.aggregators[i](
                M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        return M0

    def representation_generator(self, _input_words, _input_chars):
        """
        Encode words and chars into single representation using the embedding and encoding layers.
        """
        embeddings, mask = self.word_embedding(
            _input_words)  # batch x time x emb
        char_embeddings, _ = self.char_embedding(
            _input_chars)  # batch x time x nchar x emb
        merged_embeddings = self.merge_embeddings(
            embeddings, char_embeddings, mask)  # batch x time x emb
        # batch x time x time
        square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoders[i](merged_embeddings, mask, square_mask, i * (
                self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc

        return encoding_sequence, mask

    def action_scorer(self, state_representation_sequence, word_masks):
        
        state_representation, _ = torch.max(state_representation_sequence, 1)
        
        hidden = self.action_scorer_shared_linear(
            state_representation)  # batch x hid
        
        hidden = torch.relu(hidden)  # batch x hid
        action_ranks = []
        if self.dqn_conditioning:
            a_rank = self.action_scorers[0](hidden)  # batch x n_vocab
            action_ranks.append(a_rank)
            prev = a_rank

            for i in range(1, self.generate_length):
                a_rank = self.action_scorers[i](
                    torch.cat((hidden, prev), dim=1))  # batch x n_vocab
                action_ranks.append(a_rank)
                prev = torch.cat((prev, a_rank), dim=1)

        else:
            for i in range(self.generate_length):
                # batch x n_vocab, or batch x n_vocab*atoms
                a_rank = self.action_scorers[i](hidden)
                if self.use_distributional:
                    if self.dueling_networks:
                        a_rank_advantage = self.action_scorers_advantage[i](
                            hidden)  # advantage stream
                        a_rank = a_rank.view(-1, 1, self.atoms)
                        a_rank_advantage = a_rank_advantage.view(
                            -1, self.word_vocab_size, self.atoms)
                        a_rank_advantage = a_rank_advantage * \
                            word_masks[i].unsqueeze(-1)
                        q = a_rank + a_rank_advantage - \
                            a_rank_advantage.mean(
                                1, keepdim=True)  # combine streams
                    else:
                        # batch x n_vocab x atoms
                        q = a_rank.view(-1, self.word_vocab_size, self.atoms)
                    # batch x n_vocab x atoms
                    q = masked_softmax(q, word_masks[i].unsqueeze(-1), axis=-1)
                else:
                    if self.dueling_networks:
                        a_rank_advantage = self.action_scorers_advantage[i](
                            hidden)  # advantage stream, batch x vocab
                        a_rank_advantage = a_rank_advantage * word_masks[i]
                        # combine streams  # batch x vocab
                        q = a_rank + a_rank_advantage - \
                            a_rank_advantage.mean(1, keepdim=True)
                    else:
                        q = a_rank  # batch x vocab
                    q = q * word_masks[i]
                action_ranks.append(q)
        return action_ranks

    def answer_question(self, matching_representation_sequence, doc_mask):
        """
        Answer question based on representation
        :return prediction distribution.
        """
        square_mask = torch.bmm(doc_mask.unsqueeze(-1),
                                doc_mask.unsqueeze(1))  # batch x time x time
        M0 = matching_representation_sequence
        M1 = M0
        for i in range(self.aggregation_layers):
            M0 = self.aggregators[i](
                M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        M2 = M0
        pred = self.answer_pointer(M1, M2, doc_mask)  # batch x time
        # pred_distribution: batch x time
        pred_distribution = masked_softmax(pred, m=doc_mask, axis=-1)  #
        if self.answer_type == "pointing":
            return pred_distribution

        z = torch.bmm(pred_distribution.view(pred_distribution.size(
            0), 1, pred_distribution.size(1)), M2)  # batch x 1 x inp
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


###################################################################


class ActorCritic(torch.nn.Module):
    model_name = 'ActorCritic'

    def __init__(self, config, word_vocab, char_vocab, answer_type="pointing", generate_length=3):
        super(ActorCritic, self).__init__()
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
        self.n_heads = model_config['n_heads']
        self.block_dropout = model_config['block_dropout']
        self.attention_dropout = model_config['attention_dropout']
        self.action_scorer_hidden_dim = model_config['action_scorer_hidden_dim']
        self.question_answerer_hidden_dim = model_config['question_answerer_hidden_dim']
        self.noisy_net = self.config['epsilon_greedy']['noisy_net']
        self.icm = self.config['icm']['enable']

    def _def_layers(self):
        """
        Create the layers of the Actor-Critic networks
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

        self.merge_embeddings = MergeEmbeddings(block_hidden_dim=self.block_hidden_dim, word_emb_dim=self.word_embedding_size,
                                                char_emb_dim=self.char_embedding_size, dropout=self.embedding_dropout)

        self.encoders = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=7,
                                                          block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        self.context_question_attention = CQAttention(
            block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)

        self.context_question_attention_resizer = torch.nn.Linear(
            self.block_hidden_dim * 4, self.block_hidden_dim)

        self.aggregators = torch.nn.ModuleList([EncoderBlock(conv_num=self.aggregation_conv_num, ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim,
                                                             n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.aggregation_layers)])

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear
        self.action_scorer_shared_linear = linear_function(
            self.block_hidden_dim, self.action_scorer_hidden_dim)

        action_scorer_output_size = self.word_vocab_size

        action_scorers = []

        for i in range(self.generate_length):
            action_scorers.append(linear_function(
                self.action_scorer_hidden_dim+i*action_scorer_output_size, action_scorer_output_size))

        self.action_scorers = torch.nn.ModuleList(action_scorers)
        self.critic = linear_function(self.action_scorer_hidden_dim, 1)

        self.answer_pointer = AnswerPointer(
            block_hidden_dim=self.block_hidden_dim, noisy_net=self.noisy_net)

        if self.answer_type in ["2 way"]:
            self.question_answerer_output_1 = linear_function(
                self.block_hidden_dim, self.question_answerer_hidden_dim)
            self.question_answerer_output_2 = linear_function(
                self.question_answerer_hidden_dim, 2)

        if self.icm:
            self.curiosity_module = ICM(self.config,self.block_hidden_dim,3*self.word_embedding_size,self.word_vocab_size)
            

    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask):
        # node encoding: batch x num_node x hid
        # node mask: batch x num_node
        X = self.context_question_attention(
            doc_encodings, q_encodings, doc_mask, q_mask)
        M0 = self.context_question_attention_resizer(X)
        M0 = F.dropout(M0, p=self.block_dropout, training=self.training)
        square_mask = torch.bmm(doc_mask.unsqueeze(-1),
                                doc_mask.unsqueeze(1))  # batch x time x time
        for i in range(self.aggregation_layers):
            M0 = self.aggregators[i](
                M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        return M0

    def representation_generator(self, _input_words, _input_chars):
        embeddings, mask = self.word_embedding(
            _input_words)  # batch x time x emb
        char_embeddings, _ = self.char_embedding(
            _input_chars)  # batch x time x nchar x emb
        merged_embeddings = self.merge_embeddings(
            embeddings, char_embeddings, mask)  # batch x time x emb
        # batch x time x time
        square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoders[i](merged_embeddings, mask, square_mask, i * (
                self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc

        return encoding_sequence, mask

    def action_scorer(self, state_representation_sequence, word_masks):
        """

        :return action_probs: three probability distributions for each word in the action triplet.
        :return value: the critics state value.
        """

        state_representation, _ = torch.max(state_representation_sequence, 1)
        hidden = self.action_scorer_shared_linear(
            state_representation)  # batch x hid
        
        hidden = torch.relu(hidden)  # batch x hid

        action_probs = []
        a_rank = self.action_scorers[0](hidden)  # batch x n_vocab

        q = masked_softmax(a_rank, word_masks[0])  # batch x vocab

        action_probs.append(q)
        prev = q

        for i in range(1, self.generate_length):
            a_rank = self.action_scorers[i](
                torch.cat((hidden, prev), dim=1))  # batch x n_vocab

            q = masked_softmax(a_rank, word_masks[i])  # batch x vocab

            action_probs.append(q)
            prev = torch.cat((prev, q), dim=1)

        state_value = self.critic(hidden)

        return action_probs, state_value

    def answer_question(self, matching_representation_sequence, doc_mask):
        """
        Answer question based on representation
        :return prediction distribution.
        """
        square_mask = torch.bmm(doc_mask.unsqueeze(-1),
                                doc_mask.unsqueeze(1))  # batch x time x time
        M0 = matching_representation_sequence
        M1 = M0
        for i in range(self.aggregation_layers):
            M0 = self.aggregators[i](
                M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        M2 = M0
        pred = self.answer_pointer(M1, M2, doc_mask)  # batch x time
        # pred_distribution: batch x time
        pred_distribution = masked_softmax(pred, m=doc_mask, axis=-1)  #
        if self.answer_type == "pointing":
            return pred_distribution

        z = torch.bmm(pred_distribution.view(pred_distribution.size(
            0), 1, pred_distribution.size(1)), M2)  # batch x 1 x inp
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


class ICM_Inverse(torch.nn.Module):
    """
    ICM - Inverse Model - used to predict action from two consecutive states. This enables the feature model to learn a valuable feature representation.
    """

    def __init__(self, input_size, hidden_size, output_size,vocab_size):
        super(ICM_Inverse, self).__init__()
        
        self.action_decoder = torch.nn.Sequential(
            torch.nn.Linear(input_size,hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,vocab_size),
            torch.nn.Softmax(dim=-1))
        self.modifier_decoder = torch.nn.Sequential(
            torch.nn.Linear(input_size+vocab_size,hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,vocab_size),
            torch.nn.Softmax(dim=-1))
        self.object_decoder = torch.nn.Sequential(
            torch.nn.Linear(input_size+2*vocab_size,hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size,vocab_size),
            torch.nn.Softmax(dim=-1))

    def forward(self, state_feature, next_state_feature):
        input = torch.cat([state_feature, next_state_feature], dim=-1)
        # rep = self.inverse_net(input)
        rep = input
        action_dist = self.action_decoder(rep)
        modifier_dist = self.modifier_decoder(torch.cat([rep,action_dist],dim=-1))
        object_dist = self.object_decoder(torch.cat([rep,action_dist,modifier_dist],dim=-1))
        return action_dist,modifier_dist,object_dist


class ICM_Forward(torch.nn.Module):

    def __init__(self, input_size, hidden_size, feature_size):
        super(ICM_Forward, self).__init__()
        self.forward_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, feature_size)
        )

    def forward(self, state_feature, action_embedding):
        
        action_embedding = action_embedding.reshape(len(action_embedding),3*len(action_embedding[0][0]))
        input = torch.cat([state_feature, action_embedding], dim=-1)
        
        return self.forward_net(input)


class ICM_Feature(torch.nn.Module):

    def __init__(self, state_embedding_size, hidden_size, feature_size):
        super(ICM_Feature, self).__init__()
        
        self.feature_net = torch.nn.Sequential(
            torch.nn.Linear(state_embedding_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, feature_size)
        )

    def forward(self, input):
        state, _ = torch.max(input, 1)
        
        return self.feature_net(state)


class ICM(torch.nn.Module):
    """
    Intrinsic Curiosity Module - used to instill curiosity into agent.
    """

    def __init__(self, config, state_input_size, action_size,vocab_size):
        super(ICM, self).__init__()
        self.config = config
        self.read_config()
        
        if not self.use_feature_net:
            feature_size = state_input_size
        else:
            feature_size = self.feature_size
            self.feature_model = ICM_Feature(
            state_input_size, self.hidden_size, self.feature_size)

        self.inverse_model = ICM_Inverse(
            feature_size*2, self.hidden_size, action_size,vocab_size)
        self.forward_model = ICM_Forward(
            feature_size+action_size, self.hidden_size, feature_size)
        
    
    def read_config(self):
        """
        Read config file to set ICM hyperparameters
        """
        self.scaling_factor = self.config['icm']['scaling_factor']
        self.beta = self.config['icm']['beta']
        self.lambda_weight = self.config['icm']['lambda']
        self.hidden_size = self.config['icm']['hidden_size']
        self.feature_size = self.config['icm']['state_feature_size']
        self.use_inverse_model = self.config['icm']['inverse_reward']
        self.use_feature_net = self.config['icm']['use_feature_net']

    def get_feature(self, state):
        """
        Use the feature model to get learned feature representation of the state.
        :param state: the state to convert into a feature representation.
        :return : state feature
        """
        return self.feature_model(state)

    def get_predicted_action(self, state, next_state):
        """
        Use the inverse model to get the predicted action.
        :param state: the current state.
        :param next_state: the next state.
        :return : vocab distributions for action, modifier, object
        """
        # Using Feature Net
        if self.use_feature_net:
            state_feature = self.get_feature(state)
            next_state_feature = self.get_feature(next_state)
        else:
            # Using max pooling of transformer layers
            state_feature, _ = torch.max(state, 1)
            next_state_feature,_ = torch.max(next_state, 1)

        return self.inverse_model(state_feature, next_state_feature)

    def get_predicted_state(self, state, action):
        """

        TODO: depending on action representation maybe dont detach

        Use the forward model to predict the next state's feature representation.
        :param state: the current state.
        :param action: the action performed.
        :return : the feature representation of the predicted next state.
        """
        # Using FeatureNet
        if self.use_feature_net:
            state_feature = self.get_feature(state)
        else:
            # Using max pooling of transformer layers
            state_feature, _ = torch.max(state, 1)

        return self.forward_model(state_feature, action)

    def get_inverse_loss(self, state, action, next_state):
        """
        # TODO Investigate the difference between mult of probs and addition of probs
        Get the loss of the inverse model.
        :param state: the current state.
        :param action: the action performed.
        :param next_state: the next state after action.
        :return : Inverse models loss

        """
        predicted_action, predicted_modifier, predicted_object = self.get_predicted_action(state, next_state)
        
        action_probs = torch.gather(predicted_action,-1,action)[:,0]
        modifier_probs = torch.gather(predicted_modifier,-1,action)[:,1]
        object_probs = torch.gather(predicted_object,-1,action)[:,2]
        loss = -torch.log(action_probs+modifier_probs+object_probs)
        return loss

    def get_forward_loss(self, state, action, next_state):
        """
        Get the loss of the forward model. This is the difference between the actual next state feature and the predicted next state feature.
        :param state: the current state.
        :param action: the action performed.
        :param next_state: the next state after action.
        :return : MSE loss of the forward model.
        """
        next_state_feature = self.get_feature(next_state)
        predicted_state_feature = self.get_predicted_state(state, action)
        
        return F.mse_loss(predicted_state_feature, next_state_feature,reduction='none').mean(dim=-1)

    def get_intrinsic_reward(self, state, action, next_state):
        """
        Calculate the intrinsic reward.
        :param state: the current state.
        :param action: the action performed.
        :param next_state: the next state after action.
        :return : the intrinsic reward.
        """
        with torch.no_grad():
            if not self.use_inverse_model:
                intrinsic_reward = self.scaling_factor * \
                    self.get_forward_loss(state, action, next_state).detach()
            else:
                intrinsic_reward = self.scaling_factor * \
                    self.get_inverse_loss(state, action, next_state).detach()
            
            return intrinsic_reward


    def forward(self,state,action,next_state):
        forward_loss = self.get_forward_loss(state,action,next_state)
        inverse_loss = self.get_inverse_loss(state,action,next_state)

        return forward_loss,inverse_loss


    