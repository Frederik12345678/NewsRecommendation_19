

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NRMSModelPytorch(nn.Module):
    def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=300, vocab_size=32000, seed=None):
        super(NRMSModelPytorch, self).__init__()
        self.hparams = hparams
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize word embeddings
        if word2vec_embedding is None:
            self.word2vec_embedding = torch.randn(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = torch.from_numpy(word2vec_embedding).float()
        self.embedding_layer = nn.Embedding.from_pretrained(self.word2vec_embedding, freeze=False)

        # Build model components
        self.news_encoder = self._build_newsencoder()
        self.user_encoder = self._build_userencoder(self.news_encoder)

        # Define optimizer and loss
        self.criterion = self._get_loss(hparams.loss)
        self.optimizer = self._get_opt(hparams.optimizer, hparams.learning_rate)

    def _get_loss(self, loss):
        if loss == "cross_entropy_loss":
            return nn.CrossEntropyLoss()
        elif loss == "log_loss":
            return nn.BCELoss()
        else:
            raise ValueError(f"this loss not defined {loss}")

    def _get_opt(self, optimizer, lr):
        if optimizer == "adam":
            return optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")

    def _build_userencoder(self, titleencoder):
        # Define user encoder using SelfAttention and AttLayer2 modules
        class UserEncoder(nn.Module):
            def __init__(self, hparams, titleencoder):
                super(UserEncoder, self).__init__()
                self.titleencoder = titleencoder
                self.attention = SelfAttention_torch(hparams.head_num, hparams.head_num)
                self.att_layer = AttLayer2_torch(hparams.attention_hidden_dim)

            def forward(self, his_input_title):
                # Encode each news in the history
                click_title_presents = torch.stack([self.titleencoder(title) for title in his_input_title], dim=1)
                y = self.attention(click_title_presents)
                user_present = self.att_layer(y)
                return user_present

        return UserEncoder(self.hparams, titleencoder)

    def _build_newsencoder(self):
        # Define news encoder using embedding and attention layers
        class NewsEncoder(nn.Module):
            def __init__(self, embedding_layer, hparams, seed):
                super(NewsEncoder, self).__init__()
                self.embedding = embedding_layer
                self.dropout1 = nn.Dropout(hparams.dropout)  # Use attribute access here
                self.attention = SelfAttention_torch(hparams.head_num, hparams.head_dim, seed=seed)
                self.dropout2 = nn.Dropout(hparams.dropout)  # Use attribute access here
                self.att_layer = AttLayer2_torch(hparams.attention_hidden_dim, seed=seed)


            def forward(self, sequences_input_title):
                # Convert input to LongTensor
                sequences_input_title = sequences_input_title.long()
                embedded_sequences_title = self.embedding(sequences_input_title)
                y = self.dropout1(embedded_sequences_title)
                y = self.attention(y)
                y = self.dropout2(y)
                pred_title = self.att_layer(y)
                return pred_title

        return NewsEncoder(self.embedding_layer, self.hparams, self.seed)

    def forward(self, his_input_title, pred_input_title):
        user_present = self.user_encoder(his_input_title)
        news_present = torch.stack([self.news_encoder(news) for news in pred_input_title], dim=1)
        preds = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
        return torch.softmax(preds, dim=-1)

    def predict(self, his_input_title, pred_input_title_one):
        user_present = self.user_encoder(his_input_title)
        news_present_one = self.news_encoder(pred_input_title_one)
        pred_one = torch.sigmoid(torch.dot(news_present_one, user_present))
        return pred_one


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttLayer2_torch(nn.Module):
    """Soft alignment attention implementation in PyTorch."""
    
    def __init__(self, dim=200, seed=0):
        super(AttLayer2_torch, self).__init__()
        self.dim = dim
        torch.manual_seed(seed)

        # Define trainable weights
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.q = nn.Parameter(torch.empty(dim, 1))
        
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs):
        # Apply soft attention mechanism
        attention = torch.tanh(inputs @ self.W + self.b)
        attention = attention @ self.q
        attention = torch.squeeze(attention, dim=-1)
        
        attention_weights = F.softmax(attention, dim=-1)
        attention_weights = attention_weights.unsqueeze(-1)
        
        weighted_input = inputs * attention_weights
        return torch.sum(weighted_input, dim=1)

class SelfAttention_torch(nn.Module):
    """Multi-head self-attention implementation in PyTorch."""
    
    def __init__(self, multiheads, head_dim, seed=0, mask_right=False):
        super(SelfAttention_torch, self).__init__()
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        torch.manual_seed(seed)

        # Define trainable weights for Q, K, and V transformations
        self.WQ = nn.Linear(self.output_dim, self.output_dim)
        self.WK = nn.Linear(self.output_dim, self.output_dim)
        self.WV = nn.Linear(self.output_dim, self.output_dim)

    def _mask(self, inputs, seq_len, mode="add"):
        """Apply masking operation to inputs based on sequence length."""
        if seq_len is None:
            return inputs
        mask = (torch.arange(inputs.size(1)) < seq_len.unsqueeze(1)).float()
        if mode == "mul":
            return inputs * mask
        elif mode == "add":
            return inputs - (1 - mask) * 1e12

    def forward(self, Q_seq, K_seq, V_seq, Q_len=None, V_len=None):
        # Linear transformations for Q, K, and V
        Q = self.WQ(Q_seq).view(-1, Q_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        K = self.WK(K_seq).view(-1, K_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        V = self.WV(V_seq).view(-1, V_seq.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        A = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if self.mask_right:
            ones = torch.ones_like(A[0, 0])
            mask = torch.tril(ones) * 1e12
            A = A - mask

        # Apply softmax to attention scores
        A = F.softmax(A, dim=-1)

        # Weighted sum of values
        O = (A @ V).permute(0, 2, 1, 3).contiguous().view(-1, Q_seq.size(1), self.output_dim)
        return self._mask(O, Q_len, "mul") if Q_len is not None else O
