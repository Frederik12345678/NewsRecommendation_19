

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class AttLayer2_torch(nn.Module):
    """Soft alignment attention implementation in PyTorch."""
    
    def __init__(self, dim=200, seed=0):
        super(AttLayer2_torch, self).__init__()
        self.dim = dim
        torch.manual_seed(seed)

        # Initialize W, b, and q but do not specify input dimension yet
        self.W = None
        self.b = nn.Parameter(torch.zeros(dim))
        self.q = nn.Parameter(torch.empty(dim, 1))
        
        # Initialize q using Xavier initialization
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs):
        # Dynamically initialize W based on the input's feature size
        if self.W is None:
            input_dim = inputs.size(-1)  # Get the feature dimension of the input
            self.W = nn.Parameter(torch.empty(input_dim, self.dim))
            nn.init.xavier_uniform_(self.W)  # Xavier initialization for W
        
        # Apply soft attention mechanism
        #print("input", inputs.shape)
        attention = torch.tanh(inputs @ self.W + self.b)
        attention = attention @ self.q
        #print("attention", attention.shape)
        attention = torch.squeeze(attention, dim=-1)
        #print("attention2", attention.shape)
        
        attention_weights = F.softmax(attention, dim=-1)
        attention_weights = attention_weights.unsqueeze(-1)
        #print("attention_weights", attention_weights.shape)

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

        # Initially set input_dim as None
        self.input_dim = None

        # Placeholder for the Linear layers for Q, K, V
        self.WQ = None
        self.WK = None
        self.WV = None

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
        # Print input shapes before passing them to attention
        
        # Set input_dim dynamically based on the input shape
        if self.input_dim is None:
            self.input_dim = Q_seq.size(-1)  # Set input_dim from the last dimension of Q_seq

            # Initialize the Linear layers with the correct input dimension
            self.WQ = nn.Linear(self.input_dim, self.output_dim)
            self.WK = nn.Linear(self.input_dim, self.output_dim)
            self.WV = nn.Linear(self.input_dim, self.output_dim)

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
        #print("atention shape: ", O.shape)
        # Apply the mask (if applicable)
        return self._mask(O, Q_len, "mul") if Q_len is not None else O





class NRMSModelPytorch(nn.Module):
    def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=300, vocab_size=32000, seed=None, time_inclusion = "false"):
        super(NRMSModelPytorch, self).__init__()
        """
        Description: 
        This class implements the NRMS Model

        _______Input to initialize model_______ 
        hparams: parameter class which holds values for loss function, amount of dropout, which optimizer to use and learing rate
        word2vec_embedding: A predefined word to vector embedding. 
        time_inclusion: A string indicating, which time embedding method we use or if we include time at all in our model
            - Can take values ["false", "add", "ffnn"]
        

        NOTE: 
        activation: Since we test our model on both Softmax and Sigmoid this is a varible changing the output of the model. 
                    if softmax is chosen the model spits out logits and cross_entropy_loss function has to be chosen since it does 
                    on the logit output. If sigmoid is chosen the model takes the logits and preforms a sigmoid on it. 
                    Binary cross entropy is then needed for loss function. 


        ______ Input to forward method__________
        his_input_title: shape = [B,hs,mtl]
        pred_input_title: shape =[B,ps,mtl]

        b = batch size
        hs = History size. Amount of articles we use to "learn" a representation for the individual user
        ps = prediction size. Amount of articles we use for prediction. (always 5 in our case)
        mtl = Max Title Length. The length of title for each article. If the article has less words than this value,
        we include words from the subtitle until the max title length has been reached. 
        (We do this because we need a constant input size and titles of articles have different lengths)

        _____Output_______
        Outputs [B,5] tensor, where B is batch size and 5 is the prediction for the 5 prediction articles we give the model. 
        """

        #Initialize parameters to self
        self.hparams = hparams
        self.seed = seed
        self.time_inclusion = time_inclusion   

        if self.time_inclusion == "ffnn":
            self.fc = nn.Sequential(
                nn.Linear(800, 512),  # Hidden layer with 512 units
                nn.ReLU(),           # Activation function
                nn.Linear(512, 400)  # Output layer
            )


        #Initialize activation function for final output based on loss function
        if hparams.loss == "cross_entropy_loss":
            print("Using Softmax")
            self.activation = "softmax"
        elif hparams.loss == "BCE_loss":
            print("Using Sigmoid")
            self.activation = "sigmoid"
        else: 
            self.activation = "softmax"


        #Set seed 
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
        #Notice we use the newsencoder to initialize the userencoder
        self.user_encoder = self._build_userencoder(self.news_encoder)

        # Define optimizer and loss
        self.criterion = self._get_loss(hparams.loss)
        self.criterion_val = nn.CrossEntropyLoss()
        self.optimizer = self._get_opt(hparams.optimizer, hparams.learning_rate)

    def _get_loss(self, loss):
        #Returns the loss function

        if loss == "cross_entropy_loss":
            return nn.CrossEntropyLoss()
        elif loss == "BCE_loss":
            return nn.BCELoss()
        else:
            raise ValueError(f"this loss not defined {loss}")

    def _get_opt(self, optimizer, lr):
        # Returns adam optimizer 

        if optimizer == "adam":
            return optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")

    def _build_userencoder(self, titleencoder):
        """
        This class builds the userencoder part of our model.

        Works by taking an input, his_input_title, which contains articles which we use to learn a representation of a user. 
        For each individual we take all articles and send them individually through the newsencoder. 
        This then outputs a news representation for each article. All these news representations are then send through 2 different attention
        layers to learn a user representation. This user representation is then the output of the userencoder.
        """


        class UserEncoder(nn.Module):
            def __init__(self, hparams, titleencoder):
                super(UserEncoder, self).__init__()
                
                #Initializes the 2 attentionlayers and newsencoder
                self.titleencoder = titleencoder
                self.attention = SelfAttention_torch(hparams.head_num, hparams.head_num)
                self.att_layer = AttLayer2_torch(hparams.attention_hidden_dim)

            def forward(self, his_input_title,time_embedding_hist=None,time_inclusion=None,fc=None):
                
                # titleencoder = newsencoder 

                #Sends each article through the newsencoder 
                click_title_presents = torch.stack([self.titleencoder(title) for title in his_input_title], dim=0)

                #_______________Time Inclusion:_______________________________________
                if time_inclusion == "add":
                    #Adds the time embedding and prediction news embedding together
                    click_title_presents = time_embedding_hist + click_title_presents
                elif time_inclusion == "ffnn":
                    #stacks the time embedding and prediction news embedding ontop of eachother
                    #then sends through a small feed forward network, that reduced embedding dim back to 400
                    x = torch.cat((click_title_presents,  time_embedding_hist), dim=2)
                    click_title_presents = fc(x)
                else: 
                    pass #do nothing

                #______________________________________________________________________
                #sends news representations though first attention layer
                y = self.attention(click_title_presents,click_title_presents,click_title_presents)
                
                #sends output of first attention layer through second attention layer. 
                user_present = self.att_layer(y)

                #returns user representation. shape [B, 400], B=batch size
                return user_present

        return UserEncoder(self.hparams, titleencoder)

    def _build_newsencoder(self):
        """
        This class builds the newsencoder part of our model.

        works by first sending the input through an embedding layer, getting an embedding representation for each word in the title of a given article.
        Then we send the embeddings though 2 attention layers, with some dropout between
        """
        class NewsEncoder(nn.Module):
            def __init__(self, embedding_layer, hparams, seed):
                super(NewsEncoder, self).__init__()

                #initializes embedding layer and self attention layers
                self.embedding = embedding_layer
                self.dropout1 = nn.Dropout(hparams.dropout)  
                self.attention = SelfAttention_torch(hparams.head_num, hparams.head_dim, seed=seed)
                self.dropout2 = nn.Dropout(hparams.dropout)  
                self.att_layer = AttLayer2_torch(hparams.attention_hidden_dim, seed=seed)


            def forward(self, sequences_input_title):
                #The forward method only takes a single news article at a time. 
            
                #this is a single news article
                sequences_input_title = sequences_input_title.long()

                #gets embedding of news article
                embedded_sequences_title = self.embedding(sequences_input_title)
                
                # Seds embedding though dropout 1
                y = self.dropout1(embedded_sequences_title)

                # attention layer 1 

                y = self.attention(y,y,y)

                #dropout 2
                y = self.dropout2(y)

                # Attentionlayer 2
                pred_title = self.att_layer(y)
                
                #returns the news representation. shape=[B,400]
                return pred_title

        return NewsEncoder(self.embedding_layer, self.hparams, self.seed)


    def forward(self, his_input_title, pred_input_title, time_embedding_hist = None, time_embedding_pred = None):
        """
        This forward method is the main working horse of our model. It takes the defined userencoder and newsencoder and uses them on the given input. 

        Output is dependend on activation parameter
        """
        
        #Sends history articles through the userencoder 
        if self.time_inclusion == "add":
            user_present = self.user_encoder(his_input_title,time_embedding_hist,self.time_inclusion)
        elif self.time_inclusion == "ffnn":
            user_present = self.user_encoder(his_input_title,time_embedding_hist,self.time_inclusion,self.fc)
        else:
            user_present = self.user_encoder(his_input_title)
        
        #Sends the prediction articles though the newsencoder
        news_present = torch.stack([self.news_encoder(news) for news in pred_input_title], dim=0)

        #_______________Time Inclusion:_______________________________________
        if self.time_inclusion == "add":
            #Adds the time embedding and prediction news embedding together
            news_present = time_embedding_pred + news_present
        elif self.time_inclusion == "ffnn":
            #stacks the time embedding and prediction news embedding ontop of eachother
            #then sends through a small feed forward network, that reduced embedding dim back to 400
            x = torch.cat((news_present,  time_embedding_pred), dim=2)
            news_present = self.fc(x)
        else: 
            pass #Do nothing
        #______________________________________________________________________

        #gets the dot product between the user representation and news representation of the prediction articles
        preds = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
       
        #Applies relevant activation function and returns 
        if self.activation == "softmax" or self.activation == "Softmax":
            return preds
        elif self.activation == "sigmoid" or self.activation == "Sigmoid":
            return preds # torch.sigmoid(preds)
        else:
            print("Not valid activation function: - Using Softmax")
            return preds

    





class NRMSModel_addition(nn.Module):
    def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=300, vocab_size=32000, seed=None, activation = "softmax"):
        super(NRMSModelPytorch, self).__init__()
        self.hparams = hparams
        self.seed = seed
        self.activation = activation
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

        self.fc = nn.Sequential(
            nn.Linear(403, 512),  # Hidden layer with 512 units
            nn.ReLU(),           # Activation function
            nn.Linear(512, 400)  # Output layer
        )

    def _get_loss(self, loss):
        if loss == "cross_entropy_loss":
            return nn.CrossEntropyLoss()
        elif loss == "log_loss":
            print("BCE is used")
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
                #print("input:", his_input_title.shape)
                # Encode each news in the history
                click_title_presents = torch.stack([self.titleencoder(title) for title in his_input_title], dim=0)
                #print("Vlick",click_title_presents.shape)
                y = self.attention(click_title_presents,click_title_presents,click_title_presents)
                #print("y.shape",y.shape)
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
                #print("sequence",sequences_input_title.shape)
                sequences_input_title = sequences_input_title.long()
                embedded_sequences_title = self.embedding(sequences_input_title)
                #print("embedded",embedded_sequences_title.shape)
                y = self.dropout1(embedded_sequences_title)

                y = self.attention(y,y,y)
                y = self.dropout2(y)
                pred_title = self.att_layer(y)
                #print("pred",pred_title.shape)
                return pred_title

        return NewsEncoder(self.embedding_layer, self.hparams, self.seed)

    def forward(self, his_input_title, pred_input_title, Time_embedding):
        #print(his_input_title.shape)
       # print(pred_input_title.shape)
        user_present = self.user_encoder(his_input_title)
        #print("user: ", user_present.shape)
        news_present = torch.stack([self.news_encoder(news) for news in pred_input_title], dim=0)
        #print("news: ", news_present.shape)
        
        #_________ Time Embedding Part Started_________________-
        # feed forwards 
        news_present = news_present + Time_embedding
        #_________ Time Embedding Part Ended_________________

        
        #print("new user: ", user_present.unsqueeze(-1).shape)
        preds = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
        #print("npreds: ", preds.shape)


        if self.activation == "softmax":
            return torch.softmax(preds, dim=-1)
        elif self.activation == "sigmoid":
            return torch.sigmoid(preds)
        else:
            
            return preds

    def predict(self, his_input_title, pred_input_title_one):
        
        user_present = self.user_encoder(his_input_title)
        news_present_one = self.news_encoder(pred_input_title_one)
        pred_one = torch.sigmoid(torch.dot(news_present_one, user_present))
        return pred_one




class NRMSModel_FFNN(nn.Module):
    def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=300, vocab_size=32000, seed=None, activation = "softmax"):
        super(NRMSModelPytorch, self).__init__()
        self.hparams = hparams
        self.seed = seed
        self.activation = activation
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

        self.fc = nn.Sequential(
            nn.Linear(403, 512),  # Hidden layer with 512 units
            nn.ReLU(),           # Activation function
            nn.Linear(512, 400)  # Output layer
        )

    def _get_loss(self, loss):
        if loss == "cross_entropy_loss":
            return nn.CrossEntropyLoss()
        elif loss == "log_loss":
            print("BCE is used")
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
                #print("input:", his_input_title.shape)
                # Encode each news in the history
                click_title_presents = torch.stack([self.titleencoder(title) for title in his_input_title], dim=0)
                #print("Vlick",click_title_presents.shape)
                y = self.attention(click_title_presents,click_title_presents,click_title_presents)
                #print("y.shape",y.shape)
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
                #print("sequence",sequences_input_title.shape)
                sequences_input_title = sequences_input_title.long()
                embedded_sequences_title = self.embedding(sequences_input_title)
                #print("embedded",embedded_sequences_title.shape)
                y = self.dropout1(embedded_sequences_title)

                y = self.attention(y,y,y)
                y = self.dropout2(y)
                pred_title = self.att_layer(y)
                #print("pred",pred_title.shape)
                return pred_title

        return NewsEncoder(self.embedding_layer, self.hparams, self.seed)

    def forward(self, his_input_title, pred_input_title, Time_embedding):
        #print(his_input_title.shape)
       # print(pred_input_title.shape)
        user_present = self.user_encoder(his_input_title)
        #print("user: ", user_present.shape)
        news_present = torch.stack([self.news_encoder(news) for news in pred_input_title], dim=0)
        #print("news: ", news_present.shape)
        
        #_________ Time Embedding Part Started_________________-
        # feed forwards 
        x = torch.cat((news_present,  Time_embedding), dim=1)
        
        news_present = self.fc(x)
        #_________ Time Embedding Part Ended_________________

        
        #print("new user: ", user_present.unsqueeze(-1).shape)
        preds = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
        #print("npreds: ", preds.shape)


        if self.activation == "softmax":
            return torch.softmax(preds, dim=-1)
        elif self.activation == "sigmoid":
            return torch.sigmoid(preds)
        else:
            
            return preds

    def predict(self, his_input_title, pred_input_title_one):
        
        user_present = self.user_encoder(his_input_title)
        news_present_one = self.news_encoder(pred_input_title_one)
        pred_one = torch.sigmoid(torch.dot(news_present_one, user_present))
        return pred_one




