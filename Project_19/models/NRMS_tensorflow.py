import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class AttLayer2(layers.Layer):
    """Soft alignment attention implement.

    Attributes:
        dim (int): attention hidden dim
    """

    def __init__(self, dim=200, seed=0, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            dim (int): attention hidden dim
        """

        self.dim = dim
        self.seed = seed
        super(AttLayer2, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initialization for variables in AttLayer2
        There are there variables in AttLayer2, i.e. W, b and q.

        Args:
            input_shape (object): shape of input tensor.
        """

        assert len(input_shape) == 3
        dim = self.dim
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(dim, 1),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )

        #print("W shape:", self.W.shape)
        #print("b shape:", self.b.shape)
        #print("q shape:", self.q.shape)
        super(AttLayer2, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, inputs, mask=None, **kwargs):
        """Core implemention of soft attention

        Args:
            inputs (object): input tensor.

        Returns:
            object: weighted sum of input tensors.
        """

        attention = K.tanh(K.dot(inputs, self.W) + self.b)
        attention = K.dot(attention, self.q)

        attention = K.squeeze(attention, axis=2)

        if mask == None:
            attention = K.exp(attention)
        else:
            attention = K.exp(attention) * K.cast(mask, dtype="float32")

        attention_weight = attention / (
            K.sum(attention, axis=-1, keepdims=True) + K.epsilon()
        )

        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight
        #print("Input shape:", inputs.shape)
        #print("Attention shape (before squeeze):", attention.shape)
        #print("Attention weight shape:", attention_weight.shape)
        #print("Weighted input shape:", weighted_input.shape)

        return K.sum(weighted_input, axis=1)

    def compute_mask(self, input, input_mask=None):
        """Compte output mask value

        Args:
            input (object): input tensor.
            input_mask: input mask

        Returns:
            object: output mask.
        """
        return None

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor

        Args:
            input_shape (tuple): shape of input tensor.

        Returns:
            tuple: shape of output tensor.
        """
        return input_shape[0], input_shape[-1]


class SelfAttention(layers.Layer):
    """Multi-head self attention implement.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimention of each head.
        mask_right (boolean): whether to mask right words.

    Returns:
        object: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimention of each head.
            mask_right (boolean): whether to mask right words.
        """

        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(SelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Returns:
            tuple: output shape tuple.
        """

        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape):
        """Initialization for variables in SelfAttention.
        There are three variables in SelfAttention, i.e. WQ, WK ans WV.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.

        Args:
            input_shape (object): shape of input tensor.
        """

        self.WQ = self.add_weight(
            name="WQ",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        print("WQ shape:", self.WQ.shape)
        #print("WK shape:", self.WK.shape)
        #print("WV shape:", self.WV.shape)
        super(SelfAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): sequence length of inputs.
            mode (str): mode of mask.

        Returns:
            object: tensors after masking.
        """

        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. qeury, key and value.

        Returns:
            object: ouput tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        
        
        #print("Q_seq:", Q_seq.shape)
        #print("K_seq:", K_seq.shape)
        #print("V_seq:", V_seq.shape)
        #print("WQ weight:", self.WQ.shape)
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads, self.head_dim)
        )
        Q_seq = K.permute_dimensions(Q_seq, pattern=(0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim)
        )
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim)
        )
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))
        A = tf.matmul(Q_seq, K_seq, adjoint_a=False, adjoint_b=True) / K.sqrt(
            K.cast(self.head_dim, dtype="float32")
        )

        A = K.permute_dimensions(
            A, pattern=(0, 3, 2, 1)
        )  # A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]

        A = self.Mask(A, V_len, "add")
        A = K.permute_dimensions(A, pattern=(0, 3, 2, 1))

        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1, num_upper=0)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = K.softmax(A)

        O_seq = tf.matmul(A, V_seq, adjoint_a=True, adjoint_b=False)
        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))

        O_seq = K.reshape(O_seq, shape=(-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")
        #print("Q_seq shape (before transform):", Q_seq.shape)
        #print("K_seq shape (before transform):", K_seq.shape)
        #print("V_seq shape (before transform):", V_seq.shape)
        #print("Attention logits shape:", A.shape)
        #print("Output sequence shape:", O_seq.shape)
        return O_seq

    def get_config(self):
        """add multiheads, multiheads and mask_right into layer config.

        Returns:
            dict: config of SelfAttention layer.
        """
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config



class NRMSModel:
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
    """

    def __init__(self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = None,
    ):
        """Initialization steps for NRMS."""
        self.hparams = hparams
        self.seed = seed

        # SET SEED:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # INIT THE WORD-EMBEDDINGS:
        if word2vec_embedding is None:
            self.word2vec_embedding = np.random.rand(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = word2vec_embedding

        # BUILD AND COMPILE MODEL:
        #Builds the model
        self.model, self.scorer = self._build_graph()

        #
        data_loss = self._get_loss(self.hparams.loss)

        #
        train_optimizer = self._get_opt(
            optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
        )

        self.model.compile(loss=data_loss, optimizer=train_optimizer)

    def _get_loss(self, loss: str):
        """Make loss function, consists of data loss and regularization loss
        Returns:
            object: Loss function or loss function name
        """
        if loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError(f"this loss not defined {loss}")
        return data_loss

    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        # TODO: shouldn't be a string input you should just set the optimizer, to avoid stuff like this:
        # => 'WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.'
        if optimizer == "adam":
            train_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")
        return train_opt

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        #print("Model input shape (history):", his_input_title.shape)
        #print("Model input shape (prediction):", pred_input_title.shape)
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
        )
        print("USERENCODER - His_in:", his_input_title.shape)
        click_title_presents = tf.keras.layers.TimeDistributed(titleencoder)(
            his_input_title
        )

        print("USERENCODER - Clicked:", click_title_presents.shape)

        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        print("USERENCODER - att 1:", y.shape)
        user_present = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        print("USERENCODER - att 2:", user_present.shape)
        model = tf.keras.Model(his_input_title, user_present, name="user_encoder")
        
        return model

    def _build_newsencoder(self):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        embedding_layer = tf.keras.layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word2vec_embedding.shape[1],
            weights=[self.word2vec_embedding],
            trainable=True,
        )
        sequences_input_title = tf.keras.Input(
            shape=(self.hparams.title_size,), dtype="int32"
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)
        print("NEWSENCODER - embedded layer:", embedded_sequences_title.shape)
        y = tf.keras.layers.Dropout(self.hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [y, y, y]
        )
        print("NEWSENCODER - att 1:", y.shape)
        y = tf.keras.layers.Dropout(self.hparams.dropout)(y)
        pred_title = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)
        print("NEWSENCODER - att 2:", pred_title.shape)
        model = tf.keras.Model(sequences_input_title, pred_title, name="news_encoder")
        
        
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size),
            dtype="int32",
        )
        #print("MODEL - his_input: ", his_input_title.shape)
        pred_input_title = tf.keras.Input(
            # shape = (hparams.npratio + 1, hparams.title_size)
            shape=(None, self.hparams.title_size),
            dtype="int32",
        )
        #print("MODEL - pred_input: ", pred_input_title.shape)
        pred_input_title_one = tf.keras.Input(
            shape=(
                1,
                self.hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_one_reshape = tf.keras.layers.Reshape((self.hparams.title_size,))(
            pred_input_title_one
        )
        titleencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        user_present = self.userencoder(his_input_title)
        news_present = tf.keras.layers.TimeDistributed(self.newsencoder)(
            pred_input_title
        )
        news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = tf.keras.layers.Dot(axes=-1)([news_present, user_present])
        #print("MODEL - final pred: ", preds.shape)

        preds = tf.keras.layers.Activation(activation="softmax")(preds)

        pred_one = tf.keras.layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)
        
        model = tf.keras.Model([his_input_title, pred_input_title], preds)
        scorer = tf.keras.Model([his_input_title, pred_input_title_one], pred_one)

        
        return model, scorer
    


