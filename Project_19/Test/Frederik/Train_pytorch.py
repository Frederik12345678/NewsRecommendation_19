import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import os
import sys
import polars as pl # used to read the .parquet files so its important
import numpy as np
import warnings
import tensorflow as tf

print("training started:")

# Suppress all warnings
warnings.filterwarnings("ignore")

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# Correct imports
from Project_19.utils import add_known_user_column, add_prediction_scores
from Project_19.dataloader.NRMSdataloader import NRMSDataLoader
from Project_19.models.nrms import NRMSModelPytorch
from Project_19.eval.metricEval import MetricEvaluator, AucScore, MrrScore, NdcgScore

# Import additional utilities from utils
from Project_19.utils import (
    get_transformers_word_embeddings,
    concat_str_columns,
    convert_text2encoding_with_transformers,
    create_article_id_to_value_mapping,
    get_script_directory,
    slice_join_dataframes,
    truncate_history,
    sampling_strategy_wu2019,
    create_binary_labels_column
)

def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select("user_id", "article_id_fixed")
        .pipe(
            truncate_history,
            column="article_id_fixed",
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on="user_id",
            how="left",
        )
    )
    return df_behaviors


basic_path = get_script_directory()

PATH = Path(basic_path+"/Data")
DATASPLIT = f"ebnerd_small"  # [ebnerd_demo, ebnerd_small, ebnerd_large]

COLUMNS = [
    "user_id",
    "article_id_fixed",
    "article_ids_inview",
    "article_ids_clicked",
    "impression_id",
]
HISTORY_SIZE = 10
FRACTION = 0.01

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)
# =>
df_validation = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)

df_articles = pl.read_parquet(PATH.joinpath(DATASPLIT,"articles.parquet"))

TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = ["subtitle", "title"]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)

train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column="article_id_fixed",
    eval_mode=False,
    batch_size=8,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column="article_id_fixed",
    eval_mode=True,
    batch_size=8,
)
warnings.filterwarnings("ignore")


class hparams_nrms:
    # INPUT DIMENTIONS:
    title_size: int = 30
    history_size: int = 50
    # MODEL ARCHITECTURE
    head_num: int = 20
    head_dim: int = 20
    attention_hidden_dim: int = 200
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 0.0001

MODEL_NAME = "NRMS_pytorch"
LOG_DIR = f"downloads/runs/{MODEL_NAME}"
MODEL_WEIGHTS = "downloads/data/state_dict/NRMS/weights.weights.h5"

# CALLBACKS
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
#modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    #filepath=MODEL_WEIGHTS, save_best_only=False, save_weights_only=True, verbose=1)

hparams_nrms.history_size = HISTORY_SIZE
model = NRMSModelPytorch(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)
hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=30,
    #callbacks=[tensorboard_callback, early_stopping], #, modelcheckpoint
)

#_ = model.model.load_weights(filepath=MODEL_WEIGHTS)


pred_validation = model.scorer.predict(val_dataloader)


df_validation = add_prediction_scores(df_validation, pred_validation.tolist()).pipe(
    add_known_user_column, known_users=df_train["user_id"]
)

metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
metrics.evaluate()

print(metrics)

print("Training of "+DATASPLIT+" has finished")