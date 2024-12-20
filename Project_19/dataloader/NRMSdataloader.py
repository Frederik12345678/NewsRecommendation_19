import polars as pl
import numpy as np
from dataclasses import dataclass, field
import tensorflow as tf


from utils import (
    map_list_article_id_to_value, 
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)

@dataclass
class NewsrecDataLoader(tf.keras.utils.Sequence):
    """
    A DataLoader for news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    article_dict_2: dict
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    inview_col: str = "article_ids_inview"
    labels_col: str = "labels"
    user_col: str = "user_id"
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation, boo=False
        )

        self.lookup_article_index_2, self.lookup_article_matrix_2 = create_lookup_objects(
            self.article_dict_2, unknown_representation=self.unknown_representation, boo=True
        )

        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self):
        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class NRMSDataLoader(NewsrecDataLoader):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]


        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]

            his_input_time = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix_2,
                repeats=repeats,
            )
            # =>
            pred_input_time = self.lookup_article_matrix_2[
                batch_X[self.inview_col].explode().to_list()
            ]

        else:
            
            batch_y = np.array(batch_y.to_list())

            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]

            his_input_time = self.lookup_article_matrix_2[
                batch_X[self.history_column].to_list()
            ]
            pred_input_time = self.lookup_article_matrix_2[
                batch_X[self.inview_col].to_list()
            ]

            pred_input_time = np.squeeze(pred_input_time, axis=2)

            pred_input_title = np.squeeze(pred_input_title, axis=2)
            #print("pred after :",pred_input_title.shape)

        his_input_time = np.squeeze(his_input_time, axis=2)
        his_input_title = np.squeeze(his_input_title, axis=2)
        #print("pred after :",his_input_title.shape)
        return (his_input_title, pred_input_title), batch_y, (his_input_time,pred_input_time)


