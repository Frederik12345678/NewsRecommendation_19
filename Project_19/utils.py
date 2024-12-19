import os
import polars as pl
from typing import Any, Iterable
import warnings
import inspect
import random
from transformers import AutoTokenizer, AutoModel
import numpy as np


def get_script_directory():
    """
    Returns the path of the current folder
    """

    return os.path.dirname(os.path.abspath(__file__))


def compute_npratio(n_pos: int, n_neg: int) -> float:
    """
    Similar approach as:
        "Neural News Recommendation with Long- and Short-term User Representations (An et al., ACL 2019)"

    Example:
    >>> pos = 492_185
    >>> neg = 9_224_537
    >>> round(compute_npratio(pos, neg), 2)
        18.74
    """
    return 1 / (n_pos / n_neg)


def truncate_history(
    df: pl.DataFrame,
    column: str,
    history_size: int,
    padding_value: Any = None,
    enable_warning: bool = True,
) -> pl.DataFrame:
    """Truncates the history of a column containing a list of items.

    It is the tail of the values, i.e. the history ids should ascending order
    because each subsequent element (original timestamp) is greater than the previous element

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to truncate.
        history_size (int): The maximum size of the history to retain.
        padding_value (Any): Pad each list with specified value, ensuring
            equal length to each element. Default is None (no padding).
        enable_warning (bool): warn the user that history is expected in ascedings order.
            Default is True

    Returns:
        pl.DataFrame: A new DataFrame with the specified column truncated.

    Examples:
    >>> df = pl.DataFrame(
            {"id": [1, 2, 3], "history": [["a", "b", "c"], ["d", "e", "f", "g"], ["h", "i"]]}
        )
    >>> df
        shape: (3, 2)
        ┌─────┬───────────────────┐
        │ id  ┆ history           │
        │ --- ┆ ---               │
        │ i64 ┆ list[str]         │
        ╞═════╪═══════════════════╡
        │ 1   ┆ ["a", "b", "c"]   │
        │ 2   ┆ ["d", "e", … "g"] │
        │ 3   ┆ ["h", "i"]        │
        └─────┴───────────────────┘
    >>> truncate_history(df, 'history', 3)
        shape: (3, 2)
        ┌─────┬─────────────────┐
        │ id  ┆ history         │
        │ --- ┆ ---             │
        │ i64 ┆ list[str]       │
        ╞═════╪═════════════════╡
        │ 1   ┆ ["a", "b", "c"] │
        │ 2   ┆ ["e", "f", "g"] │
        │ 3   ┆ ["h", "i"]      │
        └─────┴─────────────────┘
    >>> truncate_history(df.lazy(), 'history', 3, '-').collect()
        shape: (3, 2)
        ┌─────┬─────────────────┐
        │ id  ┆ history         │
        │ --- ┆ ---             │
        │ i64 ┆ list[str]       │
        ╞═════╪═════════════════╡
        │ 1   ┆ ["a", "b", "c"] │
        │ 2   ┆ ["e", "f", "g"] │
        │ 3   ┆ ["-", "h", "i"] │
        └─────┴─────────────────┘
    """
    if enable_warning:
        function_name = inspect.currentframe().f_code.co_name
        warnings.warn(f"{function_name}: The history IDs expeced in ascending order")
    if padding_value is not None:
        df = df.with_columns(
            pl.col(column)
            .list.reverse()
            .list.eval(pl.element().extend_constant(padding_value, n=history_size))
            .list.reverse()
        )
    return df.with_columns(pl.col(column).list.tail(history_size))


def remove_positives_from_inview(
    df: pl.DataFrame,
    inview_col: str = "article_ids_inview",
    clicked_col: str = "article_ids_clicked",
):
    """Removes all positive article IDs from a DataFrame column containing inview articles and another column containing
    clicked articles. Only negative article IDs (i.e., those that appear in the inview articles column but not in the
    clicked articles column) are retained.

    Args:
        df (pl.DataFrame): A DataFrame with columns containing inview articles and clicked articles.

    Returns:
        pl.DataFrame: A new DataFrame with only negative article IDs retained.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                DEFAULT_CLICKED_ARTICLES_COL: [
                    [1, 2],
                    [1],
                    [3],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ],
            }
        )
    >>> remove_positives_from_inview(df)
        shape: (3, 3)
        ┌─────────┬─────────────────────┬────────────────────┐
        │ user_id ┆ article_ids_clicked ┆ article_ids_inview │
        │ ---     ┆ ---                 ┆ ---                │
        │ i64     ┆ list[i64]           ┆ list[i64]          │
        ╞═════════╪═════════════════════╪════════════════════╡
        │ 1       ┆ [1, 2]              ┆ [3]                │
        │ 1       ┆ [1]                 ┆ [2, 3]             │
        │ 2       ┆ [3]                 ┆ [1, 2]             │
        └─────────┴─────────────────────┴────────────────────┘
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    negative_article_ids = (
        list(filter(lambda x: x not in clicked, inview))
        for inview, clicked in zip(df[inview_col].to_list(), df[clicked_col].to_list())
    )
    return df.with_columns(pl.Series(inview_col, list(negative_article_ids)))



def _check_columns_in_df(df: pl.DataFrame, columns: list[str]) -> None:
    """
    Checks whether all specified columns are present in a Polars DataFrame.
    Raises a ValueError if any of the specified columns are not present in the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        columns (list[str]): The names of the columns to check for.

    Returns:
        None.

    Examples:
    >>> df = pl.DataFrame({"user_id": [1], "first_name": ["J"]})
    >>> check_columns_in_df(df, columns=["user_id", "not_in"])
        ValueError: Invalid input provided. The dataframe does not contain columns ['not_in'].
    """
    columns_not_in_df = [col for col in columns if col not in df.columns]
    if columns_not_in_df:
        raise ValueError(
            f"Invalid input provided. The DataFrame does not contain columns {columns_not_in_df}."
        )


def sampling_strategy_wu2019(
    df: pl.DataFrame,
    npratio: int,
    shuffle: bool = False,
    with_replacement: bool = True,
    seed: int = None,
    inview_col: str = "article_ids_inview",
    clicked_col: str = "article_ids_clicked",
) -> pl.DataFrame:
    """
    Samples negative articles from the inview article pool for a given negative-position-ratio (npratio).
    The npratio (negative article per positive article) is defined as the number of negative article samples
    to draw for each positive article sample.

    This function follows the sampling strategy introduced in the paper "NPA: Neural News Recommendation with
    Personalized Attention" by Wu et al. (KDD '19).

    This is done according to the following steps:
    1. Remove the positive click-article id pairs from the DataFrame.
    2. Explode the DataFrame based on the clicked articles column.
    3. Downsample the inview negative article ids for each exploded row using the specified npratio, either
        with or without replacement.
    4. Concatenate the clicked articles back to the inview articles as lists.
    5. Convert clicked articles column to type List(Int)

    References:
        Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. 2019.
        Npa: Neural news recommendation with personalized attention. In KDD, pages 2576-2584. ACM.

    Args:
        df (pl.DataFrame): The input DataFrame containing click-article id pairs.
        npratio (int): The ratio of negative in-view article ids to positive click-article ids.
        shuffle (bool, optional): Whether to shuffle the order of the in-view article ids in each list. Default is True.
        with_replacement (bool, optional): Whether to sample the inview article ids with or without replacement.
            Default is True.
        seed (int, optional): Random seed for reproducibility. Default is None.
        inview_col (int, optional): inview column name. Default is DEFAULT_INVIEW_ARTICLES_COL,
        clicked_col (int, optional): clicked column name. Default is DEFAULT_CLICKED_ARTICLES_COL,

    Returns:
        pl.DataFrame: A new DataFrame with downsampled in-view article ids for each click according to the specified npratio.
        The DataFrame has the same columns as the input DataFrame.

    Raises:
        ValueError: If npratio is less than 0.
        ValueError: If the input DataFrame does not contain the necessary columns.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL
    >>> import polars as pl
    >>> df = pl.DataFrame(
            {
                "impression_id": [0, 1, 2, 3],
                "user_id": [1, 1, 2, 3],
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3], [1]],
                DEFAULT_CLICKED_ARTICLES_COL: [[1, 2], [1, 3], [1], [1]],
            }
        )
    >>> df
        shape: (4, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [1, 2, 3]          ┆ [1, 2]              │
        │ 1             ┆ 1       ┆ [1, 2, … 4]        ┆ [1, 3]              │
        │ 2             ┆ 2       ┆ [1, 2, 3]          ┆ [1]                 │
        │ 3             ┆ 3       ┆ [1]                ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=1, shuffle=False, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 1]             ┆ [1]                 │
        │ 0             ┆ 1       ┆ [3, 2]             ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 1]             ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 3]             ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 1]             ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, 1]          ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=1, shuffle=True, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 1]             ┆ [1]                 │
        │ 0             ┆ 1       ┆ [2, 3]             ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 1]             ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 3]             ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 1]             ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, 1]          ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    >>> sampling_strategy_wu2019(df, npratio=2, shuffle=False, with_replacement=True, seed=123)
        shape: (6, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ list[i64]           │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 0             ┆ 1       ┆ [3, 3, 1]          ┆ [1]                 │
        │ 0             ┆ 1       ┆ [3, 3, 2]          ┆ [2]                 │
        │ 1             ┆ 1       ┆ [4, 2, 1]          ┆ [1]                 │
        │ 1             ┆ 1       ┆ [4, 2, 3]          ┆ [3]                 │
        │ 2             ┆ 2       ┆ [3, 2, 1]          ┆ [1]                 │
        │ 3             ┆ 3       ┆ [null, null, 1]    ┆ [1]                 │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    # If we use without replacement, we need to ensure there are enough negative samples:
    >>> sampling_strategy_wu2019(df, npratio=2, shuffle=False, with_replacement=False, seed=123)
        polars.exceptions.ShapeError: cannot take a larger sample than the total population when `with_replacement=false`
    ## Either you'll have to remove the samples or split the dataframe yourself and only upsample the samples that doesn't have enough
    >>> min_neg = 2
    >>> sampling_strategy_wu2019(
            df.filter(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len() > (min_neg + 1)),
            npratio=min_neg,
            shuffle=False,
            with_replacement=False,
            seed=123,
        )
        shape: (2, 4)
        ┌───────────────┬─────────┬────────────────────┬─────────────────────┐
        │ impression_id ┆ user_id ┆ article_ids_inview ┆ article_ids_clicked │
        │ ---           ┆ ---     ┆ ---                ┆ ---                 │
        │ i64           ┆ i64     ┆ list[i64]          ┆ i64                 │
        ╞═══════════════╪═════════╪════════════════════╪═════════════════════╡
        │ 1             ┆ 1       ┆ [2, 4, 1]          ┆ 1                   │
        │ 1             ┆ 1       ┆ [2, 4, 3]          ┆ 3                   │
        └───────────────┴─────────┴────────────────────┴─────────────────────┘
    """
    df = (
        # Step 1: Remove the positive 'article_id' from inview articles
        df.pipe(
            remove_positives_from_inview, inview_col=inview_col, clicked_col=clicked_col
        )
        # Step 2: Explode the DataFrame based on the clicked articles column
        .explode(clicked_col)
        # Step 3: Downsample the inview negative 'article_id' according to npratio (negative 'article_id' per positive 'article_id')
        .pipe(
            sample_article_ids,
            n=npratio,
            with_replacement=with_replacement,
            seed=seed,
            inview_col=inview_col,
        )
        # Step 4: Concatenate the clicked articles back to the inview articles as lists
        .with_columns(pl.concat_list([inview_col, clicked_col]))
        # Step 5: Convert clicked articles column to type List(Int):
        .with_columns(pl.col(inview_col).list.tail(1).alias(clicked_col))
    )
    if shuffle:
        df = shuffle_list_column(df, inview_col, seed)
    return df


def shuffle_list_column(
    df: pl.DataFrame, column: str, seed: int = None
) -> pl.DataFrame:
    """Shuffles the values in a list column of a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to shuffle.
        seed (int, optional): An optional seed value.
            Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame with the specified column shuffled.

    Example:
    >>> df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "list_col": [["a-", "b-", "c-"], ["a#", "b#"], ["a@", "b@", "c@"]],
                "rdn": ["h", "e", "y"],
            }
        )
    >>> shuffle_list_column(df, 'list_col', seed=1)
        shape: (3, 3)
        ┌─────┬────────────────────┬─────┐
        │ id  ┆ list_col           ┆ rdn │
        │ --- ┆ ---                ┆ --- │
        │ i64 ┆ list[str]          ┆ str │
        ╞═════╪════════════════════╪═════╡
        │ 1   ┆ ["c-", "b-", "a-"] ┆ h   │
        │ 2   ┆ ["a#", "b#"]       ┆ e   │
        │ 3   ┆ ["b@", "c@", "a@"] ┆ y   │
        └─────┴────────────────────┴─────┘

    No seed:
    >>> shuffle_list_column(df, 'list_col', seed=None)
        shape: (3, 3)
        ┌─────┬────────────────────┬─────┐
        │ id  ┆ list_col           ┆ rdn │
        │ --- ┆ ---                ┆ --- │
        │ i64 ┆ list[str]          ┆ str │
        ╞═════╪════════════════════╪═════╡
        │ 1   ┆ ["b-", "a-", "c-"] ┆ h   │
        │ 2   ┆ ["a#", "b#"]       ┆ e   │
        │ 3   ┆ ["a@", "c@", "b@"] ┆ y   │
        └─────┴────────────────────┴─────┘

    Test_:
    >>> assert (
            sorted(shuffle_list_column(df, "list_col", seed=None)["list_col"].to_list()[0])
            == df["list_col"].to_list()[0]
        )

    >>> df = pl.DataFrame({
            'id': [1, 2, 3],
            'list_col': [[6, 7, 8], [-6, -7, -8], [60, 70, 80]],
            'rdn': ['h', 'e', 'y']
        })
    >>> shuffle_list_column(df.lazy(), 'list_col', seed=2).collect()
        shape: (3, 3)
        ┌─────┬──────────────┬─────┐
        │ id  ┆ list_col     ┆ rdn │
        │ --- ┆ ---          ┆ --- │
        │ i64 ┆ list[i64]    ┆ str │
        ╞═════╪══════════════╪═════╡
        │ 1   ┆ [7, 6, 8]    ┆ h   │
        │ 2   ┆ [-8, -7, -6] ┆ e   │
        │ 3   ┆ [60, 80, 70] ┆ y   │
        └─────┴──────────────┴─────┘

    Test_:
    >>> assert (
            sorted(shuffle_list_column(df, "list_col", seed=None)["list_col"].to_list()[0])
            == df["list_col"].to_list()[0]
        )
    """
    _COLUMN_ORDER = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMN_ORDER, "_groupby_id")

    df = df.with_row_count(GROUPBY_ID)
    df_shuffle = (
        df.explode(column)
        .pipe(shuffle_rows, seed=seed)
        .group_by(GROUPBY_ID)
        .agg(column)
    )
    return (
        df.drop(column)
        .join(df_shuffle, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMN_ORDER)
    )


def generate_unique_name(existing_names: list[str], base_name: str = "new_name"):
    """
    Generate a unique name based on a list of existing names.

    Args:
        existing_names (list of str): The list of existing names.
        base_name (str): The base name to start with. Default is 'newName'.

    Returns:
        str: A unique name.
    Example
    >>> existing_names = ['name1', 'name2', 'newName', 'newName_1']
    >>> generate_unique_name(existing_names, 'newName')
        'newName_2'
    """
    if base_name not in existing_names:
        return base_name

    suffix = 1
    new_name = f"{base_name}_{suffix}"

    while new_name in existing_names:
        suffix += 1
        new_name = f"{base_name}_{suffix}"

    return new_name



def shuffle_rows(df: pl.DataFrame, seed: int = None) -> pl.DataFrame:
    """
    Shuffle the rows of a DataFrame. This methods allows for LazyFrame,
    whereas, 'df.sample(fraction=1)' is not compatible.

    Examples:
    >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    >>> shuffle_rows(df.lazy(), seed=123).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 1   │
        │ 3   ┆ 3   ┆ 3   │
        │ 2   ┆ 2   ┆ 2   │
        └─────┴─────┴─────┘
    >>> shuffle_rows(df.lazy(), seed=None).collect().sort("a")
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 1   │
        │ 2   ┆ 2   ┆ 2   │
        │ 3   ┆ 3   ┆ 3   │
        └─────┴─────┴─────┘

    Test_:
    >>> all([sum(row) == row[0]*3 for row in shuffle_rows(df, seed=None).iter_rows()])
        True

    Note:
        Be aware that 'pl.all().shuffle()' shuffles columns-wise, i.e., with if pl.all().shuffle(None)
        each column's element are shuffled independently from each other (example might change with no seed):
    >>> df_ = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}).select(pl.all().shuffle(None)).sort("a")
    >>> df_
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 3   ┆ 1   │
        │ 2   ┆ 2   ┆ 3   │
        │ 3   ┆ 1   ┆ 2   │
        └─────┴─────┴─────┘
    >>> all([sum(row) == row[0]*3 for row in shuffle_rows(df_, seed=None).iter_rows()])
        False
    """
    seed = seed if seed is not None else random.randint(1, 1_000_000)
    return df.select(pl.all().shuffle(seed))



def sample_article_ids(
    df: pl.DataFrame,
    n: int,
    with_replacement: bool = False,
    seed: int = None,
    inview_col: str = "article_ids_inview",
) -> pl.DataFrame:
    """
    Randomly sample article IDs from each row of a DataFrame with or without replacement

    Args:
        df: A polars DataFrame containing the column of article IDs to be sampled.
        n: The number of article IDs to sample from each list.
        with_replacement: A boolean indicating whether to sample with replacement.
            Default is False.
        seed: An optional seed to use for the random number generator.

    Returns:
        A new polars DataFrame with the same columns as `df`, but with the article
        IDs in the specified column replaced by a list of `n` sampled article IDs.

    Examples:
    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "clicked": [
                    [1],
                    [4, 5],
                    [7, 8, 9],
                ],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    ["A", "B", "C"],
                    ["D", "E", "F"],
                    ["G", "H", "I"],
                ],
                "col" : [
                    ["h"],
                    ["e"],
                    ["y"]
                ]
            }
        )
    >>> print(df)
        shape: (3, 3)
        ┌──────────────────┬─────────────────┬───────────┐
        │ list_destination ┆ article_ids     ┆ col       │
        │ ---              ┆ ---             ┆ ---       │
        │ list[i64]        ┆ list[str]       ┆ list[str] │
        ╞══════════════════╪═════════════════╪═══════════╡
        │ [1]              ┆ ["A", "B", "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "E", "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "H", "I"] ┆ ["y"]     │
        └──────────────────┴─────────────────┴───────────┘
    >>> sample_article_ids(df, n=2, seed=42)
        shape: (3, 3)
        ┌──────────────────┬─────────────┬───────────┐
        │ list_destination ┆ article_ids ┆ col       │
        │ ---              ┆ ---         ┆ ---       │
        │ list[i64]        ┆ list[str]   ┆ list[str] │
        ╞══════════════════╪═════════════╪═══════════╡
        │ [1]              ┆ ["A", "C"]  ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "F"]  ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "I"]  ┆ ["y"]     │
        └──────────────────┴─────────────┴───────────┘
    >>> sample_article_ids(df.lazy(), n=4, with_replacement=True, seed=42).collect()
        shape: (3, 3)
        ┌──────────────────┬───────────────────┬───────────┐
        │ list_destination ┆ article_ids       ┆ col       │
        │ ---              ┆ ---               ┆ ---       │
        │ list[i64]        ┆ list[str]         ┆ list[str] │
        ╞══════════════════╪═══════════════════╪═══════════╡
        │ [1]              ┆ ["A", "A", … "C"] ┆ ["h"]     │
        │ [4, 5]           ┆ ["D", "D", … "F"] ┆ ["e"]     │
        │ [7, 8, 9]        ┆ ["G", "G", … "I"] ┆ ["y"]     │
        └──────────────────┴───────────────────┴───────────┘
    """
    _check_columns_in_df(df, [inview_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")
    df = df.with_row_count(name=GROUPBY_ID)

    df_ = (
        df.explode(inview_col)
        .group_by(GROUPBY_ID)
        .agg(
            pl.col(inview_col).sample(n=n, with_replacement=with_replacement, seed=seed)
        )
    )
    return (
        df.drop(inview_col)
        .join(df_, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS)
    )



def create_binary_labels_column(
    df: pl.DataFrame,
    shuffle: bool = True,
    seed: int = None,
    clicked_col: str = "article_ids_clicked",
    inview_col: str = "article_ids_inview",
    label_col: str = "labels",
) -> pl.DataFrame:
    """Creates a new column in a DataFrame containing binary labels indicating
    whether each article ID in the "article_ids" column is present in the corresponding
    "list_destination" column.

    Args:
        df (pl.DataFrame): The input DataFrame.

    Returns:
        pl.DataFrame: A new DataFrame with an additional "labels" column.

    Examples:
    >>> from ebrec.utils._constants import (
            DEFAULT_CLICKED_ARTICLES_COL,
            DEFAULT_INVIEW_ARTICLES_COL,
            DEFAULT_LABELS_COL,
        )
    >>> df = pl.DataFrame(
            {
                DEFAULT_INVIEW_ARTICLES_COL: [[1, 2, 3], [4, 5, 6], [7, 8]],
                DEFAULT_CLICKED_ARTICLES_COL: [[2, 3, 4], [3, 5], None],
            }
        )
    >>> create_binary_labels_column(df)
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [1, 2, 3]          ┆ [2, 3, 4]           ┆ [0, 1, 1] │
        │ [4, 5, 6]          ┆ [3, 5]              ┆ [0, 1, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    >>> create_binary_labels_column(df.lazy(), shuffle=True, seed=123).collect()
        shape: (3, 3)
        ┌────────────────────┬─────────────────────┬───────────┐
        │ article_ids_inview ┆ article_ids_clicked ┆ labels    │
        │ ---                ┆ ---                 ┆ ---       │
        │ list[i64]          ┆ list[i64]           ┆ list[i8]  │
        ╞════════════════════╪═════════════════════╪═══════════╡
        │ [3, 1, 2]          ┆ [2, 3, 4]           ┆ [1, 0, 1] │
        │ [5, 6, 4]          ┆ [3, 5]              ┆ [1, 0, 0] │
        │ [7, 8]             ┆ null                ┆ [0, 0]    │
        └────────────────────┴─────────────────────┴───────────┘
    Test_:
    >>> assert create_binary_labels_column(df, shuffle=False)[DEFAULT_LABELS_COL].to_list() == [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0],
        ]
    >>> assert create_binary_labels_column(df, shuffle=True)[DEFAULT_LABELS_COL].list.sum().to_list() == [
            2,
            1,
            0,
        ]
    """
    _check_columns_in_df(df, [inview_col, clicked_col])
    _COLUMNS = df.columns
    GROUPBY_ID = generate_unique_name(_COLUMNS, "_groupby_id")

    df = df.with_row_index(GROUPBY_ID)

    if shuffle:
        df = shuffle_list_column(df, column=inview_col, seed=seed)

    df_labels = (
        df.explode(inview_col)
        .with_columns(
            pl.col(inview_col).is_in(pl.col(clicked_col)).cast(pl.Int8).alias(label_col)
        )
        .group_by(GROUPBY_ID)
        .agg(label_col)
    )
    return (
        df.join(df_labels, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
        .select(_COLUMNS + [label_col])
    )



def get_transformers_word_embeddings(model: AutoModel):
    return model.embeddings.word_embeddings.weight.data.to("cpu").numpy()


def concat_str_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    >>> df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "first_name": ["John", "Jane", "Alice"],
                "last_name": ["Doe", "Doe", "Smith"],
            }
        )
    >>> concatenated_df, concatenated_column_name = concat_str_columns(df, columns=['first_name', 'last_name'])
    >>> concatenated_df
        shape: (3, 4)
        ┌─────┬────────────┬───────────┬──────────────────────┐
        │ id  ┆ first_name ┆ last_name ┆ first_name-last_name │
        │ --- ┆ ---        ┆ ---       ┆ ---                  │
        │ i64 ┆ str        ┆ str       ┆ str                  │
        ╞═════╪════════════╪═══════════╪══════════════════════╡
        │ 1   ┆ John       ┆ Doe       ┆ John Doe             │
        │ 2   ┆ Jane       ┆ Doe       ┆ Jane Doe             │
        │ 3   ┆ Alice      ┆ Smith     ┆ Alice Smith          │
        └─────┴────────────┴───────────┴──────────────────────┘
    """
    concat_name = "-".join(columns)
    concat_columns = df.select(pl.concat_str(columns, separator=" ").alias(concat_name))
    return df.with_columns(concat_columns), concat_name




def convert_text2encoding_with_transformers(
    df: pl.DataFrame,
    tokenizer: AutoTokenizer,
    column: str,
    max_length: int = None,
) -> pl.DataFrame:
    """Converts text in a specified DataFrame column to tokens using a provided tokenizer.
    Args:
        df (pl.DataFrame): The input DataFrame containing the text column.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text. (from transformers import AutoTokenizer)
        column (str): The name of the column containing the text.
        max_length (int, optional): The maximum length of the encoded tokens. Defaults to None.
    Returns:
        pl.DataFrame: A new DataFrame with an additional column containing the encoded tokens.
    Example:
    >>> from transformers import AutoTokenizer
    >>> import polars as pl
    >>> df = pl.DataFrame({
            'text': ['This is a test.', 'Another test string.', 'Yet another one.']
        })
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> encoded_df, new_column = convert_text2encoding_with_transformers(df, tokenizer, 'text', max_length=20)
    >>> print(encoded_df)
        shape: (3, 2)
        ┌──────────────────────┬───────────────────────────────┐
        │ text                 ┆ text_encode_bert-base-uncased │
        │ ---                  ┆ ---                           │
        │ str                  ┆ list[i64]                     │
        ╞══════════════════════╪═══════════════════════════════╡
        │ This is a test.      ┆ [2023, 2003, … 0]             │
        │ Another test string. ┆ [2178, 3231, … 0]             │
        │ Yet another one.     ┆ [2664, 2178, … 0]             │
        └──────────────────────┴───────────────────────────────┘
    >>> print(new_column)
        text_encode_bert-base-uncased
    """
    text = df[column].to_list()
    # set columns
    new_column = f"{column}_encode_{tokenizer.name_or_path}"
    # If 'max_length' is provided then set it, else encode each string its original length
    padding = "max_length" if max_length else False
    encoded_tokens = tokenizer(
        text,
        add_special_tokens=False,
        padding=padding,
        max_length=max_length,
        truncation=True,
    )["input_ids"]
    return df.with_columns(pl.Series(new_column, encoded_tokens)), new_column


def create_article_id_to_value_mapping_list(
    df: pl.DataFrame,
    value_col: list[str],  # Allow a list of column names
    article_col: str = "article_id",
) -> dict:
    """
    Create a mapping from article_id to multiple values.
    Combines the values from multiple columns into a tuple for each article_id.
    """
    # Combine value columns into a single structured column
    df = df.with_columns(pl.struct(value_col).alias("combined_values"))
    
    # Use `combined_values` as the value column in the lookup dictionary
    return create_lookup_dict(df, key=article_col, value="combined_values")


def create_article_id_to_value_mapping(
    df: pl.DataFrame,
    value_col: str,
    article_col: str = "article_id",
):
    return create_lookup_dict(
        df.select(article_col, value_col), key=article_col, value=value_col
    )



def create_lookup_dict(df: pl.DataFrame, key: str, value: str) -> dict:
    """
    Creates a dictionary lookup table from a Pandas-like DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame from which to create the lookup table.
        key (str): The name of the column containing the keys for the lookup table.
        value (str): The name of the column containing the values for the lookup table.

    Returns:
        dict: A dictionary where the keys are the values from the `key` column of the DataFrame
            and the values are the values from the `value` column of the DataFrame.

    Example:
        >>> df = pl.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        >>> create_lookup_dict(df, 'id', 'name')
            {1: 'Alice', 2: 'Bob', 3: 'Charlie'}
    """
    return dict(zip(df[key], df[value]))



def add_prediction_scores(
    df: pl.DataFrame,
    scores: Iterable[float],
    prediction_scores_col: str = "scores",
    inview_col: str = "article_ids_inview",
) -> pl.DataFrame:
    """
    Adds prediction scores to a DataFrame for the corresponding test predictions.

    Args:
        df (pl.DataFrame): The DataFrame to which the prediction scores will be added.
        test_prediction (Iterable[float]): A list, array or simialr of prediction scores for the test data.

    Returns:
        pl.DataFrame: The DataFrame with the prediction scores added.

    Raises:
        ValueError: If there is a mismatch in the lengths of the list columns.

    >>> from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL
    >>> df = pl.DataFrame(
            {
                "id": [1,2],
                DEFAULT_INVIEW_ARTICLES_COL: [
                    [1, 2, 3],
                    [4, 5],
                ],
            }
        )
    >>> test_prediction = [[0.3], [0.4], [0.5], [0.6], [0.7]]
    >>> add_prediction_scores(df.lazy(), test_prediction).collect()
        shape: (2, 3)
        ┌─────┬─────────────┬────────────────────────┐
        │ id  ┆ article_ids ┆ prediction_scores_test │
        │ --- ┆ ---         ┆ ---                    │
        │ i64 ┆ list[i64]   ┆ list[f32]              │
        ╞═════╪═════════════╪════════════════════════╡
        │ 1   ┆ [1, 2, 3]   ┆ [0.3, 0.4, 0.5]        │
        │ 2   ┆ [4, 5]      ┆ [0.6, 0.7]             │
        └─────┴─────────────┴────────────────────────┘
    ## The input can can also be an np.array
    >>> add_prediction_scores(df.lazy(), np.array(test_prediction)).collect()
        shape: (2, 3)
        ┌─────┬─────────────┬────────────────────────┐
        │ id  ┆ article_ids ┆ prediction_scores_test │
        │ --- ┆ ---         ┆ ---                    │
        │ i64 ┆ list[i64]   ┆ list[f32]              │
        ╞═════╪═════════════╪════════════════════════╡
        │ 1   ┆ [1, 2, 3]   ┆ [0.3, 0.4, 0.5]        │
        │ 2   ┆ [4, 5]      ┆ [0.6, 0.7]             │
        └─────┴─────────────┴────────────────────────┘
    """
    GROUPBY_ID = generate_unique_name(df.columns, "_groupby_id")
    # df_preds = pl.DataFrame()
    scores = (
        df.lazy()
        .select(pl.col(inview_col))
        .with_row_index(GROUPBY_ID)
        .explode(inview_col)
        .with_columns(pl.Series(prediction_scores_col, scores).explode())
        .group_by(GROUPBY_ID)
        .agg(inview_col, prediction_scores_col)
        .sort(GROUPBY_ID)
        .collect()
    )
    # Only drop GROUPBY_ID if it exists in the DataFrame
    if GROUPBY_ID in df.columns:
        return df.with_columns(scores.select(prediction_scores_col)).drop(GROUPBY_ID)
    else:
        return df.with_columns(scores.select(prediction_scores_col))



def add_known_user_column(
    df: pl.DataFrame,
    known_users: Iterable[int],
    user_col: str = "user_id",
    known_user_col: str = "is_known_user",
) -> pl.DataFrame:
    """
    Adds a new column to the DataFrame indicating whether the user ID is in the list of known users.
    Args:
        df: A Polars DataFrame object.
        known_users: An iterable of integers representing the known user IDs.
    Returns:
        A new Polars DataFrame with an additional column 'is_known_user' containing a boolean value
        indicating whether the user ID is in the list of known users.
    Examples:
        >>> df = pl.DataFrame({'user_id': [1, 2, 3, 4]})
        >>> add_known_user_column(df, [2, 4])
            shape: (4, 2)
            ┌─────────┬───────────────┐
            │ user_id ┆ is_known_user │
            │ ---     ┆ ---           │
            │ i64     ┆ bool          │
            ╞═════════╪═══════════════╡
            │ 1       ┆ false         │
            │ 2       ┆ true          │
            │ 3       ┆ false         │
            │ 4       ┆ true          │
            └─────────┴───────────────┘
    """
    return df.with_columns(pl.col(user_col).is_in(known_users).alias(known_user_col))



def map_list_article_id_to_value(
    behaviors: pl.DataFrame,
    behaviors_column: str,
    mapping: dict[int, pl.Series],
    drop_nulls: bool = False,
    fill_nulls: any = None,
) -> pl.DataFrame:
    """

    Maps the values of a column in a DataFrame `behaviors` containing article IDs to their corresponding values
    in a column in another DataFrame `articles`. The mapping is performed using a dictionary constructed from
    the two DataFrames. The resulting DataFrame has the same columns as `behaviors`, but with the article IDs
    replaced by their corresponding values.

    Args:
        behaviors (pl.DataFrame): The DataFrame containing the column to be mapped.
        behaviors_column (str): The name of the column to be mapped in `behaviors`.
        mapping (dict[int, pl.Series]): A dictionary with article IDs as keys and corresponding values as values.
            Note, 'replace' works a lot faster when values are of type pl.Series!
        drop_nulls (bool): If `True`, any rows in the resulting DataFrame with null values will be dropped.
            If `False` and `fill_nulls` is specified, null values in `behaviors_column` will be replaced with `fill_null`.
        fill_nulls (Optional[any]): If specified, any null values in `behaviors_column` will be replaced with this value.

    Returns:
        pl.DataFrame: A new DataFrame with the same columns as `behaviors`, but with the article IDs in
            `behaviors_column` replaced by their corresponding values in `mapping`.

    Example:
    >>> behaviors = pl.DataFrame(
            {"user_id": [1, 2, 3, 4, 5], "article_ids": [["A1", "A2"], ["A2", "A3"], ["A1", "A4"], ["A4", "A4"], None]}
        )
    >>> articles = pl.DataFrame(
            {
                "article_id": ["A1", "A2", "A3"],
                "article_type": ["News", "Sports", "Entertainment"],
            }
        )
    >>> articles_dict = dict(zip(articles["article_id"], articles["article_type"]))
    >>> map_list_article_id_to_value(
            behaviors=behaviors,
            behaviors_column="article_ids",
            mapping=articles_dict,
            fill_nulls="Unknown",
        )
        shape: (4, 2)
        ┌─────────┬─────────────────────────────┐
        │ user_id ┆ article_ids                 │
        │ ---     ┆ ---                         │
        │ i64     ┆ list[str]                   │
        ╞═════════╪═════════════════════════════╡
        │ 1       ┆ ["News", "Sports"]          │
        │ 2       ┆ ["Sports", "Entertainment"] │
        │ 3       ┆ ["News", "Unknown"]         │
        │ 4       ┆ ["Unknown", "Unknown"]      │
        │ 5       ┆ ["Unknown"]                 │
        └─────────┴─────────────────────────────┘
    >>> map_list_article_id_to_value(
            behaviors=behaviors,
            behaviors_column="article_ids",
            mapping=articles_dict,
            drop_nulls=True,
        )
        shape: (4, 2)
        ┌─────────┬─────────────────────────────┐
        │ user_id ┆ article_ids                 │
        │ ---     ┆ ---                         │
        │ i64     ┆ list[str]                   │
        ╞═════════╪═════════════════════════════╡
        │ 1       ┆ ["News", "Sports"]          │
        │ 2       ┆ ["Sports", "Entertainment"] │
        │ 3       ┆ ["News"]                    │
        │ 4       ┆ null                        │
        │ 5       ┆ null                        │
        └─────────┴─────────────────────────────┘
    >>> map_list_article_id_to_value(
            behaviors=behaviors,
            behaviors_column="article_ids",
            mapping=articles_dict,
            drop_nulls=False,
        )
        shape: (4, 2)
        ┌─────────┬─────────────────────────────┐
        │ user_id ┆ article_ids                 │
        │ ---     ┆ ---                         │
        │ i64     ┆ list[str]                   │
        ╞═════════╪═════════════════════════════╡
        │ 1       ┆ ["News", "Sports"]          │
        │ 2       ┆ ["Sports", "Entertainment"] │
        │ 3       ┆ ["News", null]              │
        │ 4       ┆ [null, null]                │
        │ 5       ┆ [null]                      │
        └─────────┴─────────────────────────────┘
    """
    GROUPBY_ID = generate_unique_name(behaviors.columns, "_groupby_id")
    behaviors = behaviors.lazy().with_row_index(GROUPBY_ID)
    # =>
    select_column = (
        behaviors.select(pl.col(GROUPBY_ID), pl.col(behaviors_column))
        .explode(behaviors_column)
        .with_columns(pl.col(behaviors_column).replace(mapping, default=None))
        .collect()
    )
    # =>
    if drop_nulls:
        select_column = select_column.drop_nulls()
    elif fill_nulls is not None:
        select_column = select_column.with_columns(
            pl.col(behaviors_column).fill_null(fill_nulls)
        )
    # =>
    select_column = (
        select_column.lazy().group_by(GROUPBY_ID).agg(behaviors_column).collect()
    )
    return (
        behaviors.drop(behaviors_column)
        .collect()
        .join(select_column, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
    )



def repeat_by_list_values_from_matrix(
    input_array: np.array,
    matrix: np.array,
    repeats: np.array,
) -> np.array:
    """
    Example:
        >>> input = np.array([[1, 0], [0, 0]])
        >>> matrix = np.array([[7,8,9], [10,11,12]])
        >>> repeats = np.array([1, 2])
        >>> repeat_by_list_values_from_matrix(input, matrix, repeats)
            array([[[10, 11, 12],
                    [ 7,  8,  9]],
                    [[ 7,  8,  9],
                    [ 7,  8,  9]],
                    [[ 7,  8,  9],
                    [ 7,  8,  9]]])
    """
    return np.repeat(matrix[input_array], repeats=repeats, axis=0)



def create_lookup_objects(
    lookup_dictionary: dict[int, np.array], unknown_representation: str, boo: False
) -> tuple[dict[int, pl.Series], np.array]:
    """Creates lookup objects for efficient data retrieval.

    This function generates a dictionary of indexes and a matrix from the given lookup dictionary.
    The generated lookup matrix has an additional row based on the specified unknown representation
    which could be either zeros or the mean of the values in the lookup dictionary.

    Args:
        lookup_dictionary (dict[int, np.array]): A dictionary where keys are unique identifiers (int)
            and values are some representations which can be any data type, commonly used for lookup operations.
        unknown_representation (str): Specifies the method to represent unknown entries.
            It can be either 'zeros' to represent unknowns with a row of zeros, or 'mean' to represent
            unknowns with a row of mean values computed from the lookup dictionary.

    Raises:
        ValueError: If the unknown_representation is not either 'zeros' or 'mean',
            a ValueError will be raised.

    Returns:
        tuple[dict[int, pl.Series], np.array]: A tuple containing two items:
            - A dictionary with the same keys as the lookup_dictionary where values are polars Series
                objects containing a single value, which is the index of the key in the lookup dictionary.
            - A numpy array where the rows correspond to the values in the lookup_dictionary and an
                additional row representing unknown entries as specified by the unknown_representation argument.

    Example:
    >>> data = {
            10: np.array([0.1, 0.2, 0.3]),
            20: np.array([0.4, 0.5, 0.6]),
            30: np.array([0.7, 0.8, 0.9]),
        }
    >>> lookup_dict, lookup_matrix = create_lookup_objects(data, "zeros")

    >>> lookup_dict
        {10: shape: (1,)
            Series: '' [i64]
            [
                    1
            ], 20: shape: (1,)
            Series: '' [i64]
            [
                    2
            ], 30: shape: (1,)
            Series: '' [i64]
            [
                    3
        ]}
    >>> lookup_matrix
        array([[0. , 0. , 0. ],
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]])
    """
    # MAKE LOOKUP DICTIONARY
    lookup_indexes = {
        id: pl.Series("", [i]) for i, id in enumerate(lookup_dictionary, start=1)
    }
    # MAKE LOOKUP MATRIX
    lookup_matrix = np.array(list(lookup_dictionary.values()))

    if boo:
        lookup_matrix = np.array([[item] for item in lookup_matrix])

    if unknown_representation == "zeros":
        UNKNOWN_ARRAY = np.zeros(lookup_matrix.shape[1], dtype=lookup_matrix.dtype)
    elif unknown_representation == "mean":
        UNKNOWN_ARRAY = np.mean(lookup_matrix, axis=0, dtype=lookup_matrix.dtype)
    else:
        raise ValueError(
            f"'{unknown_representation}' is not a specified method. Can be either 'zeros' or 'mean'."
        )

    lookup_matrix = np.vstack([UNKNOWN_ARRAY, lookup_matrix])
    return lookup_indexes, lookup_matrix



def slice_join_dataframes(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    on: str,
    how: str,
) -> pl.DataFrame:
    """
    Join two dataframes optimized for memory efficiency.
    """
    return pl.concat(
        (
            rows.join(
                df2,
                on=on,
                how=how,
            )
            for rows in df1.iter_slices()
        )
    )