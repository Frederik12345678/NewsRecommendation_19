import os

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