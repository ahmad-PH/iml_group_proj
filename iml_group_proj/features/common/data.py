import pickle
import pandas as pd
from typing import Tuple, List

from iml_group_proj.features.common.config import LOC_DATA_PATH

def load(sample_frac:float = 1.0) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    with open(LOC_DATA_PATH, 'rb') as f:
        classes = pickle.load(f)
        train = pickle.load(f)
        test = pickle.load(f)

        return classes, tuple_to_df(train).sample(frac=sample_frac), tuple_to_df(test).sample(frac=sample_frac)


def tuple_to_df(data: List[Tuple]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=["class", "title", "sypnosis", "id"])
