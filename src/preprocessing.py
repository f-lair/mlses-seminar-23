from typing import Dict, Tuple

import numpy as np
import pandas as pd


def format_datetime(row: pd.Series) -> str:
    date = str(row['date'])  # YYYYmmdd
    time = row['time']  # in 0.1s

    Y, m, d = date[0:4], date[4:6], date[6:8]
    H, M, S = int(time // 3600), int((time % 3600) // 60), int(time % 60)

    date = f'{Y}-{m}-{d}'  # YYYY-mm-dd
    time = f'{H:02d}-{M:02d}-{S:02d}'  # HH-MM-SS

    return f'{date} {time}'


# def create_delta_temp_features(row: pd.Series) -> Dict[str, float]:
#     return {f'TD-{sensor}': row[f'T2-{sensor}'] - row[f'T1-{sensor}'] for sensor in SENSORS}


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df.apply(format_datetime, axis=1), format='%Y-%m-%d %H-%M-%S')

    # delta_temp_features = df.apply(
    #     lambda row: create_delta_temp_features(row), axis=1, result_type='expand'
    # )
    # df = pd.concat([df, delta_temp_features], axis=1)

    return df
