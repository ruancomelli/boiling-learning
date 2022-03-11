from typing import Optional

import modin.pandas as pd


def sync_dataframes(
    source_df: pd.DataFrame,
    dest_df: pd.DataFrame,
    source_time_column: Optional[str] = None,
    dest_time_column: Optional[str] = None,
) -> pd.DataFrame:
    allowed_index = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.Float64Index)

    if source_time_column is not None:
        source_df = source_df.set_index(source_time_column, drop=False)
    if not isinstance(source_df.index, allowed_index):
        raise ValueError(
            f'the source DataFrame index must be one of {allowed_index}.'
            ' Ensure this or pass a valid column name as input.'
            f' Got {type(source_df.index)}'
        )

    if dest_time_column is not None:
        dest_df = dest_df.set_index(dest_time_column, drop=False)
    if not isinstance(dest_df.index, allowed_index):
        raise ValueError(
            f'the dest DataFrame index must be one of {allowed_index}.'
            ' Ensure this or pass a valid column name as input.'
            f' Got {type(dest_df.index)}'
        )

    if isinstance(source_df.index, pd.TimedeltaIndex):
        source_df.index = source_df.index.total_seconds()

    if isinstance(dest_df.index, pd.TimedeltaIndex):
        dest_df.index = dest_df.index.total_seconds()

    if type(source_df.index) is not type(dest_df.index):
        raise ValueError(
            f'the source and dest DataFrames indices must be the same type.'
            f' Got {type(source_df.index)} and {type(dest_df.index)}'
        )

    concat = pd.concat([source_df, dest_df]).sort_index()
    if isinstance(source_df.index, pd.Float64Index):
        concat = concat.interpolate(method='index', limit_direction='both')
    else:
        concat = concat.interpolate(method='time', limit_direction='both')
    concat = concat.loc[dest_df.index]
    return concat
