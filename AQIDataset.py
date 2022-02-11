from collections import namedtuple
from random import shuffle
from typing import List, Hashable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def get_date_range(df: pd.DataFrame):
    return pd.date_range(df.index.min(), df.index.max())


def ndarray_to_location(location: np.ndarray) -> Tuple:
    return float(location[0]), float(location[1]), location[2], location[3]


class AQIDataset:
    LAT_COL = 'Latitude'
    LONG_COL = 'Longitude'
    STATE_COL = 'State Name'
    COUNTY_COL = 'County Name'
    DATE_COL = 'Date'
    AQI_COL = 'AQI'
    FEATURE_SCALE_01 = '01'
    FEATURE_SCALE_Z = 'zscore'

    def __init__(self, df: pd.DataFrame, batch_size, input_size=(28,), output_size=(14,),
                 train_test_split=0.8, feature_scaling=None, random_seed=1249):
        self.df = AQIDataset.scale_features(
            df[[AQIDataset.LAT_COL, AQIDataset.LONG_COL, AQIDataset.STATE_COL,
                AQIDataset.COUNTY_COL, AQIDataset.DATE_COL, AQIDataset.AQI_COL]],
            rescale_col=AQIDataset.AQI_COL,
            rescale_type=feature_scaling
        )
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.train_test_split = train_test_split
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.location_groups = AQIDataset.get_location_groups(df)
        self.locations = np.array(list(self.location_groups))
        self.n_locations = self.locations.shape[0]

        locations_train_idx = np.random.randint(0, self.n_locations, size=int(train_test_split * self.n_locations))
        locations_test_idx = np.setdiff1d(range(self.n_locations), locations_train_idx)
        self.locations_train = self.locations[locations_train_idx]
        self.locations_test = self.locations[locations_test_idx]

    @staticmethod
    def scale_features(df: pd.DataFrame, rescale_col, rescale_type) -> pd.DataFrame:
        if rescale_type is None:
            return df
        elif rescale_type == AQIDataset.FEATURE_SCALE_01:
            df.loc[df.index, rescale_col] = (df[rescale_col] - df[rescale_col].min()) / \
                                            (df[rescale_col].max() - df[rescale_col].min())
            return df
        elif rescale_type == AQIDataset.FEATURE_SCALE_Z:
            df.loc[df.index, rescale_col] = (df[rescale_col] - df[rescale_col].mean()) / df[rescale_col].std()
            return df
        raise ValueError(f'feature_scaling must be one of '
                         f'{None, AQIDataset.FEATURE_SCALE_01, AQIDataset.FEATURE_SCALE_Z}')

    @staticmethod
    def get_location_groups(df: pd.DataFrame):
        return df.groupby([AQIDataset.LAT_COL, AQIDataset.LONG_COL, AQIDataset.STATE_COL, AQIDataset.COUNTY_COL]).groups

    def get_df_subset(self, location) -> pd.DataFrame:
        return self.df.loc[self.location_groups[ndarray_to_location(location)]]

    def get_subset(self, location):
        return AQISeries(
            location=location,
            subset=self.get_df_subset(location),
            batch_size=self.batch_size,
            input_size=self.input_size,
            output_size=self.output_size
        )


class AQISeries:
    def __init__(self, location: List, subset: pd.DataFrame, batch_size, input_size, output_size):
        self.location = location
        self.df = subset[[AQIDataset.DATE_COL, AQIDataset.AQI_COL]]
        self.df.index = self.df[AQIDataset.DATE_COL]
        del self.df[AQIDataset.DATE_COL]
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.date_range = get_date_range(self.df)
        self.df = self.df.reindex(self.date_range, fill_value=0).sort_index()
        chunk_size = self.input_size[0] + self.output_size[0]
        self.n = self.df.shape[0] // chunk_size
        self.n_batches = self.n // self.batch_size
        self.xy = self.df.values[:self.n_batches * self.batch_size * chunk_size]\
            .reshape((self.n_batches, self.batch_size, chunk_size, 1))

    def __repr__(self):
        return 'Latitude:{}\nLongitude:{}\nState:{}\nCounty:{}'.format(*self.location)

    def __getitem__(self, i):
        return self.xy[i, :, :self.input_size[0]], self.xy[i, :, self.input_size[0]:]

    def __len__(self):
        return self.n_batches


class AQIDataGenerator(tf.keras.utils.Sequence):
    class SubsetIndexer:
        def __init__(self, bounds):
            self.bounds = np.cumsum(bounds)

        def __getitem__(self, z):
            try:
                i = np.where(z < self.bounds)[0][0]
                j = z - self.bounds[i - 1] if i > 0 else z
                return i, j
            except IndexError:
                raise IndexError('index out of range')

    def __init__(self, dataset: AQIDataset, locations):
        self.dataset = dataset
        self.locations = locations
        self.subsets = list(map(self.dataset.get_subset, self.locations))
        self.n = sum(map(len, self.subsets))
        self.ssi = AQIDataGenerator.SubsetIndexer(list(map(len, self.subsets)))

    # def on_epoch_end(self):
    #     shuffle(self.subsets)
    #     self.ssi = AQIDataGenerator.SubsetIndexer(list(map(len, self.subsets)))

    def __getitem__(self, k):
        i, j = self.ssi[k]
        return self.subsets[i][j]

    def __len__(self):
        return self.n
