import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager
import pandas as pd

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '../features/'
    
    def __init__(self, name):
        self.name = name
        self.df = pd.DataFrame()
        self.df_path = Path(self.dir) / f'{name}.ftr'
    
    def run(self,name = ''):
        with timer(self.name):
            self.create_features(name)
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.df.columns = prefix + self.df.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.df.to_feather(str(self.df_path))


def load_datasets(feats):
    dfs = [pd.read_feather(f'../features/{f}.ftr') for f in feats]
    X_df = pd.concat(dfs, axis=1)
    return X_df