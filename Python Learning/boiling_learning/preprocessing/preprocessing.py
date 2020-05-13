from pathlib import Path
from functools import partial

import more_itertools as mit
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
import pandas as pd
import numpy as np

import boiling_learning as bl

# an ImageDataset is a file CSV in df_path and the correspondent images. The file in df_path contains at least two columns. One of this columns contains file paths, and the other the targets for training, validation or test. This is intended for using flow_from_dataframe. There may be an optional column which specifies if that image belongs to the training, the validation or the test sets.
class ImageDataset:
    def __init__(
        self,
        df_path=None,
        path_column='path',
        target_column='target',
        set_column=None,
        train_key='train',
        val_key='val',
        test_key='test',
        df=None,
        exist_load=False
    ):
        self.df_path = Path(df_path)
        self.path_column = path_column
        self.target_column = target_column
        self.set_column = set_column
        self.train_key = train_key
        self.val_key = val_key
        self.test_key = test_key
        
        if exist_load and self.df_path.is_file():
            self.load()
        else:
            if df is None:
                df = pd.DataFrame()
            self.df = df
    
    def __repr__(self):
        return f'ImageDataset(df_path={self.df_path}, path_column={self.path_column}, target_column={self.target_column}, set_column={self.set_column}, train_key={self.train_key}, val_key={self.val_key}, test_key={self.test_key})'
    
    def __str__(self):
        return '\n'.join((
            f'ImageDataset at {self.df_path}',
            f'path_column={self.path_column}',
            f'target_column={self.target_column}',
            f'set_column={self.set_column}',
            f'train_key={self.train_key}',
            f'val_key={self.val_key}',
            f'test_key={self.test_key}',
            f'{self.df}'
        ))
        
    def load(self, path=None, cheap=False):
        if path is None:
            path = self.df_path
        else:
            self.df_path = path
        
        if cheap:
            if self.set_column is None:
                usecols = [self.path_column, self.target_column]
            else:
                usecols = [self.path_column, self.target_column, self.set_column]
            self.df = pd.read_csv(self.df_path, skipinitialspace=True, usecols=usecols)
        else:
            self.df = pd.read_csv(self.df_path, skipinitialspace=True)
            
        return self
            
    def save(self, path=None, exist_skip=False):
        if path is None:
            path = self.df_path
        path = Path(path)
        
        if not (exist_skip and path.is_file()):
            path.parent.mkdir(exist_ok=True, parents=True)
            self.df.to_csv(path)
        
        return self
    
    def move(self, path, renaming=False, erase=False, exist_skip=False):
        if erase:
            old_path = self.df_path
            
        if renaming:
            self.df_path = self.df_path.with_name(path)
        else:
            self.df_path = Path(path)
            
        self.save(exist_skip=exist_skip)
        
        if erase:
            old_path.unlink(missing_ok=True)
    
    @property
    def data(self):
        return self.df[[self.path_column, self.target_column]]
    
    @data.setter
    def data(self, other):
        self.df[[self.path_column, self.target_column]] = other
    
    @property
    def train_data(self):
        return self.data[self.df[self.set_column] == self.train_key]
    
    @train_data.setter
    def train_data(self, other):
        self.df.loc[self.df[self.set_column] == self.train_key, [self.path_column, self.target_column]] = other
    
    @property
    def val_data(self):
        return self.data[self.df[self.set_column] == self.val_key]
    
    @val_data.setter
    def val_data(self, other):
        self.df.loc[self.df[self.set_column] == self.val_key, [self.path_column, self.target_column]] = other
    
    @property
    def test_data(self):
        return self.data[self.df[self.set_column] == self.test_key]
    
    @test_data.setter
    def test_data(self, other):
        self.df.loc[self.df[self.set_column] == self.test_key, [self.path_column, self.target_column]] = other
    
    @property
    def paths(self):
        return self.df[self.path_column]
    
    @paths.setter
    def paths(self, other):
        self.df[self.path_column] = other
    
    @property
    def targets(self):
        return self.df[self.target_column]
    
    @targets.setter
    def targets(self, other):
        self.df[self.target_column] = other
    
    def modify_path(self, old_path, new_path, many=False):
        if many:
            old_to_new = dict(zip(map(Path, old_path), map(Path, new_path)))
            self.df[self.path_column] = self.df[self.path_column].apply(
                lambda y: old_to_new.get(Path(y), y)
            )
        else:
            self.df[self.path_column] = self.df[self.path_column].mask(
                lambda x: Path(x) == Path(old_path),
                new_path
            )
        
        return self
            
    def transform_images(self, transformer, **kwargs):
        # TODO: remove this code duplication
        # TODO: allow dataset augmentation, not only transformation
        
        return transformer.transform_images(self.paths, **kwargs)
    
    def append(self, other):
        self.append_dataframe(other.df)
        
        return self
    
    def append_dataframe(self, df):
        self.df.append(df, ignore_index=True)
        
        return self        
    
    def split(self, train_size=None, val_size=None, test_size=None, **options):       
        n_samples = len(self.df.index)
        indices = np.arange(n_samples, dtype=int)
        
        indices_train, indices_val, indices_test = bl.model.train_val_test_split(
            indices, n_samples, train_size=train_size, val_size=val_size, test_size=test_size,
            **options
        )
                
        self.df[self.set_column] = pd.Series(pd.concat((
            pd.Series(data=self.train_key, index=indices_train),
            pd.Series(data=self.val_key, index=indices_val),
            pd.Series(data=self.test_key, index=indices_test)
        )), dtype='category')
        
        return self

def load_persistent(path, auto_purge=False):
    def imread_as_float(path):
        try:
            return img_as_float(imread(path))
        except (SyntaxError, IOError):
            if auto_purge:
                path.unlink()
            raise
    
    def imsave_as_ubyte(path, img):
        return imsave(path, img_as_ubyte(img))
    
    return bl.management.Persistent(
        path,
        checker=lambda x: x.is_file(),
        reader=imread_as_float,
        writer=imsave_as_ubyte,
        record_paths=True
    )

class TransformationPipeline:
    def __init__(self, *transformers):
        self._pipe = boiling_learning.utils.packed_functional.compose(*transformers)
        
    def transform(
        self,
        X,
        many=False,
        fetch=None,
        # parallel=False
    ):
        should_fetch = callable(fetch) or bool(fetch)
        
        if many:
            transformer = partial(self.transform, many=False, fetch=fetch)
            
            # if parallel:
                # from pathos.multiprocessing import ProcessingPool
                # # from multiprocessing import Pool
                
                # if isinstance(parallel, bool):
                #     processes = []
                # else:
                #     processes = [parallel]
                    
                # with ProcessingPool(*processes) as pool:
                # # with Pool(*processes) as pool:
                #     bl.utils.print_header(f'Creating pool with ncpus={pool.ncpus} and nodes={pool.nodes}')
                #     if should_fetch:
                #         return pool.map(transformer, X)
                #         # return pool.map(bl.utils.worker.apply_to_f, ((transformer, x, {}) for x in X))
                #         # return pool.map(bl.utils.worker.apply_to_obj, ((self, 'transform', x, {}) for x in X))
                #     else:
                #         mit.consume(pool.uimap(transformer, X))
                #         # mit.consume(pool.imap_unordered(transformer, X))
                #         # mit.consume(pool.imap_unordered(bl.utils.worker.apply_to_f, ((transformer, x, {}) for x in X)))
                #         # mit.consume(pool.imap_unordered(bl.utils.worker.apply_to_obj, ((self, 'transform', x, {}) for x in X)))
                #         return self
            # else:
            mapper = map(transformer, X)
            if should_fetch:
                return list(mapper)
            else:
                mit.consume(mapper)
                return self
        else:
            if should_fetch:
                if callable(fetch):
                    return fetch(self._pipe(X))
                else:
                    return self._pipe(X)
            else:
                self._pipe(X)
                return self

class ImageDatasetTransformer:
    def __init__(self, transformers, loader=None, persist_intermediate=False, persist_last=True, auto_purge=False):
        # transformers is an iterable yielding (path_transformer, value_transformer)
        
        if loader is None:
            loader = partial(load_persistent, auto_purge=auto_purge)
        
        self.loader = loader
        self.transformers = transformers
        self._persist_intermediate = persist_intermediate
        self._persist_last = persist_last
        self._pipe = None
        self._assembled = False
    
    def _assemble(self):        
        if self._persist_intermediate:
            transformers = mit.intersperse(bl.management.Persistent.persist, self.transformers)        
        else:
            transformers = self.transformers

        transformers = mit.prepend(
            self.loader,
            transformers
        )
        
        if self._persist_last:
            transformers = bl.utils.append(transformers, bl.management.Persistent.persist)
        
        self._assembled = True
        self._pipe = TransformationPipeline(*transformers)
    
    def transform_images(self, images, **kwargs):
        if not self._assembled:
            self._assemble()
        return self._pipe.transform(images, many=True, **kwargs)