from pathlib import Path
from functools import partial
import operator
import itertools as it

import tensorflow as tf    
import more_itertools as mit
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
import pandas as pd
import numpy as np

import boiling_learning as bl

# DEBUG
import pprint
# DEBUG END

_sentinel = object()
AUTOTUNE = tf.data.experimental.AUTOTUNE

# an ImageDataset is a file CSV in df_path and the correspondent images. The file in df_path contains at least two columns. One of this columns contains file paths, and the other the targets for training, validation or test. This is intended for using flow_from_dataframe. There may be an optional column which specifies if that image belongs to the training, the validation or the test sets.
class ImageDataset(bl.utils.SimpleRepr, bl.utils.SimpleStr):
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
    @property
    def train_paths(self):
        return self.paths[self.df[self.set_column] == self.train_key]
    @property
    def val_paths(self):
        return self.paths[self.df[self.set_column] == self.val_key]
    @property
    def test_paths(self):
        return self.paths[self.df[self.set_column] == self.test_key]
    
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
        self._pipe = bl.utils.packed_functional.compose(*transformers)
        
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
    def __init__(
        self,
        transformers,
        loader=None,
        saver=None,
        persist_intermediate=False,
        persist_last=True,
        auto_purge=False
    ):
        # transformers is an iterable yielding (path_transformer, value_transformer)
        
        if loader is None:
            loader = partial(load_persistent, auto_purge=auto_purge)
        self.loader = loader
        
        if saver is None:
            saver = bl.management.Persistent.persist
        self.saver = saver
        
        self.transformers = transformers
        self._persist_intermediate = persist_intermediate
        self._persist_last = persist_last
        self._pipe = None
        self._assembled = False
    
    def _assemble(self): 
        if self._persist_intermediate:
            transformers = mit.intersperse(self.saver, self.transformers)        
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
    
# TODO: test this
class ImageDatasetTransformerTF(bl.utils.SimpleRepr, bl.utils.SimpleStr):
    '''Transforms a sequence of images using a sequence of transformations.
    '''
    
    def __init__(
        self,
        path_transformers,
        transformers,
        batch_size,
        loader,
        saver,
        split_id='all',
        chunk_size=None,
        chunk_index=None,
    ):
        self.path_transformers = path_transformers
        self.transformers = transformers
        self.batch_size = batch_size
        self.loader = loader
        self.saver = saver
        
        self.split_id = bl.model.SplitSubset.get_split(split_id)
        
        if (
            (chunk_size is None and chunk_index is not None)
            or (chunk_size is not None and chunk_index is None)
        ):
            raise ValueError('chunk_size and chunk_index must be both None or both integers.')
        
        self.chunk_size = chunk_size
        self.chunk_index = chunk_index
        
    def is_using_chunks(self):
        return self.chunk_size is not None

    def _extract_paths(self, img_ds, split_id=None):
        if split_id is bl.model.SplitSubset.TRAIN:
            df = img_ds.train_paths
        elif split_id is bl.model.SplitSubset.VAL:
            df = img_ds.val_paths
        elif split_id is bl.model.SplitSubset.TEST:
            df = img_ds.test_paths
        elif split_id is bl.model.SplitSubset.ALL:
            df = img_ds.paths
        else:
            raise ValueError(f'split_id={split_id} not supported.')
        df = self._get_chunk(df)
        
        return df
    
    def _get_chunk(self, df):
        if self.is_using_chunks():
            chunks = mit.chunked(df, self.chunk_size)
            chunk = mit.nth_or_last(chunks, self.chunk_index)
            return chunk
        else:
            return df
    
    def _load_tensor(self, sources):
        sources = map(str, sources)
        
        ds = tf.data.Dataset.from_generator(
            lambda: sources,
            tf.string    
        )
        ds = ds.map(
            self.loader,
            num_parallel_calls=AUTOTUNE
        )
        ds = ds.prefetch(AUTOTUNE)
        
        return ds
    
    def _save_tensor(self, dests, ds):
        for dest_chunk, img_chunk in zip(
            mit.ichunked(dests, self.batch_size),
            ds.batch(self.batch_size).as_numpy_iterator()
        ):
            for dest, img in zip(dest_chunk, img_chunk):
                self.saver(img, dest)
    
    def _full_trajectories(self, img_ds):
        sources = self._extract_paths(img_ds, self.split_id)
        sources = map(Path, sources)
        
        def trajectory(source):
            return it.accumulate(
                # self.path_transformers, # Python 3.8 only
                mit.prepend(source, self.path_transformers),
                lambda current_path, path_transformer: path_transformer(current_path),
                # initial=source # Python 3.8 only
            )
        
        trajs = map(trajectory, sources)
        trajs = map(list, trajs)
        
        return trajs
    
    def _valid_trajectories(self, trajs, erased_marker, cmp_marker=operator.is_):
        def split_source_dest(traj):
            traj, erased = mit.partition(
                partial(cmp_marker, erased_marker),
                traj
            )
            dests, possible_sources = mit.partition(
                operator.methodcaller('is_file'),
                traj
            )
            
            possible_sources = list(possible_sources)
            source = possible_sources.pop()
            
            erased = it.chain(
                erased,
                it.repeat(erased_marker, len(possible_sources))
            )
            
            return list(erased) + [source] + list(dests)
        
        trajs = map(split_source_dest, trajs)
        
        return trajs
    
    def _step_from_idx(self, trajs, step_idx, erased_marker, cmp_marker=operator.is_):        
        trajs = it.filterfalse(
            lambda traj: cmp_marker(traj[step_idx], erased_marker), # removes erased trajectories, i.e., the ones that already exist and don't need to be transformed
            trajs
        )
        trajs = map(
            operator.itemgetter(step_idx, step_idx+1),
            trajs
        )
        trajs = mit.peekable(trajs)
        
        if trajs:
            sources, dests = mit.unzip(trajs)
        else:
            sources = bl.utils.empty_gen()
            dests = bl.utils.empty_gen()
            
        return sources, dests
    
    def transform_images(self, img_ds):
        erased_marker = None
        full_trajs = list(self._full_trajectories(img_ds))
        
        for step_idx, transformer in enumerate(self.transformers):
            trajs = trajs if step_idx else full_trajs
            
            trajs = self._valid_trajectories(trajs, erased_marker=erased_marker)
            trajs = list(trajs)
            sources, dests = self._step_from_idx(trajs, step_idx, erased_marker=erased_marker)
            
            ds = self._load_tensor(sources)
            ds = ds.map(
                transformer,
                num_parallel_calls=AUTOTUNE
            )
            
            self._save_tensor(dests, ds)
                
        final = map(
            mit.last,
            full_trajs
        )
        ds = self._load_tensor(final)
            
        return full_trajs, ds
    
    
# Message to mateus.stahelin@lepten.ufsc.br
# Além disso, gostaria de saber se conheces uma solução para o problema que estou tendo, relacionado à mesma questão.Vou explicar o que estou fazendo, e talvez você tenha até uma ideia melhor do que fazer. Eu estou treinando redes neurais na plataforma online Google Colab, que é basicamente um processador de notebooks em Python com acesso direto ao Google Drive e a GPUs. Não é necessário ter nada instalado no computador (além do Google Chrome), nem mesmo o Google Drive. Tudo é feito na nuvem.Um dos obstáculos é que o Google Colab disponibiliza apenas uma quantidade limitada de RAM e processamento para cada usuário (um usuário é definido pela conta de e-mail utilizada). Então o que fiz para paralelizar o trabalho, além de utilizar as GPUs, foi logar no Google Colab em várias páginas do Google Chrome utilizando contas de e-mail diferentes. Por exemplo, em uma página eu entro com a minha conta do laboratório (ruan.comelli@lepten.ufsc.br), e, em outra, com a minha conta pessoal (ruancomelli@gmail.com). No total, estou utilizando 6 contas de e-mail, cada uma treinando uma rede neural diferente. Ao fim do treinamento, eu tenho 6 redes treinadas paralelamente.Os dois problemas que surgiram com isso são:O Google Colab exige que o usuário fique conectado à internet durante a utilização da plataforma. Porém a conexão aqui em casa é ruim e falha sempre. Como resultado, o treinamento das redes é paralisado. Isso me impede de deixá-las treinando durante a noite, por exemplo.Cada página do Google Chrome consome uma quantidade enorme de RAM, e meu computador pessoal não tem capacidade para isso. Quando coloco as redes para treinar, o computador fica inutilizável.Todo o processamento que é feito no Google Colab poderia ser feito localmente, mas isso consumiria muito mais do processamento do computador e exigiria que o meu banco de dados fosse constantemente downloaded e uploaded entre a nuvem e o meu computador.
