import datetime

from tensorflow.keras.callbacks import Callback

# Source: <https://stackoverflow.com/q/47731935/5811400>
class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 1-tuples (validation_data,)
        or 2-tuples (validation_data, validation_set_name)
        or 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super().__init__()
        
        self.validation_sets = [
            self._expand_validation_set(val_set, idx)
            for idx, val_set in enumerate(validation_sets)
        ]
            
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        
    def _expand_validation_set(self, val_set, idx):
        if len(val_set) == 1:
            return val_set[0], None, None, f'val_{idx}'
        elif len(val_set) == 2:
            return val_set[0], None, None, val_set[1]
        elif len(val_set) == 3:
            return val_set[0], val_set[1], None, val_set[2]
        elif len(val_set) == 4:
            return val_set
        else:
            raise ValueError('every validation set must be either in the form (data,), (data, name), (x, y, name) or (x, y, sample_weights, name).')

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            validation_data, validation_targets, sample_weights, validation_set_name = validation_set

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)
            
            names = ['loss'] + [m.name for m in self.model.metrics]
            full_names = [validation_set_name + '_' + name for name in names]
            full_results = [logs['loss']] + results

            for full_name, result in zip(full_names, full_results):
                self.history.setdefault(full_name, []).append(result)
                
            if self.verbose >= 1:
                values_str = ' - '.join([
                    f'{name}: {result}'
                    for name, result in zip(names, full_results)
                ])
                print(f'{validation_set_name}[{values_str}]')
                
class TimePrinter(Callback):
    def __init__(
        self,
        streamer=print,
        fmt='%Y-%m-%d %H:%M:%S',
        when=None,
    ):
        super().__init__()
                
        self.streamer = streamer
        self.fmt = fmt
        
        if when is None:
            when = {
                'on_batch_begin',
                'on_batch_end',
                'on_epoch_begin',
                'on_epoch_end',
                'on_predict_batch_begin',
                'on_predict_batch_end',
                'on_predict_begin',
                'on_predict_end',
                'on_test_batch_begin',
                'on_test_batch_end',
                'on_test_begin',
                'on_test_end',
                'on_train_batch_begin',
                'on_train_batch_end',
                'on_train_begin',
                'on_train_end',
            }
        self.when = when
        
    def _str_now(self):
        return datetime.datetime.now().strftime(self.fmt)
    
    def on_batch_begin(self, *args, **kwargs):
        if 'on_batch_begin' in self.when:
            self.streamer(f'--- beginning batch at {self._str_now()}', end='')

    def on_batch_end(self, *args, **kwargs):
        if 'on_batch_end' in self.when:
            self.streamer(f' | ending batch at {self._str_now()}')

    def on_epoch_begin(self, *args, **kwargs):
        if 'on_epoch_begin' in self.when:
            self.streamer(f'-- beginning epoch at {self._str_now()}')

    def on_epoch_end(self, *args, **kwargs):
        if 'on_epoch_end' in self.when:
            self.streamer(f'-- ending epoch at {self._str_now()}')

    def on_predict_batch_begin(self, *args, **kwargs):
        if 'on_predict_batch_begin' in self.when:
            self.streamer(f'--- beginning predict_batch at {self._str_now()}', end='')

    def on_predict_batch_end(self, *args, **kwargs):
        if 'on_predict_batch_end' in self.when:
            self.streamer(f' | ending predict_batch at {self._str_now()}')

    def on_predict_begin(self, *args, **kwargs):
        if 'on_predict_begin' in self.when:
            self.streamer(f'- beginning predict at {self._str_now()}')

    def on_predict_end(self, *args, **kwargs):
        if 'on_predict_end' in self.when:
            self.streamer(f'- ending predict at {self._str_now()}')

    def on_test_batch_begin(self, *args, **kwargs):
        if 'on_test_batch_begin' in self.when:
            self.streamer(f'--- beginning test_batch at {self._str_now()}', end='')

    def on_test_batch_end(self, *args, **kwargs):
        if 'on_test_batch_end' in self.when:
            self.streamer(f' | ending test_batch at {self._str_now()}')

    def on_test_begin(self, *args, **kwargs):
        if 'on_test_begin' in self.when:
            self.streamer(f'- beginning test at {self._str_now()}')

    def on_test_end(self, *args, **kwargs):
        if 'on_test_end' in self.when:
            self.streamer(f'- ending test at {self._str_now()}')

    def on_train_batch_begin(self, *args, **kwargs):
        if 'on_train_batch_begin' in self.when:
            self.streamer(f'--- beginning train_batch at {self._str_now()}', end='')

    def on_train_batch_end(self, *args, **kwargs):
        if 'on_train_batch_end' in self.when:
            self.streamer(f' | ending train_batch at {self._str_now()}')

    def on_train_begin(self, *args, **kwargs):
        if 'on_train_begin' in self.when:
            self.streamer(f'- beginning train at {self._str_now()}')

    def on_train_end(self, *args, **kwargs):
        if 'on_train_end' in self.when:
            self.streamer(f'- ending train at {self._str_now()}')

        