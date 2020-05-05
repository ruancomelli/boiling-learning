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
                