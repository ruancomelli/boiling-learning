from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPool2D

import boiling_learning.utils
from boiling_learning.management import ModelCreator

def is_classification(problem):
    return problem.lower() in {'classification', 'regime'}

def is_regression(problem):
    return problem.lower() in {'regression', 'heat flux', 'h', 'power'}

# CNN #2 implemented according to the paper Hobold and da Silva (2019): Visualization-based nucleate boiling heat flux quantification using machine learning.
def build(
    input_shape,
    dropout_ratio,
    problem='regression',
    num_classes=None,
):
    input_data = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), padding='same', activation='relu')(input_data)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = Dropout(dropout_ratio)(x)
    x = Flatten()(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(dropout_ratio)(x)

    if is_classification(problem):
        predictions = Dense(num_classes, activation='softmax')(x)
    elif is_regression(problem):
        predictions = Dense(1)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=input_data, outputs=predictions)

    return model

def restore(checkpoint):
    last_epoch = -1
    model = None
    if checkpoint.get('restore', False):
        from pathlib import Path
        import parse
        from boiling_learning.utils import append
        
        epoch_str = 'epoch'
        
        path = Path(checkpoint['path'])
        glob_pattern = path.name.replace(f'{{{epoch_str}}}', '*')
        parser = parse.compile(path.name)
        
        paths = path.parent.glob(glob_pattern) 
        parsed = (parser.parse(path_item.name) for path_item in paths)
        succesfull_parsed = filter(lambda p: p is not None and epoch_str in p, parsed)
        epochs = append((int(p[epoch_str]) for p in parsed), last_epoch)
        last_epoch = max(epochs)
                
        if last_epoch != -1:
            path_str = str(path).format(epoch=last_epoch)
            model = checkpoint['load_method'](path_str)
            
    return last_epoch, model

def creator_method(
    input_shape,
    verbose,
    checkpoint,
    dropout_ratio,
    num_classes,
    problem,
    compile_setup,
    fit_setup,
    fetch,
):
    compile_setup, fit_setup = boiling_learning.utils.regularize_default(
        (compile_setup, fit_setup),
        cond=lambda x: x is not None,
        default=lambda x: dict(do=False),
        many=True,
        call_default=True
    )
    
    last_epoch, model = restore(checkpoint)
    initial_epoch = max(last_epoch, 0)
    
    if model is None:        
        model = build(
            input_shape,
            dropout_ratio,
            problem,
            num_classes
        )

        if compile_setup['do']:
            model.compile(**compile_setup['params'])

    history = None
    if fit_setup['do']:
        if initial_epoch < fit_setup['params']['epochs']:
            fit_setup['params']['initial_epoch'] = initial_epoch
            history = model.fit(**fit_setup['params'])

    available_data = {
        'model': model,
        'history': history
    }

    return {
        k: available_data[k]
        for k in fetch
    }

creator = ModelCreator(
    creator_method=creator_method,
    creator_name='HoboldNet2',
    default_params=dict(
        input_shape=[224, 224, 1],
        verbose=0,
        checkpoint={'restore': False},
        dropout_ratio=0.5,
        num_classes=3,
        problem='regression',
        fetch=['model', 'history'],
    ),
    expand_params=True
)