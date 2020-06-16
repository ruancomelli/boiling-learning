import enum

_sentinel = object()

class ProblemType(enum.Enum):
    CLASSIFICATION = enum.auto()
    REGRESSION = enum.auto()
    
    @classmethod
    def get_type(cls, s, default=_sentinel):
        if s in cls:
            return s
        else:
            return cls.from_string(s, default=default)

    @classmethod
    def from_string(cls, s, default=_sentinel):
        for k, v in cls.conversion_table.items():
            if s in v:
                return k
        if default is _sentinel:
            raise ValueError(f'string {s} was not found in the conversion table. Available values are {list(cls.conversion_table.values())}.')
        else:
            return default
 
ProblemType.conversion_table = {
    ProblemType.CLASSIFICATION: {'classification', 'regime'},
    ProblemType.REGRESSION: {'regression', 'heat flux', 'h', 'power'},
}

def default_compiler(model, **params):
    return model.compile(**params)
    
def default_fitter(model, **params):
    return model.fit(**params)

def make_creator_method(
	    builder,
		compiler=default_compiler,
		fitter=default_fitter
):
	def creator_method(
		verbose,
		checkpoint,
		num_classes,
		problem,
		architecture_setup,
		compile_setup,
		fit_setup,
		fetch,
	):    
		last_epoch, model = bl.model.restore(**checkpoint)
		initial_epoch = max(last_epoch, 0)
		
		if model is None:        
			model = builder(
				problem,
				num_classes,
				**architecture_setup
			)

			if compile_setup.get('do', False):
				compiler(model, **compile_setup['params'])
				# model.compile(**compile_setup['params'])

		history = None
		if fit_setup.get('do', False):
			fit_setup['params']['initial_epoch'] = initial_epoch
			history = fitter(model, **fit_setup['params'])
			# history = model.fit(**fit_setup['params'])

		available_data = {
			'model': model,
			'history': history
		}

		return {
			k: available_data[k]
			for k in fetch
		}

	return creator_method