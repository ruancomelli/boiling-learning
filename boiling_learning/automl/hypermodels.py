from __future__ import annotations

import typing
from typing import Any, Dict

import autokeras as ak

from boiling_learning.io import json
from boiling_learning.model.model import anonymize_model_json


class HyperModel:
    def __init__(self, automodel: ak.AutoModel) -> None:
        self.automodel = automodel

    def get_config(self) -> Dict[str, Any]:
        return self.automodel.tuner.hypermodel.get_config()

    def __json_encode__(self) -> Dict[str, Any]:
        model_json = self.get_config()
        return anonymize_model_json(
            {key: value for key, value in model_json['config'].items() if key != 'name'}
        )

    def __describe__(self) -> Dict[str, Any]:
        return typing.cast(Dict[str, Any], json.encode(self))
