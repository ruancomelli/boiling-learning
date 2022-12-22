from boiling_learning.app.datasets.bridged.boiling1d import default_boiling_bridging_gt10
from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.evaluate import UncertainValue, evaluate_with_uncertainty
from boiling_learning.model.model import ModelArchitecture


def evaluate_boiling_model_with_dataset(
    model: LazyDescribed[ModelArchitecture],
    evaluation_dataset: LazyDescribed[ImageDatasetTriplet],
) -> DatasetTriplet[dict[str, UncertainValue[float]]]:
    ds_train, ds_val, ds_test = default_boiling_bridging_gt10(
        evaluation_dataset,
        batch_size=None,
    )

    train_metrics = evaluate_with_uncertainty(model(), ds_train())
    validation_metrics = evaluate_with_uncertainty(model(), ds_val())
    test_metrics = evaluate_with_uncertainty(model(), ds_test())

    return DatasetTriplet(train_metrics, validation_metrics, test_metrics)
