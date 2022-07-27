import tensorflow as tf


def calculate_stats(ds: tf.data.Dataset, metric: tf.keras.metrics.Metric) -> float:
    """Calculate metric (or statistic) over dataset.

    Args:
        ds (tf.data.Dataset): dataset yielding batches or single elements
        metric (tf.keras.metrics.Metric): metric or statistic to be measured

    Returns:
        float: metric value
    """
    metric.reset_state()

    for batch in ds:
        metric.update_state(batch)

    return float(metric.result().numpy())


def calculate_metric(ds: tf.data.Dataset, metric: tf.keras.metrics.Metric) -> float:
    """Calculate metric (error, accuracy etc.) between true and predicted values in a dataset.

    Args:
        ds (tf.data.Dataset): dataset yielding `(y_true, y_pred)` pairs where `y_true` is the
            reference value and `y_pred` is the predicted value. Both `y_true` and `y_pred` may be
            either single or batched values
        metric (tf.keras.metrics.Metric): metric (error, accuracy etc.) to be calculated

    Returns:
        float: calculated metric
    """
    metric.reset_state()

    for y_true, y_pred in ds:
        metric.update_state(y_true, y_pred)

    return float(metric.result().numpy())
