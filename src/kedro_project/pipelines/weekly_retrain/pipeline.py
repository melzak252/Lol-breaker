"""Kedro pipeline definition for weekly retraining."""

from kedro.pipeline import Pipeline, node

from kedro_project.pipelines.weekly_retrain.nodes import run_weekly_retrain


def create_pipeline(**kwargs) -> Pipeline:
    """Create the weekly retraining pipeline.

    Single-node pipeline that wraps the existing plain-Python
    ``run_weekly_retrain_pipeline`` function.  Parameters come from
    ``conf/base/parameters.yml`` under the ``weekly_retrain`` key.
    """
    return Pipeline(
        [
            node(
                func=run_weekly_retrain,
                inputs="params:weekly_retrain",
                outputs="weekly_retrain_result",
                name="run_weekly_retrain",
            ),
        ]
    )