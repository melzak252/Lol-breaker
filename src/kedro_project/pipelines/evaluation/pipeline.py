"""Kedro pipeline definition for historical evaluation."""

from kedro.pipeline import Pipeline, node

from kedro_project.pipelines.evaluation.nodes import run_historical_evaluation


def create_pipeline(**kwargs) -> Pipeline:
    """Create the historical evaluation pipeline.

    Single-node pipeline that wraps the existing plain-Python
    ``run_evaluation_pipeline`` function.  Parameters come from
    ``conf/base/parameters.yml`` under the ``evaluation`` key.
    """
    return Pipeline(
        [
            node(
                func=run_historical_evaluation,
                inputs="params:evaluation",
                outputs="evaluation_result",
                name="run_historical_evaluation",
            ),
        ]
    )