"""Kedro project settings.

Kept intentionally minimal: the ML logic lives in ``betting_app.ml.training``
and Kedro is only an offline orchestrator.
"""

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}
