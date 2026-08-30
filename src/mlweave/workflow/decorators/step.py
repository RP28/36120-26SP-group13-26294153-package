from mlweave.workflow.decorators.base import WorkflowFinalizingDecorator

class ModelSelectionDecorator(WorkflowFinalizingDecorator):
    """Declare a function as an sklearn-compatible search refit policy.

    The decorated function receives ``cv_results_`` and must return the integer
    candidate index that sklearn should refit.
    """

    mode = "model_selection"

class InferenceStepDecorator(WorkflowFinalizingDecorator):
    """Declare a function as the business-output stage of an MLWorkflow.

    The decorated function receives a ``WorkflowContext`` and may return any
    object or perform side effects such as writing a submission file or
    producing evaluation plots.
    """

    mode = "inference"

model_selection = ModelSelectionDecorator()
inference_step = InferenceStepDecorator()
