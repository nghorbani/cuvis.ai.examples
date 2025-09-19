from cuvisai_examples.registry import EVALUATORS


@EVALUATORS.register("DefaultEvaluator")
class DefaultEvaluator:
    def __init__(self):
        pass

    def evaluate(self, inputs):
        return {"ok": True}
