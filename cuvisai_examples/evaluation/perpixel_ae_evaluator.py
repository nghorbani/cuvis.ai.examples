from cuvisai_examples.registry import EVALUATORS

@EVALUATORS.register("PerPixelAEEvaluator")
class PerPixelAEEvaluator:
    def __init__(self):
        pass

    def evaluate(self, inputs):
        return {"ok": True}
