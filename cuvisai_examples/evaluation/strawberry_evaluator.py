from cuvisai_examples.registry import EVALUATORS


@EVALUATORS.register("StrawberryEvaluator")
class StrawberryEvaluator:
    def __init__(self):
        pass

    def evaluate(self, inputs):
        return {"ok": True}
