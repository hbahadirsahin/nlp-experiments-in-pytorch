from evaluation.interactive_evaluator import InteractiveEvaluator
from evaluation.single_model_evaluator import SingleModelEvaluator


class Evaluator(object):
    @staticmethod
    def evaluator_factory(type, device):
        if type == "single_model_evaluator":
            print("Evaluator type is", type)
            dev_evaluator = SingleModelEvaluator(device, is_vali=True)
            test_evaluator = SingleModelEvaluator(device, is_vali=False)
            return dev_evaluator, test_evaluator
        elif type == "encoder_decoder_evaluator":
            return
        elif type == "multiple_model_evaluator":
            return
        elif type == "interactive_evaluator":
            return InteractiveEvaluator(device)
        else:
            ValueError("Unrecognized trainer type")
