import logging.config

from evaluation.interactive_evaluator import InteractiveEvaluator
from evaluation.single_model_ner_evaluator import SingleModelNerEvaluator
from evaluation.single_model_evaluator import SingleModelEvaluator

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Evaluator")


class Evaluator(object):
    @staticmethod
    def evaluator_factory(type, device):
        if type == "single_model_evaluator":
            logger.info("Evaluator type is %s", type)
            dev_evaluator = SingleModelEvaluator(device, is_vali=True)
            test_evaluator = SingleModelEvaluator(device, is_vali=False)
            return dev_evaluator, test_evaluator
        elif type == "single_model_ner_evaluator":
            logger.info("Evaluator type is %s", type)
            dev_evaluator = SingleModelNerEvaluator(device, is_vali=True)
            test_evaluator = SingleModelNerEvaluator(device, is_vali=False)
            return dev_evaluator, test_evaluator
        elif type == "interactive_evaluator":
            return InteractiveEvaluator(device)
        else:
            ValueError("Unrecognized trainer type")
