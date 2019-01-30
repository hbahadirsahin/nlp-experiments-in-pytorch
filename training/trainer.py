import logging.config

from training.single_model_trainer import SingleModelTrainer
from training.single_model_ner_trainer import SingleModelNerTrainer

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Trainer")


class Trainer(object):
    @staticmethod
    def trainer_factory(type, training_properties, train_iter, dev_iter, test_iter, device):
        if type == "single_model_trainer":
            logger.info("Trainer type is %s", type)
            return SingleModelTrainer(training_properties, train_iter, dev_iter, test_iter, device)
        elif type == "single_model_ner_trainer":
            logger.info("Trainer type is %s", type)
            return SingleModelNerTrainer(training_properties, train_iter, dev_iter, test_iter, device)
        else:
            ValueError("Unrecognized trainer type")
