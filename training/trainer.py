import logging.config

from training.single_model_trainer import SingleModelTrainer

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Trainer")


class Trainer(object):
    @staticmethod
    def trainer_factory(type, training_properties, train_iter, dev_iter, test_iter, device):
        if type == "single_model_trainer":
            logger.info("Trainer type is %s", type)
            return SingleModelTrainer(training_properties, train_iter, dev_iter, test_iter, device)
        elif type == "encoder_decoder_trainer":
            return
        elif type == "multiple_model_trainer":
            return
        else:
            ValueError("Unrecognized trainer type")
