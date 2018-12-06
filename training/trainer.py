from training.single_model_trainer import SingleModelTrainer


class Trainer(object):
    @staticmethod
    def trainer_factory(type, training_properties, train_iter, dev_iter, test_iter, device):
        if type == "single_model_trainer":
            print("Trainer type is", type)
            return SingleModelTrainer(training_properties, train_iter, dev_iter, test_iter, device)
        elif type == "encoder_decoder_trainer":
            return
        elif type == "conv_deconv_trainer":
            return
        else:
            ValueError("Unrecognized trainer type")
