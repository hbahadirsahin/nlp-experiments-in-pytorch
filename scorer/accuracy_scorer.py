import torch


class AccuracyScorer(object):
    def __init__(self, topk=(2,)):
        super(AccuracyScorer, self).__init__()
        self.topk = topk

    @staticmethod
    def calculate_accuracy(predictions, ground_truths):
        correct = (torch.max(predictions, 1)[1].view(ground_truths.size()).data == ground_truths.data)
        return float(correct.sum()) / len(correct) * 100

    def calculate_topk_accuracy(self, predictions, ground_truths):
        """
        Code copied/pasted from PyTorch Imagenet example: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        Computes the accuracy over the k top predictions for the specified values of k
        """
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = ground_truths.size(0)

            _, pred = predictions.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(ground_truths.view(1, -1).expand_as(pred))

            res = []
            for k in self.topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res
