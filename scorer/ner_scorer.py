

class NerScorer(object):
    def __init__(self):
        super(NerScorer, self).__init__()

    @staticmethod
    def token_level_accuracy(prediction, ground_truth):
        token_count = 0
        matched = 0

        for p_seq, gt_seq in zip(prediction, ground_truth):
            for p, gt in zip(p_seq, gt_seq):
                token_count += 1
                if p == gt:
                    matched += 1

        return matched * 100.0 / token_count
