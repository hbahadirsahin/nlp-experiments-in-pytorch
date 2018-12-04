import torch
import torch.nn as nn

from utils.utils import log_sum_exp


class ConditionalRandomField(nn.Module):
    def __init__(self, tag_size, start_tag_id, end_tag_id, use_start_end=True):
        raise NotImplementedError()
        super(ConditionalRandomField, self).__init__()
        self.tag_size = tag_size
        self.use_start_end = use_start_end
        self.start_tag_id = start_tag_id
        self.end_tag_id = end_tag_id

        # Matrix of transition parameters.
        # Entry i,j is the score of  transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

        #
        if use_start_end:
            self.start_transitions = nn.Parameter(torch.Tensor(self.tag_size))
            self.end_transitions = nn.Parameter(torch.Tensor(self.tag_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.transitions)
        if self.use_start_end:
            nn.init.xavier_normal_(self.start_transitions)
            nn.init.xavier_normal_(self.end_transitions)

    def denominator_log_likelihood(self, inputs, mask):
        batch_size, seq_len, tag_size = inputs.size()

        mask = mask.float().transpose(0, 1)
        inputs = inputs.transpose(0, 1)

        if self.use_start_end:
            alpha = self.start_transitions.view(1, tag_size) + inputs[0]
        else:
            alpha = inputs[0]

        for i in range(1, seq_len):
            emition_scores = inputs[i].view(batch_size, 1, tag_size)
            transition_scores = self.transitions.view(1, tag_size, tag_size)
            forward_alphas = alpha.view(batch_size, tag_size, 1)

            next_tag_var = forward_alphas + emition_scores + transition_scores

            alpha = log_sum_exp(next_tag_var) * mask[i].view(batch_size, 1) + alpha * (1 - mask[i]).view(batch_size, 1)

        if self.use_start_end:
            stops = alpha + self.end_transitions.view(1, tag_size)
        else:
            stops = alpha

        return log_sum_exp(stops)

    def numerator_log_likelihood(self, inputs, tags, mask):
        batch_size, seq_len, tag_size = inputs.size()

        mask = mask.float().transpose(0, 1)
        inputs = inputs.transpose(0, 1)
        tags = tags.transpose(0, 1)

        if self.use_start_end:
            score = self.start_transitions[self.start_tag_id]
        else:
            score = torch.zeros(1)

        for i in range(seq_len - 1):
            emition_score = inputs[i].gather(1, tags[i].view(batch_size, 1)).squeeze(1)
            transition_score = self.transitions[tags[i].view(-1), tags[i + 1].view(-1)]
            score = score + transition_score * mask[i + 1] + emition_score * mask[i]

        last_tag_indices = mask.long().sum(0) - 1
        last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)

        if self.use_start_end:
            last_tag_to_end_tag_transition_score = self.end_transitions[last_tags]
        else:
            last_tag_to_end_tag_transition_score = torch.zeros(1)

        last_inputs = inputs[-1]
        last_emition_score = last_inputs.gather(1, last_tags.view(-1, 1)).squeeze()

        score = score + last_tag_to_end_tag_transition_score + last_emition_score * mask[-1]

        return score

    def forward(self, input, tags, mask=None, use_sum=False):
        if mask is None:
            mask = torch.ones(tags.size())

        loglikelihood = self.denominator_log_likelihood(input, mask) - self.numerator_log_likelihood(input, tags, mask)

        if use_sum:
            return torch.sum(loglikelihood)
        else:
            return loglikelihood

    def viterbi_decode(self, inputs, mask):
        batch_size, seq_length, tag_size = inputs.size()

        scores = []
        backpointers = []
