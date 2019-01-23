import torch
import torch.nn as nn

from utils.utils import log_sum_exp


class ConditionalRandomField(nn.Module):
    def __init__(self, args):
        super(ConditionalRandomField, self).__init__()

        self.args_common = args["common_model_properties"]

        self.tag_size = self.args_common["num_tags"]
        self.start_id = self.args_common["start_id"]
        self.end_id = self.args_common["end_id"]
        self.pad_id = self.args_common["crf_padding_id"]

        # Matrix of transition parameters. Entry i,j is the score of transitioning *to* i *from* j
        self.transition = nn.Parameter(torch.Tensor(self.tag_size, self.tag_size))

        self.transition.data[self.start_id, :] = -10000.  # no transition to SOS
        self.transition.data[:, self.end_id] = -10000.  # no transition from EOS except to PAD
        self.transition.data[:, self.pad_id] = -10000.  # no transition from PAD except to PAD
        self.transition.data[self.pad_id, :] = -10000.  # no transition to PAD except from EOS
        self.transition.data[self.pad_id, self.end_id] = 0.
        self.transition.data[self.pad_id, self.pad_id] = 0.

        torch.nn.init.xavier_normal_(self.transition)

    def _forward(self, x, mask):
        # initialize forward variables in log space
        batch_size, seq_length = x.size()

        # Size of init_alphas = [Batch_size, Tag_size]
        alpha = torch.full((batch_size, self.tag_size), -10000.)
        alpha[:, self.start_id] = 0.

        # Size of transition_scores = [1, Tag_size, Tag_size]
        transition_scores = self.transition.unsqueeze(0)

        for i in range(seq_length):
            mask_broadcast = mask[:, i].unsqueeze(1)

            # Size of emition_scores = [Batch_size, Tag_Size, 1]
            emition_scores = x[:, i].unsqueeze(2)

            # Size of alpha_broadcast: [Batch_Size, Tag_Size, Tag_Size]
            alpha_broadcast = log_sum_exp(alpha.unsqueeze(1) + emition_scores + transition_scores)

            # Size of alpha: [Batch_size, Tag_Size]
            alpha = alpha_broadcast * mask_broadcast + alpha * (1 - mask_broadcast)
        return log_sum_exp(alpha + self.transition[self.end_id])

    def _score(self, x, tags, mask):
        batch_size, seq_length = x.size()

        score = torch.zeros(batch_size)

        x = x.unsqueeze(3)
        trans = self.transition.unsqueeze(2)

        for t in range(seq_length):  # recursion through the sequence
            mask_broadcast = mask[:, t]
            emition_scores = torch.cat([x[t, y[t + 1]] for x, y in zip(x, y)])
            transition_scores = torch.cat([trans[y[t + 1], y[t]] for y in y])
            score += (emition_scores + transition_scores) * mask_broadcast

        last_tag = tags.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.transition[self.end_id, last_tag]
        return score

    def forward(self, input, tags, mask):
        forward_score = self._forward(input, mask)
        gold_score = self._score(input, tags, mask)
        return forward_score - gold_score

    def _viterbi_decode(self, x, mask):
        batch_size, seq_length = x.size()

        backpointers = torch.Tensor()
        # Initialize the viterbi variables in log space
        path_score = torch.full((batch_size, self.tag_size), -10000.)
        path_score[:, self.start_id] = 0.

        for next_tag in range(seq_length):
            mask_broadcast = mask[:, next_tag].unsqueeze(1)
            path_score_broadcast = path_score + self.transition
            path_score_broadcast, backpointers_broadcast = torch.max(path_score_broadcast, 2)
            path_score_broadcast += x[:, next_tag]
            path_score = path_score_broadcast * mask_broadcast + path_score * (1 - mask_broadcast)
            backpointers = torch.cat((backpointers, backpointers_broadcast.unsqueeze(1)), 1)

        path_score += self.transition[self.end_id]
        best_path_scores, best_tag_ids = torch.max(path_score, 1)

        backpointers = backpointers.tolist()
        best_paths = [[tag_id] for tag_id in best_tag_ids.tolist()]

        for batch in range(batch_size):
            best_tag = best_tag_ids[batch]
            idx = int(mask[batch].sum().item())
            for bptr_t in reversed(backpointers[batch][:idx]):
                best_tag = bptr_t[best_tag]
                best_paths[batch].append(best_tag)
            best_paths[batch].pop()
            best_paths[batch].reverse()

        return best_path_scores, best_paths
