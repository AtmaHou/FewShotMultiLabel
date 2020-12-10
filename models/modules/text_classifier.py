#!/usr/bin/env python
from typing import Tuple, Union, List, Dict
import torch
from torch import nn
from collections import Counter
from models.modules.similarity_scorer_base import GaussianKernelSimilarityScorer


class MultiLabelTextClassifier(torch.nn.Module):
    def __init__(self, threshold=0.6, grad_threshold=True):
        super(MultiLabelTextClassifier, self).__init__()
        self.threshold = nn.Parameter(torch.FloatTensor([threshold]), requires_grad=grad_threshold)
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.right_estimate = None

    def forward(self,
                logits: torch.Tensor,
                mask: torch.Tensor,
                tags: torch.Tensor) -> torch.Tensor:
        """
        :param logits: (batch_size, 1, n_tags)
        :param mask: (batch_size, 1)
        :param tags: (batch_size, max_label_num), eg [[2, 15], [2, 0]]
        :return:
        """
        return self._compute_loss(logits, mask, tags)

    def _compute_loss(self,
                      logits: torch.Tensor,
                      mask: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        """
        :param logits: (batch_size, 1, n_tags)
        :param mask: (batch_size, 1)
        :param targets: (batch_size, max_label_num), eg [[2, 15], [2, 0]]
        :return:
        """

        batch_size, seq_len = mask.shape
        # todo: test different normalization effectiveness.
        # todo: check pad label

        threshold = self.get_threshold(logits)
        # Normalization has been done in emission's scaler
        filtered_logits = logits - threshold  # For each pos, > 0 for positive tag, < 0 for negative tag
        multi_hot_target = self.create_multi_hot(targets, label_num=logits.shape[-1])

        loss = self.criterion(filtered_logits, multi_hot_target)
        return loss

    def create_multi_hot(self, y: torch.Tensor, label_num: int):
        """
        :param y:  (batch_size, max_label_num), eg [[2, 15], [2, 0]]
        :param label_num: num of
        :return:
        """
        batch_size = y.shape[0]
        y_one_hot = torch.zeros(batch_size, label_num).to(y.device)
        return y_one_hot.scatter(1, y, 1)

    def decode(self, logits: torch.Tensor) -> List[List[int]]:
        """ collect the values greater than threshold. """
        # shape: (batch_size, 1, no_pad_num_tag) -> (batch_size, 1, no_pad_num_tag)
        threshold = self.get_threshold(logits)
        preds = (logits - threshold).squeeze()
        ret = []
        for pred in preds:
            temp = []
            for l_id, score in enumerate(pred):
                if bool(score > 0):
                    temp.append(int(l_id))
            # predict the label with most probability
            if not temp:
                temp = [int(torch.argmax(pred))]
            ret.append(temp)

        return ret

    def get_threshold(self, logits):
        return self.threshold


class EAMultiLabelTextClassifier(MultiLabelTextClassifier):
    """ Emission adaptive MultiLabelTextClassifier
        (1) Adaptive threshold λ is calculated by observing emission scores as:
        λ =（E_max−E_min ）× r + E_min
        where E is emission, r is emission rate.

        (2) Then different sample has different threshold for different logits.
    """
    def __init__(self, threshold=0.6, grad_threshold=True):
        """
        Here self.threshold is used as emission rate.
        """
        super(EAMultiLabelTextClassifier, self).__init__(threshold, grad_threshold)

    def get_threshold(self, logits):
        """
        :param logits: (batch_size, 1, n_tags)
        :return: (batch_size, 1, 1)
        """
        max_logits = torch.max(logits, dim=-1)[0]  # fetch value, give up indexes.
        min_logits = torch.min(logits, dim=-1)[0]
        threshold = (max_logits - min_logits) * self.threshold + min_logits  # (batch_size, 1)
        return threshold.unsqueeze(-1)


class MetaStatsMultiLabelTextClassifier(EAMultiLabelTextClassifier):
    """ Meta statistic MultiLabelTextClassifier.
    """
    def __init__(self, threshold=0.6, grad_threshold=True, meta_rate=0.5, ab_ea=False):
        super(MetaStatsMultiLabelTextClassifier, self).__init__(threshold, grad_threshold)
        self.num_stats = None
        self.meta_rate = meta_rate
        self.ab_ea = ab_ea

    def update_statistics(self, support_targets):
        """
        Update stats for each sample in batch.
        :param support_targets: one-hot targets (batch_size, support_size, max_label_num, num_tags)
        :return: None
        """
        # count label num
        batch_size, support_size, max_label_num, num_tags = support_targets.shape
        c_sup_targets = support_targets.narrow(dim=-1, start=1, length=num_tags-1)
        multi_hot_tgt = torch.sum(c_sup_targets, dim=-2)  # shape (batch_size, support_size, num_tags)
        label_num = torch.sum(multi_hot_tgt, dim=-1)  # shape (batch_size, support_size)
        # get stats for each sample in batch
        label_num = [[it for it in item.tolist() if it] for item in label_num]  # del [PAD] label data
        self.num_stats = [Counter(n) for n in label_num]

    def get_threshold(self, logits):
        """
        Step1: Estimate threshold λ′ by observing support set:
                λ′= ∑_N^i[p(k=i) E_(i+1)]
        ,where N is label num, E_j is j_th largest ranked emission scores

        Step2: Calibrate λ′ with meta parameter

        :param logits: (batch_size, 1, n_tags)
        :return: (batch_size, 1, 1)
        """
        ''' Estimate threshold '''
        # get estimate thresholds: shape (batch_size)
        estimate_thresholds = self.estimate_threshold(logits)
        ''' Calibrate threshold '''
        thresholds = self.calibrate_threshold(logits, estimate_thresholds)
        return thresholds

    def estimate_threshold(self, logits) -> torch.FloatTensor:
        """
        :param logits: (batch_size, 1, n_tags)
        :return: shape (batch_size)
        """
        # todo: check support set pad influence of
        ret = []
        for ind, logit in enumerate(logits):
            sorted_logits = sorted(logit[0], reverse=True)
            stats = self.num_stats[ind]
            stats: Counter
            l_sum = 0
            for num, count in stats.items():
                l_sum += sorted_logits[int(num)] * count  # num is already rank + 1
            ret.append(l_sum / len(list(stats.elements())))
        ret = torch.stack(ret).to(logits.device)
        return ret

    def calibrate_threshold(self, logits, thresholds) -> torch.FloatTensor:
        """

        :param logits:
        :param thresholds:
        :return: (batch_size, 1, 1)
        """
        if self.ab_ea:
            meta_threshold = self.threshold  # ablation EA threshold here
        else:
            meta_threshold = super().get_threshold(logits)  # use EA threshold here
        est_threshold = thresholds.unsqueeze(-1).unsqueeze(-1)
        return est_threshold * (1 - self.meta_rate) + self.meta_rate * meta_threshold

    def get_ea_threshold(self, logits):
        return super().get_threshold(logits)


class KRNMetaStatsMultiLabelTextClassifier(MetaStatsMultiLabelTextClassifier):

    def __init__(self, threshold=0.6, grad_threshold=True, meta_rate=0.5, ab_ea=False, kernel='gaussian', bandwidth=0.5,
                 use_gold=False, learnable=False, map_dict=None):
        super(KRNMetaStatsMultiLabelTextClassifier, self).__init__(threshold, grad_threshold, meta_rate, ab_ea)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.learnable = learnable
        self.use_gold = use_gold
        self.map_dict = map_dict
        self.similarity_scorer = self.choose_kernel_similar()
        if self.learnable:
            self.label_num_criterion = nn.MSELoss()
            self.label_num_loss = None

    def choose_kernel_similar(self):
        if self.kernel == 'gaussian':
            similarity_scorer = GaussianKernelSimilarityScorer(bandwidth=self.bandwidth, learnable=self.learnable,
                                                               map_dict=self.map_dict)
        else:
            raise NotImplementedError
        return similarity_scorer

    def update_statistics(self,
                          support_targets,
                          support_sentence_feature=None,
                          test_sentence_feature=None,
                          support_sentence_label_num=None,
                          test_sentence_label_num=None):
        """
        Update stats for each sample in batch.
        :param support_targets: one-hot targets (batch_size, support_size, max_label_num, num_tags)
        :param support_sentence_feature: (batch_size, support_size, feature_len)
        :param test_sentence_feature: (batch_size, feature_len)
        :param support_sentence_label_num: (batch_size, support_size)
        :param test_sentence_label_num: (batch_size)
        :return: None
        """

        ''' count label num '''
        if self.use_gold:
            t_target = test_sentence_label_num.squeeze(-1)
            self.num_stats = [{item: [1]} for item in t_target.long().tolist()]
            self.right_estimate = (t_target == t_target).long()
        else:
            label_num_weights = self.similarity_scorer(support_sentence_feature, test_sentence_feature,
                                                       support_sentence_label_num, test_sentence_label_num)

            # get the distributed num stats
            batch_size = label_num_weights.size(0)
            self.num_stats = []
            for b_idx in range(batch_size):
                tmp_stat = {}
                for s_label_num, weight in zip(support_sentence_label_num[b_idx].long().tolist(),
                                               label_num_weights[b_idx].tolist()):
                    if s_label_num not in tmp_stat:
                        tmp_stat[s_label_num] = [weight]
                    else:
                        tmp_stat[s_label_num].append(weight)
                self.num_stats.append(tmp_stat)

            # calculate the label num accuracy
            pred_label_num = torch.sum(label_num_weights * support_sentence_label_num, dim=-1)
            pred_label_num_int = torch.round(pred_label_num)
            self.right_estimate = (pred_label_num_int == test_sentence_label_num.squeeze(-1)).long()

            if self.learnable:
                self.label_num_loss = self.label_num_criterion(pred_label_num, test_sentence_label_num.squeeze(-1))

    def estimate_threshold(self, logits) -> torch.FloatTensor:
        """
        :param logits: (batch_size, 1, n_tags)
        :return: shape (batch_size)
        """
        # todo: check support set pad influence of
        ret = []
        for ind, logit in enumerate(logits):
            sorted_logits = sorted(logit[0], reverse=True)
            stats = self.num_stats[ind]
            stats: Dict
            l_sum = 0
            for num, count_lst in stats.items():
                l_sum += sorted_logits[int(num)] * sum(count_lst)
            ret.append(l_sum)
        ret = torch.stack(ret).to(logits.device)
        return ret

    def _compute_loss(self,
                      logits: torch.Tensor,
                      mask: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        """
        :param logits: (batch_size, 1, n_tags)
        :param mask: (batch_size, 1)
        :param targets: (batch_size, max_label_num), eg [[2, 15], [2, 0]]
        :return:
        """
        loss = super()._compute_loss(logits, mask, targets)
        if self.learnable:
            loss += self.label_num_loss
        return loss

    def decode(self, logits: torch.Tensor, test_label_num=None) -> List[List[int]]:
        """ collect the values greater than threshold. """
        # shape: (batch_size, 1, no_pad_num_tag) -> (batch_size, 1, no_pad_num_tag)
        if self.use_gold:
            test_label_num = test_label_num.squeeze(-1).long().tolist()  # (batch_size, )
            ret = []
            for label_num, logit in zip(test_label_num, logits):
                sorted_logits = sorted(logit[0], reverse=True)
                threshold_logit = sorted_logits[label_num]
                threshold_logit = threshold_logit.unsqueeze(-1).expand_as(logit)
                pred = (logit - threshold_logit).squeeze()  # (batch_size, )
                temp = []
                for l_id, score in enumerate(pred):
                    if bool(score >= 0):
                        temp.append(int(l_id))
                # predict the label with most probability
                if not temp:
                    temp = [int(torch.argmax(pred))]
                ret.append(temp)
        else:
            ret = super().decode(logits)

        return ret


class SingleLabelTextClassifier(torch.nn.Module):

    def __init__(self):
        super(SingleLabelTextClassifier, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                logits: torch.Tensor,
                mask: torch.Tensor,
                tags: torch.Tensor) -> torch.Tensor:
        """
        :param logits: (batch_size, 1, n_tags)
        :param mask: (batch_size, 1)
        :param tags: (batch_size, 1)
        :return:
        """
        logits = logits.squeeze(-2)
        tags = tags.squeeze(-2)
        loss = self.criterion(logits, tags)
        return loss

    def decode(self, logits):
        ret = []
        for logit in logits:
            tmp = []
            for pred in logit:
                tmp.append(int(torch.argmax(pred)))
            ret.append(tmp)
        return ret

