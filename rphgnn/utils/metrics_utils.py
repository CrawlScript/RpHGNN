import torch
import numpy as np
import torchmetrics

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def ndcg_mrr(pred, labels):
    """
    Compute both NDCG and MRR for single-label and multi-label. Code extracted from
    https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/OAG/train_paper_venue.py#L316 (single label)
    and
    https://github.com/acbull/pyHGT/blob/f7c4be620242d8c1ab3055f918d4c082f5060e07/OAG/train_paper_field.py#L322 (multi-label)
    """
    test_res = []
    if len(labels.shape) == 1:
        # single-label
        for ai, bi in zip(labels, pred.argsort(descending = True)):
            test_res += [(bi == ai).int().tolist()]
    else:
        # multi-label
        for ai, bi in zip(labels, pred.argsort(descending = True)):
            test_res += [ai[bi].int().tolist()]
    ndcg = np.mean([ndcg_at_k(resi, len(resi)) for resi in test_res])
    mrr = mean_reciprocal_rank(test_res)
    return ndcg, mrr


###############################################################################
# Fast re-implementation of NDCG and MRR for a batch of nodes.
# We provide unit test below using random input to verify correctness /
# equivalence.
###############################################################################

def batched_dcg_at_k(r, k):
    assert(len(r.shape) == 2 and r.size != 0 and k > 0)
    r = r[:, :k].float()
    # Usually, one defines DCG = \sum\limits_{i=0}^{n-1}\frac{r_i}/{log2(i+2)}
    # So, we should
    # return (r / torch.log2(torch.arange(0, r.shape[1], device=r.device, dtype=r.dtype).view(1, -1) + 2)).sum(dim=1)
    # However, HGT author implements DCG = r_0 + \sum\limits_{i=1}^{n-1}\frac{r_i}/{log2(i+1)}, which makes DCG and NDCG larger
    # Here, we follow HGT author for a fair comparison
    return r[:, 0] + (r[:, 1:] / torch.log2(torch.arange(1, r.shape[1], device=r.device, dtype=r.dtype).view(1, -1) + 1)).sum(dim=1)


def batched_ndcg_at_k(r, k):
    dcg_max = batched_dcg_at_k(r.sort(dim=1, descending=True)[0], k)
    dcg_max_inv = 1.0 / dcg_max
    dcg_max_inv[torch.isinf(dcg_max_inv)] = 0
    return batched_dcg_at_k(r, k) * dcg_max_inv


def batched_mrr(r):
    r = r != 0
    # torch 1.5 does not guarantee max returns first occurrence
    # https://pytorch.org/docs/1.5.0/torch.html?highlight=max#torch.max
    # So we get first occurrence of non-zero using numpy max
    max_indices = torch.from_numpy(r.cpu().numpy().argmax(axis=1))
    max_values = r[torch.arange(r.shape[0]), max_indices]
    r = 1.0 / (max_indices.float() + 1)
    r[max_values == 0] = 0
    return r


# def batched_ndcg_mrr(pred, labels):
#     pred = pred.argsort(descending=True)
#     if len(labels.shape) == 1:
#         # single-label
#         labels = labels.view(-1, 1)
#         rel = (pred == labels).int()
#     else:
#         # multi-label
#         rel = torch.gather(labels, 1, pred)
#     return batched_ndcg_at_k(rel, rel.shape[1]), batched_mrr(rel)

def compute_batched_ndcg(pred, labels):
    pred = pred.argsort(descending=True)
    if len(labels.shape) == 1:
        # single-label
        labels = labels.view(-1, 1)
        rel = (pred == labels).int()
    else:
        # multi-label
        rel = torch.gather(labels, 1, pred)
    return batched_ndcg_at_k(rel, rel.shape[1])

def compute_batched_mrr(pred, labels):
    pred = pred.argsort(descending=True)
    if len(labels.shape) == 1:
        # single-label
        labels = labels.view(-1, 1)
        rel = (pred == labels).int()
    else:
        # multi-label
        rel = torch.gather(labels, 1, pred)
    return batched_mrr(rel)


class LogitsBasedMetric(torchmetrics.Metric):
    pass


class NDCG(LogitsBasedMetric):
    def __init__(self):
        super().__init__()
        self.add_state("ndcg_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        batch_ndcgs = compute_batched_ndcg(logits, target)
        self.ndcg_sum += batch_ndcgs.sum()
        # self.total += target.numel()
        self.total += target.size(0)

    def compute(self):
        return self.ndcg_sum / self.total.float()


class MRR(LogitsBasedMetric):
    def __init__(self):
        super().__init__()
        self.add_state("mrr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        batch_mrrs = compute_batched_mrr(logits, target)
        self.mrr_sum += batch_mrrs.sum()
        # self.total += target.numel()
        self.total += target.size(0)

    def compute(self):
        return self.mrr_sum / self.total.float()