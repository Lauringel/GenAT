import numpy as np

### different usage in different files, we have here the version from MS_VQGAN_Param_tune_1, which does not compute the mean and returns F1, IoU, precision, recall, acc
def compute_metrics(pred_bin, gt_bin):
    if isinstance(pred_bin, np.ndarray):
        pred = pred_bin.astype(bool)
    else:
        pred = pred_bin.detach().cpu().numpy().astype(bool)

    gt = gt_bin.detach().cpu().numpy().astype(bool)

    TP = np.logical_and(pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    IoU = TP / (TP + FP + FN + 1e-8)
    F1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return F1, IoU, precision, recall, acc
