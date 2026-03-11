"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    a_binary = (a > 0).astype(int)
    b_binary = (b > 0).astype(int)

    intersection = np.sum(a_binary * b_binary)
    volumes = np.sum(a_binary) + np.sum(b_binary)

    if volumes == 0:
        return -1

    return 2.0 * intersection / volumes

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    a_binary = (a > 0).astype(int)
    b_binary = (b > 0).astype(int)

    intersection = np.sum(a_binary * b_binary)
    union = np.sum(a_binary) + np.sum(b_binary) - intersection

    if union == 0:
        return -1

    return intersection / union


def Sensitivity3d(a, b):
    """
    Computes sensitivity (true positive rate / recall) for two 3D binary volumes.
    Sensitivity measures the proportion of actual positives correctly identified.
    Low sensitivity means under-segmentation (missing parts of the structure).

    Arguments:
        a {Numpy array} -- 3D predicted array
        b {Numpy array} -- 3D ground truth array

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    a_binary = (a > 0).astype(int)
    b_binary = (b > 0).astype(int)

    tp = np.sum(a_binary * b_binary)
    fn = np.sum((1 - a_binary) * b_binary)

    if (tp + fn) == 0:
        return -1

    return tp / (tp + fn)


def Specificity3d(a, b):
    """
    Computes specificity (true negative rate) for two 3D binary volumes.
    Specificity measures the proportion of actual negatives correctly identified.
    Low specificity means over-segmentation (labeling background as structure).

    Arguments:
        a {Numpy array} -- 3D predicted array
        b {Numpy array} -- 3D ground truth array

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    a_binary = (a > 0).astype(int)
    b_binary = (b > 0).astype(int)

    tn = np.sum((1 - a_binary) * (1 - b_binary))
    fp = np.sum(a_binary * (1 - b_binary))

    if (tn + fp) == 0:
        return -1

    return tn / (tn + fp)


def DicePerClass(pred, gt, class_id):
    """
    Computes Dice coefficient for a specific class in multi-class segmentation.

    Arguments:
        pred {Numpy array} -- 3D predicted label array
        gt {Numpy array} -- 3D ground truth label array
        class_id {int} -- class label to compute Dice for

    Returns:
        float
    """
    pred_binary = (pred == class_id).astype(int)
    gt_binary = (gt == class_id).astype(int)

    intersection = np.sum(pred_binary * gt_binary)
    volumes = np.sum(pred_binary) + np.sum(gt_binary)

    if volumes == 0:
        return -1

    return 2.0 * intersection / volumes