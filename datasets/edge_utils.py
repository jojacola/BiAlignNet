
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import cv2

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[_mask != 0] += 1
    return _mask


def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def new_one_hot_converter(a, num_classes=19):
    ncols = num_classes+1
    out = np.zeros( (a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out

def label_to_binary_edges(mask, num_classes, linewidth, ignore_id=255):
    label = np.array(mask)
    label[label==ignore_id] = num_classes
    size = mask.shape
    ll = new_one_hot_converter(label, num_classes)
    c = []
    for i in range(num_classes):
        img = ll[:, :, i].copy()
        contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        c.extend(contours)

    backtorgb = cv2.cvtColor(np.zeros(size, np.float32),cv2.COLOR_GRAY2RGB)
    cv2.drawContours(backtorgb,c,-1,(0,0,255),linewidth)

    b, g, r = cv2.split(backtorgb)
    ret, thresh1 = cv2.threshold(r,127,255,cv2.THRESH_BINARY)
    linewidth += 1
    thresh1[:, -linewidth:] = 0
    thresh1[:, :linewidth] = 0
    thresh1[-linewidth:, :] = 0
    thresh1[:linewidth, :] = 0


    edgemap = (thresh1 > 0).astype(np.uint8)
    edgemap = np.expand_dims(edgemap, axis=0)
    return edgemap
