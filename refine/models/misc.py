import torch
import numpy as np

def nms(dets, scores, thres=0.4):
    """Non Maximum Suppression.

    Args:
        dets: (tensor) bounding boxes, sized [N, 4].
        scores: (tensor) confidence scores, sized [N].
        thres: (float) overlap threshold.

    Returns:
        keep: (tensor) selected indices.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 1:
        if order.numel() == 1:
            i = order.item()
        else:
            i = order[0].item()
        keep.append(i)
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (ovr <= thres).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)

def soft_nms(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0): # original scale
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].long()

    return keep

def diou_nms(boxes, scores, iou_thres=0.5):
    if boxes.shape[0] == 0:
        return torch.zeros(0,device=boxes.device).long()
    x1,y1,x2,y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = torch.sort(scores, descending=True)[1] #(?,)
    keep =[]
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            i = order[0].item()
            keep.append(i)

            xmin = torch.clamp(x1[order[1:]], min = float(x1[i]))
            ymin = torch.clamp(y1[order[1:]], min = float(y1[i]))
            xmax = torch.clamp(x2[order[1:]], max = float(x2[i]))
            ymax = torch.clamp(y2[order[1:]], max = float(y2[i]))

            inter_area = torch.clamp(xmax - xmin, min=0.0) * torch.clamp(ymax - ymin, min=0.0)

            iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-16)

            # diou add center
            # inter_diag
            cxpreds = (x2[i] + x1[i]) / 2
            cypreds = (y2[i] + y1[i]) / 2

            cxbbox = (x2[order[1:]] + x1[order[1:]]) / 2
            cybbox = (y1[order[1:]] + y2[order[1:]]) / 2

            inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

            # outer_diag
            ox1 = torch.min(x1[order[1:]], x1[i])
            oy1 = torch.min(y1[order[1:]], y1[i])
            ox2 = torch.max(x2[order[1:]], x2[i])
            oy2 = torch.max(y2[order[1:]], y2[i])

            outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

            diou = iou - inter_diag / outer_diag
            diou = torch.clamp(diou, min=-1.0, max=1.0)


            # mask_ind = (iou <= iou_thres).nonzero().squeeze()
            mask_ind = (diou <= iou_thres).nonzero().squeeze()

            if mask_ind.numel() == 0:
                break
            order = order[mask_ind + 1]
    return torch.LongTensor(keep)

# def wbf()