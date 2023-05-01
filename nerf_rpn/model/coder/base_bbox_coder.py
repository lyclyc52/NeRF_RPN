from abc import ABCMeta, abstractmethod
from typing import List
import torch
from torch import Tensor

from ..utils import print_shape

class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder"""

    def __init__(self, **kwargs):
        pass

    def encode(self, bboxes_ref: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        boxes_per_image = [len(b) for b in bboxes_ref]
        bboxes_ref = torch.cat(bboxes_ref, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(bboxes_ref, proposals)
        return targets.split(boxes_per_image, 0)
    
    def decode(self, bboxes_ref: Tensor, deltas: List[Tensor]) -> Tensor:
        torch._assert(
            isinstance(deltas, (list, tuple)),
            "This function expects deltas of type list or tuple.",
        )
        torch._assert(
            isinstance(bboxes_ref, torch.Tensor),
            "This function expects bboxes_ref of type torch.Tensor.",
        )
        boxes_per_image = [b.size(0) for b in deltas]
        concat_deltas = torch.cat(deltas, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            bboxes_ref = bboxes_ref.reshape(box_sum, -1)
        pred_boxes = self.decode_single(bboxes_ref, concat_deltas)
        if box_sum > 0:
            # pred_boxes = pred_boxes.reshape(box_sum, -1, 6) # what is this doing?
            pred_boxes = pred_boxes[:, None, :]
        return pred_boxes

    @abstractmethod
    def encode_single(self, bboxes: Tensor, proposals: Tensor) -> Tensor:
        """Get deltas between bboxes and ground truth boxes"""
        pass

    @abstractmethod
    def decode_single(self, deltas: Tensor, proposals: Tensor) -> Tensor:
        """
        Decode the predicted bboxes according to deltas_prediction and base boxes
        """
        pass
    
    def decode_list(self, delta_list: List[Tensor], boxes_list: List[List[Tensor]], is_cat = True) -> List:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            delta (List): encoded boxes
            boxes (List): reference boxes.
        """
        pred_boxes_list = []
        for i in range(len(delta_list)):
            cur_pred_boxes_list = []
            cur_rel_code = delta_list[i]
            for j in range(cur_rel_code.size(0)):
                if isinstance(boxes_list[j], list):
                    boxes = boxes_list[j][i].detach()
                else:    
                    boxes = boxes_list[j].detach()
                delta = cur_rel_code[j].detach()
                pred_boxes = self.decode_single(delta, boxes)
                level_index= pred_boxes.new_tensor([i])
                level_index = level_index.repeat(pred_boxes.size(0), 1)

                pred_boxes = torch.cat([pred_boxes, level_index], dim = 1).flatten(1)
                cur_pred_boxes_list.append(pred_boxes)
            pred_boxes_list.append(torch.stack(cur_pred_boxes_list))
            

        if is_cat:
            return torch.cat(pred_boxes_list, dim = 1)
        else:
            return pred_boxes_list