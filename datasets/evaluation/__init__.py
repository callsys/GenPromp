
from typing import Any
import numpy as np
from .cam import evaluate_cls_loc, list2acc

# Evaluation code from TS-CAM
class Evaluator():
    def __init__(self, logfile='log.txt', len_dataloader=1):
        self.cls_top1 = []
        self.cls_top5 = []
        self.loc_top1 = []
        self.loc_top5 = []
        self.loc_gt_known = []
        self.top1_loc_right = []
        self.top1_loc_cls = []
        self.top1_loc_mins = []
        self.top1_loc_part = []
        self.top1_loc_more = []
        self.top1_loc_wrong = []
        self.logfile = logfile
        self.len_dataloader = len_dataloader

    def __call__(self, input, target, bbox, logits, pad_cams, image_names, cfg, step):
        cls_top1_b, cls_top5_b, loc_top1_b, loc_top5_b, loc_gt_known_b, top1_loc_right_b, \
        top1_loc_cls_b, top1_loc_mins_b, top1_loc_part_b, top1_loc_more_b, top1_loc_wrong_b = \
        evaluate_cls_loc(input, target, bbox, logits, pad_cams, image_names, cfg)
        self.cls_top1.extend(cls_top1_b)
        self.cls_top5.extend(cls_top5_b)
        self.loc_top1.extend(loc_top1_b)
        self.loc_top5.extend(loc_top5_b)
        self.top1_loc_right.extend(top1_loc_right_b)
        self.top1_loc_cls.extend(top1_loc_cls_b)
        self.top1_loc_mins.extend(top1_loc_mins_b)
        self.top1_loc_more.extend(top1_loc_more_b)
        self.top1_loc_part.extend(top1_loc_part_b)
        self.top1_loc_wrong.extend(top1_loc_wrong_b)

        self.loc_gt_known.extend(loc_gt_known_b)

        if step != 0 and (step % 100 == 0 or step == self.len_dataloader - 1):
            str1 = 'Val Epoch: [{0}][{1}/{2}]\t'.format(0, step + 1, self.len_dataloader)
            str2 = 'Cls@1:{0:.3f}\tCls@5:{1:.3f}\tLoc@1:{2:.3f}\tLoc@5:{3:.3f}\tLoc_gt:{4:.3f}'.format(
                list2acc(self.cls_top1), list2acc(self.cls_top5),list2acc(self.loc_top1), list2acc(self.loc_top5), list2acc(self.loc_gt_known))
            str3 = 'M-ins:{0:.3f}\tPart:{1:.3f}\tMore:{2:.3f}\tRight:{3:.3f}\tWrong:{4:.3f}\tCls:{5:.3f}'.format(
                list2acc(self.top1_loc_mins), list2acc(self.top1_loc_part), list2acc(self.top1_loc_more),
                list2acc(self.top1_loc_right), list2acc(self.top1_loc_wrong), list2acc(self.top1_loc_cls))

            if self.logfile is not None:
                with open(self.logfile, 'a') as fw:
                    fw.write('\n'+str1+'\n')
                    fw.write(str2+'\n')
                    fw.write(str3+'\n')

            print(str1)
            print(str2)
            print(str3)

