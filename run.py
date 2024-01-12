import os
import os.path as osp
import torch

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer

@torch.no_grad()
def run(weights='yolov6l.pt',
        source='data/images',
        webcam=True,
        webcam_addr='0',
        use_depth_cam=True,
        yaml='data/coco.yaml',
        img_size=[640, 640],
        conf_thres=0.4,
        iou_thres=0.45,
        max_det=1000,
        device='0',
        save_txt=False,
        not_save_img=False,
        save_dir=None,
        view_img=True,
        classes=None,
        agnostic_nms=False,
        project='runs/inference',
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False):
    """
    Simplified YOLOv6 inference function.
    """

    # Set default values for weights and use_depth_cam
    weights = osp.join(os.getcwd(), weights)

    # Create save directory
    if save_dir is None:
        save_dir = osp.join(project, name)
    if (not not_save_img or save_txt) and not osp.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize inference
    inferer = Inferer(source, webcam, webcam_addr, weights, device, yaml, img_size, half)
    inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, not not_save_img,
                  hide_labels, hide_conf, view_img)

    # Log results
    if save_txt or not not_save_img:
        LOGGER.info(f"Results saved to {save_dir}")

if __name__ == "__main__":
    run()
