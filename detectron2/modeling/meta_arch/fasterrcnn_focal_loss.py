from .build import META_ARCH_REGISTRY
from .rcnn import *
import logging
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

__all__ = ["FasterRCNNFocalLoss"]

@META_ARCH_REGISTRY.register()
class FasterRCNNFocalLoss(GeneralizedRCNN):

    def __init__(self, cfg):
        super().__init__(cfg)
        # Use prior in model initialization to improve stability
        # prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        # torch.nn.init.constant_(self.cls_score.bias, bias_value)

    # rewrite forward function of rcnn with a new loss
    def forward(self, batched_inputs):

        # print("you're in faster rcnn focal loss")
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        # print("lala")
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # print("you're in faster rcnn focal loss")
        return losses