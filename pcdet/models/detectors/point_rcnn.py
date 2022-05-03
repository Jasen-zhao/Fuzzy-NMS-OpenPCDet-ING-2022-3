from .detector3d_template import Detector3DTemplate
import time

class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # start = time.time()#查看运行时间
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # end = time.time()
        # print("model Running time: %s seconds"%(end - start))

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # start = time.time()#查看运行时间
            pred_dicts, recall_dicts ,nms_dicts = self.post_processing(batch_dict)
            # end = time.time()
            # print("post processing Running time: %s seconds"%(end - start))
            # return pred_dicts, recall_dicts#原始
            return pred_dicts, recall_dicts,nms_dicts#更改

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
