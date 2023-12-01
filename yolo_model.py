# Third Party
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import numpy as np

# In House
from yolo_utils import voting_suppression, bbox_convert_r, bbox_convert, label_to_box_xyxy, IoU

class F110Lightning(pl.LightningModule):
    def __init__(self, lr, anchor_size, input_dim, final_dim):
        super().__init__()
        self.yolo_model = F110_YOLO()
        self.lr = lr
        self.anchor_size = anchor_size
        self.input_dim = input_dim
        self.final_dim = final_dim
        self.voting_iou_threshold = 0.5
        self.confi_threshold = 0.5

    def training_step(self, batch, batch_idx):
        # Extract batch
        image_t, label_t, _ = batch 
        result = self.yolo_model(image_t)
        loss = self.yolo_model.get_loss(result, label_t)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Extract batch
        image, label_gt, label = batch
        result = self.yolo_model(image)
        loss = self.yolo_model.get_loss(result, label_gt)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Accuracy
        object_in_class = 0
        truth_in_class = 0
        result = result.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for i in range(label.shape[0]):
            bboxs, result_prob = label_to_box_xyxy(result[i], self.input_dim, self.final_dim, self.anchor_size, self.confi_threshold)
            vote_rank = voting_suppression(bboxs, self.voting_iou_threshold)
            if len(vote_rank) > 0: 
                bbox = bboxs[vote_rank[0]]
                [c_x, c_y, w, h] = bbox_convert_r(bbox[0], bbox[1], bbox[2], bbox[3])
                bboxs_2 = np.array([[c_x, c_y, w, h]])
                pos_change = np.sqrt((label[i, 0] - c_x) ** 2 + (label[i, 1] - c_y) ** 2)
                x_l, y_l, x_r, y_r = bbox_convert(label[i][0], label[i][1], label[i][2], label[i][3])
                label_xxyy = [x_l, y_l, x_r, y_r]
                if pos_change < 20 and IoU(bbox, label_xxyy) > 0.5:
                    object_in_class += 1
            truth_in_class += 1

        # Calc accuracy
        accuracy = object_in_class / truth_in_class
        self.log("val_accuracy", accuracy, on_epoch=True)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.yolo_model.parameters(), lr=self.lr)

class F110_YOLO(torch.nn.Module):
    def __init__(self):
        super(F110_YOLO, self).__init__()
        conv1_num_filters = 192
        self.conv1 = nn.Conv2d(3, conv1_num_filters, kernel_size = 7, padding = 1, stride = 2)
        self.batchnorm1 = nn.BatchNorm2d(conv1_num_filters)
        self.relu1 = nn.ReLU(inplace = True)

        conv2_num_filters = 256
        self.conv2 = nn.Conv2d(conv1_num_filters, conv2_num_filters, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm2 = nn.BatchNorm2d(conv2_num_filters)
        self.relu2 = nn.ReLU(inplace = True)

        conv3_num_filters = 512
        self.conv3 = nn.Conv2d(conv2_num_filters, conv3_num_filters, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm3 = nn.BatchNorm2d(conv3_num_filters)
        self.relu3 = nn.ReLU(inplace = True)

        conv4_num_filters = 1024
        self.conv4 = nn.Conv2d(conv3_num_filters, conv4_num_filters, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm4 = nn.BatchNorm2d(conv4_num_filters)
        self.relu4 = nn.ReLU(inplace = True)

        conv5_num_filters = 1024
        self.conv5 = nn.Conv2d(conv4_num_filters, conv5_num_filters, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm5 = nn.BatchNorm2d(conv5_num_filters)
        self.relu5 = nn.ReLU(inplace = True)

        conv6_num_filters = 1024
        self.conv6 = nn.Conv2d(conv5_num_filters, conv6_num_filters, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm6 = nn.BatchNorm2d(conv6_num_filters)
        self.relu6 = nn.ReLU(inplace = True)

        conv7_num_filters = 1024
        self.conv7 = nn.ConvTranspose2d(conv6_num_filters, conv7_num_filters, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm7 = nn.BatchNorm2d(conv7_num_filters)
        self.relu7 = nn.ReLU(inplace = True)

        conv8_num_filters = 1024
        self.conv8 = nn.ConvTranspose2d(conv7_num_filters, conv8_num_filters, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm8 = nn.BatchNorm2d(conv8_num_filters)
        self.relu8 = nn.ReLU(inplace = True)

        self.conv9 = nn.Conv2d(conv8_num_filters, 5, kernel_size = 1, padding = 0, stride = 1)
        self.relu9 = nn.ReLU()
    
    def forward(self, x):
        debug = 0 # change this to 1 if you want to check network dimensions
        if debug == 1: print(0, x.shape)
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        if debug == 1: print(1, x.shape)
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        if debug == 1: print(2, x.shape)
        x = torch.relu(self.batchnorm3(self.conv3(x)))
        if debug == 1: print(3, x.shape)
        x = torch.relu(self.batchnorm4(self.conv4(x)))
        if debug == 1: print(4, x.shape)
        x = torch.relu(self.batchnorm5(self.conv5(x)))
        if debug == 1: print(5, x.shape)
        x = torch.relu(self.batchnorm6(self.conv6(x)))
        if debug == 1: print(6, x.shape)
        x = torch.relu(self.batchnorm7(self.conv7(x)))
        if debug == 1: print(7, x.shape)
        x = torch.relu(self.batchnorm8(self.conv8(x)))
        if debug == 1: print(8, x.shape)
        x = self.conv9(x)
        if debug == 1: print(9, x.shape)
        x = torch.cat([x[:, 0:3, :, :], torch.sigmoid(x[:, 3:5, :, :])], dim=1)

        return x

    def get_loss(self, result, truth, lambda_coord = 5, lambda_noobj = 1):
        x_loss = (result[:, 1, :, :] - truth[:, 1, :, :]) ** 2
        y_loss = (result[:, 2, :, :] - truth[:, 2, :, :]) ** 2
        w_loss = (torch.sqrt(result[:, 3, :, :]) - torch.sqrt(truth[:, 3, :, :])) ** 2
        h_loss = (torch.sqrt(result[:, 4, :, :]) - torch.sqrt(truth[:, 4, :, :])) ** 2
        class_loss_obj = truth[:, 0, :, :] * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2
        class_loss_noobj = (1 - truth[:, 0, :, :]) * lambda_noobj * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2

        total_loss = torch.sum(lambda_coord * truth[:, 0, :, :] * (x_loss + y_loss + w_loss + h_loss) + class_loss_obj + class_loss_noobj)
        return total_loss