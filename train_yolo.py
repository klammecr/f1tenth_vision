# Third Party
import torch
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import pytorch_lightning as pl

# In House
from yolo_model import F110Lightning
from yolo_utils import F110Dataset, DisplayImage, DisplayLabel


# Entrypoint
if __name__ == "__main__":
    dataset_folder = 'f110_dataset_20220209/'
    path = f"{dataset_folder}labels.npy" 
    labels = np.load(path)
    print(len(labels))

    # Parameters
    final_dim = [6, 10]
    input_dim = [180, 320]
    anchor_size = [(input_dim[0] / final_dim[0]), (input_dim[1] / final_dim[1])]
    arr = np.arange(labels.shape[0])
    np.random.shuffle(arr)

    # Due to the small size of dataset, we preprocess them into memory to speed up training.
    images = []
    for ind in range(labels.shape[0]):
        img_path = dataset_folder + str(ind) + '.jpg'
        img = cv2.imread(img_path) / 255.0
        img = cv2.resize(img, (input_dim[1], input_dim[0]))
        images.append(img)

    # Create the sets
    train_set = F110Dataset([0, 9.5], dataset_folder, labels, images, arr, input_dim, anchor_size, final_dim)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    validation_set = F110Dataset([9.5, 10], dataset_folder, labels, images, arr, input_dim, anchor_size, final_dim)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=1, shuffle=True)

    ## Plot one label to see if it's correct.
    # for data_ind, data_list in enumerate(train_loader):
    #     if data_ind == 0:
    #         image = data_list[0]
    #         label_gt = data_list[1]
    #         label = data_list[2]
    #         break
    # print(image[0].numpy().shape)
    # print(label[0])
    # print(label_gt[0][0])
    # print(label)
    # DisplayLabel(np.transpose(image[0].numpy(), (1, 2, 0)), label)

    ## Training Process
    batch_size = 64
    epochs = 100
    lr = 1e-3# Setting this kinda large since we don't have much data

    # Train and validation loaders
    f110    = F110Lightning(lr, anchor_size, input_dim, final_dim)
    trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=10)
    train_loader      = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
    trainer.fit(f110, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    # save the model
    trainer.save_checkpoint("YOLO_latest.pt")
