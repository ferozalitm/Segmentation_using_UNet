#  segmentation_OxfordIIITPet.py
#           Author: T M Feroz Ali
#           Network architecture:UNet
#               Contracting/encoder layers no_of_ch: (3->64->128->256->512->1024)                   
#               Expansive/decoder layers ch: (1024->512->256->256->128->64->3)
#               All RelU non-linearity except Sigmoid for output layer
#           Use CE loss since 3 layer GT map (forground, background, boundary)
#           Use He initialization for Linear layers followed by relu.
#           Use Xavior initialization for Linear layers followed by Sigmoid (last layer)


import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import random
from torchvision.utils import save_image
import matplotlib.cm as cm

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
ip_dimn = 572
mask_dimn = 388
batch_size = 8

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = './Segmentation_OxfordIIITPet_results_maskCrop_3lyrOP_Adamlre-5_bS8_CE_SingleLR_initHeXavior/'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


class SegmetationMap1to3layer(object):
    "Convert trimap to 3 layer map"

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, sample):
        # Convert greyscale PIL Image to array
        img2 = np.array(sample[0])
        img2 = sample[0]

        # Initialize the 3-channel output
        three_channel_img = torch.zeros((3, img2.shape[0], img2.shape[1])).type(torch.float32)

        # Apply thresholds to create each channel
        for i in range(3):
            three_channel_img[i, :, :] = (img2 == self.thresholds[i]).type(torch.float32)
        
        return three_channel_img    


image_transforms_train = transforms.Compose([
    transforms.Resize((ip_dimn, ip_dimn)),
    transforms.ColorJitter(brightness=.5, contrast=.5, hue=.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor()
])

image_transforms_tst = transforms.Compose([
    transforms.Resize((ip_dimn, ip_dimn)),
    transforms.ToTensor()
])

mask_transforms = transforms.Compose([
    transforms.Resize((ip_dimn, ip_dimn)),
    transforms.CenterCrop((mask_dimn, mask_dimn)), 
    transforms.ToTensor(),
    SegmetationMap1to3layer((0.00392156875, 0.0078431375, 0.01176470625))
])

train_dataset = torchvision.datasets.OxfordIIITPet(root='../data/',
                                     split= 'trainval',
                                     target_types = 'segmentation',
                                     transform=image_transforms_train,
                                     target_transform = mask_transforms,
                                     download=True)

test_dataset = torchvision.datasets.OxfordIIITPet(root='../data/',
                                     split= 'test',
                                     target_types = 'segmentation',
                                     transform=image_transforms_tst,
                                     target_transform = mask_transforms,
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

# Finding image size and channel depth
# train_dataset[0][0].shape  -> torch.Size([3, 512, 512])
ip_image_size = train_dataset[0][0].shape[1]
print(f'image_size: {ip_image_size}')
ip_image_ch = train_dataset[0][0].shape[0]
print(f'ip_image_ch: {ip_image_ch}')
print(ip_image_ch)

no_batches_train = len(train_loader)
no_batches_tst = len(test_loader)
print(f"No_batches train: {no_batches_train}")
print(f"No_batches test: {no_batches_tst}")
no_batches_tst  = 10

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
        torch.nn.init.constant_(m.bias, 0.0)

def init_weights_Kaiming_He(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0.0)

# Build a fully connected layer and forward pass
class SemanticSegmentation(nn.Module):
    def __init__(self, ip_image_size, no_classes):
        super().__init__()

        # Compressive/encoder layers
        self.layer1a_e = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1),
            nn.ReLU())        
        self.layer1b_e = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.conv_op_image_size_encoder_layer1a = (ip_image_size - 3)//1 + 1        # Conv -> (24X24)
        self.conv_op_image_size_encoder_layer1a = (self.conv_op_image_size_encoder_layer1a - 3)//1 + 1        # Conv -> (24X24)
        self.conv_op_image_size_encoder_layer1b = (self.conv_op_image_size_encoder_layer1a - 2)//2 + 1        # Conv -> (11X11)

        self.layer2a_e = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride=1),
            nn.ReLU())
        self.layer2b_e = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.conv_op_image_size_encoder_layer2a = (self.conv_op_image_size_encoder_layer1b - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_encoder_layer2a = (self.conv_op_image_size_encoder_layer2a - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_encoder_layer2b = (self.conv_op_image_size_encoder_layer2a - 2)//2 + 1        # Conv -> (3X3)

        self.layer3a_e = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride=1),
            nn.ReLU())
        self.layer3b_e = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.conv_op_image_size_encoder_layer3a = (self.conv_op_image_size_encoder_layer2b - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_encoder_layer3a = (self.conv_op_image_size_encoder_layer3a - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_encoder_layer3b = (self.conv_op_image_size_encoder_layer3a - 2)//2 + 1        # Conv -> (3X3)

        self.layer4a_e = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1),
            nn.ReLU())
        self.layer4b_e = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.conv_op_image_size_encoder_layer4a = (self.conv_op_image_size_encoder_layer3b - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_encoder_layer4a = (self.conv_op_image_size_encoder_layer4a - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_encoder_layer4b = (self.conv_op_image_size_encoder_layer4a - 2)//2 + 1        # Conv -> (3X3)

        self.layer5a_e = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride=1),
            nn.ReLU())
        self.conv_op_image_size_encoder_layer5a = (self.conv_op_image_size_encoder_layer4b - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_encoder_layer5a = (self.conv_op_image_size_encoder_layer5a - 3)//1 + 1        # Conv -> (7X7)


        # Expansive/decoder layers
        self.layer5_d = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3, padding=1, stride=1))
        self.conv_op_image_size_decoder_layer5 = self.conv_op_image_size_encoder_layer5a*2                   # Conv -> (7X7)

        self.layer4a_d = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride=1),
            nn.ReLU())
        self.layer4b_d = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, padding=1, stride=1))
        self.conv_op_image_size_decoder_layer4a = (self.conv_op_image_size_decoder_layer5 - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_decoder_layer4a = (self.conv_op_image_size_decoder_layer4a - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_decoder_layer4b = self.conv_op_image_size_decoder_layer4a*2  

        self.layer3a_d = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride=1),
            nn.ReLU())
        self.layer3b_d = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, padding=1, stride=1))
        self.conv_op_image_size_decoder_layer3a = (self.conv_op_image_size_decoder_layer4b - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_decoder_layer3a = (self.conv_op_image_size_decoder_layer3a - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_decoder_layer3b = self.conv_op_image_size_decoder_layer3a*2  

        self.layer2a_d = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride=1),
            nn.ReLU())
        self.layer2b_d = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding=1, stride=1))
        self.conv_op_image_size_decoder_layer2a = (self.conv_op_image_size_decoder_layer3b - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_decoder_layer2a = (self.conv_op_image_size_decoder_layer2a - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_decoder_layer2b = self.conv_op_image_size_decoder_layer2a*2  

        self.layer1a_d = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1),
            nn.ReLU())
        self.conv_op_image_size_decoder_layer1a = (self.conv_op_image_size_decoder_layer2b - 3)//1 + 1        # Conv -> (7X7)
        self.conv_op_image_size_decoder_layer1a = (self.conv_op_image_size_decoder_layer1a - 3)//1 + 1        # Conv -> (7X7)

        # Output mask layer 1X1 convolution
        self.layer1b_d = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = no_classes, kernel_size = 1, stride=1))
            # nn.Sigmoid())  # Implicitly in CE loss.
        


    def forward(self, x):
        # Compressive/Encoder layers
        x = self.layer1a_e(x)
        self.layer1a_e_crop = transforms.functional.center_crop(x, self.conv_op_image_size_decoder_layer2b)
        x = self.layer1b_e(x)
        x = self.layer2a_e(x)
        self.layer2a_e_crop = transforms.functional.center_crop(x, self.conv_op_image_size_decoder_layer3b)
        x = self.layer2b_e(x)
        x = self.layer3a_e(x)
        self.layer3a_e_crop = transforms.functional.center_crop(x, self.conv_op_image_size_decoder_layer4b)
        x = self.layer3b_e(x)
        x = self.layer4a_e(x)
        self.layer4a_e_crop = transforms.functional.center_crop(x, self.conv_op_image_size_decoder_layer5)
        x = self.layer4b_e(x)
        x = self.layer5a_e(x)

        # Expansive/Decoder layers
        y = self.layer5_d(x)
        y = self.layer4a_d(torch.cat((y,self.layer4a_e_crop), 1))
        y = self.layer4b_d(y)
        y = self.layer3a_d(torch.cat((y,self.layer3a_e_crop), 1))
        y = self.layer3b_d(y)
        y = self.layer2a_d(torch.cat((y,self.layer2a_e_crop), 1))
        y = self.layer2b_d(y)
        y = self.layer1a_d(torch.cat((y,self.layer1a_e_crop), 1))
        y = self.layer1b_d(y)
        
        return y
  
        

# Build model.
no_classes = 3
model = SemanticSegmentation(ip_image_size, no_classes).to(device)
model.apply(init_weights_Kaiming_He)
model.layer1b_d.apply(init_weights_xavier)

# Build optimizer.
learning_rate = 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Build loss.
criterion = nn.CrossEntropyLoss()

no_epochs = 150
first_pass = True
epoch_all = []
loss_test_all = []
loss_train_all = []

for epoch in range(no_epochs):

  # Training for each epoch
  batch_idx = 0
  total_loss_train = 0    

  for batch_idx, (images, masks) in enumerate(train_loader):
    images = images.to(device)
    masks = masks.to(device)

    # Forward pass.
    pred_segmnt = model(images)
    
    if batch_idx < 10:
        mask_pred_segmnt_concate_RGB = torch.cat((masks, pred_segmnt.detach()), 2)
        image_mask_pred_segmnt_concate_RGB = torch.cat((transforms.functional.resize(img=images, size = mask_dimn), mask_pred_segmnt_concate_RGB), 2)
        save_image(image_mask_pred_segmnt_concate_RGB, os.path.join(sample_dir, f'train_image_mask_pred_segmnt_concate-{epoch}-{batch_idx}.png'))

    # Compute loss.
    loss = criterion(pred_segmnt, masks)
    if epoch == 0 and first_pass == True:
      print(f'Initial {epoch} loss: ', loss.item())
      first_pass = False

    # Compute gradients.
    optimizer.zero_grad()
    loss.backward()

    # 1-step gradient descent.
    optimizer.step()

    # calculating train loss
    total_loss_train += loss.item()

    if epoch == 0 and (batch_idx) % 10 == 0:
      print(f"Train Batch:{batch_idx}/{no_batches_train}, loss: {loss}, total_loss: {total_loss_train}")

  print(f'Train Epoch:{epoch}, Average Train loss:{total_loss_train/no_batches_train}' )

    
  # Testing after each epoch
  model.eval()
  with torch.no_grad():

    total_loss_test = 0

    for batch_idx, (images, masks) in enumerate(test_loader):
      if batch_idx >= no_batches_tst:
         break

      images = images.to(device)
      masks = masks.to(device)

      # Forward pass.
      pred_segmnt = model(images)

      if batch_idx < 10:
        mask_pred_segmnt_concate_RGB = torch.cat((masks, pred_segmnt.detach()), 2)
        image_mask_pred_segmnt_concate_RGB = torch.cat((transforms.functional.resize(img=images, size = mask_dimn), mask_pred_segmnt_concate_RGB), 2)
        save_image(image_mask_pred_segmnt_concate_RGB, os.path.join(sample_dir, f'test_image_mask_pred_segmnt_concate-{epoch}-{batch_idx}.png'))

      # Compute test loss.
      loss = criterion(pred_segmnt, masks)      
      total_loss_test += loss.item()

    print(f'Test Epoch:{epoch}, Average Test loss: {total_loss_test/no_batches_tst}')

  # PLotting train and test curves
  epoch_all.append(epoch)
  loss_test_all.append(total_loss_test/no_batches_tst)
  loss_train_all.append(total_loss_train/no_batches_train)
  plt.clf()
  plt.plot(epoch_all, loss_train_all, marker = 'o', mec = 'g', label='Average Train loss')
  plt.plot(epoch_all, loss_test_all, marker = 'o', mec = 'r', label='Average Test loss')
  plt.legend()
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()
  plt.savefig(os.path.join(sample_dir, '_Loss.png'))

  model.train()
