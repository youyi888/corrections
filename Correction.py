import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
import pytorch_msssim
import cv2
import csv
from tqdm import tqdm



class IlluminationCorrectionNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IlluminationCorrectionNet, self).__init__()

        # 定义特征提取器（Encoder）
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),      
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 定义解码器（Decoder）
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(64, 64+ in_channels, kernel_size=1, stride=1)
        )
        # 定义细节补充模块（Detail Enhancement Module）
        self.detail_module = nn.Sequential(
            nn.Conv2d(64+ in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        #插值调整输入大小
        downsampled_input1 = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        # 前向传递过程，包括特征提取器、光照校正、卷积层、解码器和输出层
        encoder_output1 = self.encoder[0](downsampled_input1)
        encoder_output1 = self.encoder[1](encoder_output1)

        encoder_output2 = self.encoder[2](encoder_output1)
        encoder_output2 = self.encoder[3](encoder_output2)

        encoder_output3 = self.encoder[4](encoder_output2)
        encoder_output3 = self.encoder[5](encoder_output3)

        # 进行特征解码，并使用跳跃连接
        decoder_output1 = self.decoder[0](encoder_output3)
        decoder_output1 = self.decoder[1](decoder_output1)
        decoder_output1 = self.decoder[2](decoder_output1)

        decoder_output2 = self.decoder[3](decoder_output1 + encoder_output2)
        decoder_output2 = self.decoder[4](decoder_output2)
        decoder_output2 = self.decoder[5](decoder_output2)

        decoder_output3 = self.decoder[6](decoder_output2 + encoder_output1)
        decoder_output3 = self.decoder[7](decoder_output3)
        decoder_output3 = self.decoder[8](decoder_output3)
        # decoder_output4 = self.decoder[6](decoder_output3) 
        # 插值调整输出大小
        upsampled_output1 = nn.functional.interpolate(decoder_output3, size=x.size()[2:], mode='bilinear', align_corners=True)

        # decoder_output4 = self.decoder[9](upsampled_output1)
        # 细节补充模块，添加两个残差连接
        detail_input = torch.cat([upsampled_output1, x], dim=1)

        detail_output1 = self.detail_module[0](detail_input)
        detail_output1 = self.detail_module[1](detail_output1) 

        detail_output2 = self.detail_module[2](detail_output1)
        detail_output2 = self.detail_module[3](detail_output2)

        detail_output3 = self.detail_module[4](detail_output2)
        detail_output3 = self.detail_module[5](detail_output3)

        detail_output4 = self.detail_module[6](detail_output3)
        detail_output4 = self.detail_module[7](detail_output4)

        detail_output5 = self.detail_module[8](detail_output4)
        detail_output5 = self.detail_module[9](detail_output5)

        detail_output4 = detail_output2 + detail_output4
        detail_output3 = detail_output1 + detail_output3
        
        output = detail_output5
        return output
    #########################################################
    
class IlluminationLoss(nn.Module):
    def __init__(self):
        super(IlluminationLoss, self).__init__()
        self.msssim = pytorch_msssim.ms_ssim

    def forward(self, output, target):
        l1_loss = nn.L1Loss()(output, target)  # 计算 L1 Loss
        msssim_loss = 1 - self.msssim(output, target, data_range=1.0, size_average=True)  # 计算 MS-SSIM Loss
        loss = l1_loss + msssim_loss  # 组合两部分 Loss
        return loss
    
class IlluminationDataset(data.Dataset):
    def __init__(self, image_paths, target_paths, target_size):
        self.images = [self.load_and_resize_image(path, target_size) for path in image_paths]
        self.targets = [self.load_and_resize_image(path, target_size) for path in target_paths]

    def load_and_resize_image(self, path, target_size):
        image = cv2.imread(path)
        image = cv2.resize(image, target_size)  # 调整图像大小
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        return image

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
        return image, target

    def __len__(self):
        return len(self.images)

csv_path = "7.csv"
image_paths = []
target_paths = []
target_size = (320, 320)
with open(csv_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        image_paths.append(row[0])
        target_paths.append(row[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
dataset = IlluminationDataset(image_paths, target_paths,target_size)
batch_size = 8
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

in_channels = 3
out_channels = 3
model = IlluminationCorrectionNet(in_channels, out_channels).to(device)
# model = model.half()
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)
criterion =  IlluminationLoss()

num_epochs = 100
best_loss = float('inf')
best_model = None

for epoch in (range(num_epochs)):
    for images, targets in tqdm(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        scheduler.step(loss)  #更新学习率
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    if loss < best_loss:
        best_loss = loss
        best_model = model.state_dict(),
        torch.save(best_model, f"trained_best_model_.pth")

torch.save(model.state_dict(), f"trained_last_model_.pth")


# 使用训练好的模型对新的显微图像进行光照校正
# test_image = torch.tensor(cv2.imread('./0/002x032.jpg')).permute(2, 0, 1).unsqueeze(0).float() / 255.0
# test_image = test_image.to(device)
# model.load_state_dict(torch.load("trained_model.pth"))
# model.eval()
# output = model(test_image)
