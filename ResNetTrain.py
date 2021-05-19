import os
import json
import time
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from ResNetModel import resnet50
from torchvision import transforms, datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(128),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),#来自官网参数
    "val": transforms.Compose([transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 获取数据集地址
image_root = lambda x : "/Dataset/dataset_expression/" if x=="E" else "/Dataset/dataset_gender/"
image_path = os.getcwd() + image_root(input("Gender(G) or Expression(E) = ")) # 表情识别
train_dataset = datasets.ImageFolder(root = image_path + "train", transform=data_transform["train"])
train_num = len(train_dataset)
dataset_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in dataset_list.items())
# 将要训练的数据集写为一个json标签方便查看
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

epoch_times = int(input("Epoch Times = "))
csv_note = input("Notes for csv = ")

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)
# 选择网络
net = resnet50(num_classes=2)

inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 2)

net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './resNet50.pth' # 将训练模型保存下来
train_csv = pd.DataFrame(columns=['Loss', 'Acc'])
for epoch in range(epoch_times):
    # 训练
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    # 验证
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

        # 训练过程文件存储至DataSave/CSV/中
        train_csv = train_csv.append({'Loss': running_loss / step, 'Acc': val_accurate}, ignore_index=True)

csv_path = os.getcwd() + "/DataSave/CSV/"
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
save_name = csv_path + now + csv_note + r"--ResNet.csv"
train_csv.to_csv(save_name, mode = 'a', header = True, index = True)

# 将文件存储在百度云中
try:
    from bypy import ByPy
except ImportError:
    print("If you want to upload the data files to BaiduDisk, please install the ByPy by :")
    print(">>> pip install bypy==1.6.10")
else:
    bp = ByPy()
    bp.upload(localpath=save_name, remotepath='CodeData', ondup='newcopy')
    print("Data files upload")

print('Finished Training')
