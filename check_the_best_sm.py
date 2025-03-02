import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
import torch.nn.functional as F
from models.av_MNIST import *
from models.mobilenetv2 import *
from models.resnet import *
from models.preact_resnet import *
from models.googlenet import *
from math import *
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
from sklearn.linear_model import LinearRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device('cuda')


def train_cycle(num_epochs, formula, optimizer, train_loader, model, criterion, log_filename):
    f_log = open(log_filename, 'a', buffering=1)
    scaler = GradScaler()
    coef = 0
    wind = 10
    losses = np.zeros(num_epochs)
    sm_losses = np.zeros(num_epochs)
    x_last = np.arange(1, wind+1)
    x_last = x_last.reshape(-1, 1)
    for epoch in range(1, num_epochs + 1):
        time_epoch = time.time()
        x00 = epoch
        x01 = coef
        lr = eval(formula)
        lr *= 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        running_loss = 0
        for images, labels in train_loader:  # цикл по батчам
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            torch.cuda.synchronize()
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        losses[epoch-1] = running_loss
        print('Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}, Time_for_epoch: {:.4f}'.format(
            epoch, num_epochs, lr, running_loss, time.time() - time_epoch))
        f_log.write('Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}, Time_for_epoch: {:.4f}\n'.format(
            epoch, num_epochs, lr, running_loss, time.time() - time_epoch))
        if epoch % wind == 0:
            sm_losses[0] = losses[0]
            for i in range(1, epoch):
                sm_losses[i] = sm_losses[i - 1] * 0.8 + losses[i] * 0.2
            sm_losses_last = sm_losses[epoch-wind:epoch]
            reg = LinearRegression().fit(x_last, sm_losses_last)
            coef = reg.coef_[0]
    f_log.close()


def test_cycle(test_loader, model, criterion, len_testdt):
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            test_loss += loss.item()
            # print(test_loss)
            _, predicted = torch.max(outputs.data, 1)  # получаем индекс
            correct += torch.sum(predicted == labels).item()
    # print(len(test_loader))
    test_loss /= len(test_loader)  # делим на количество батчей
    test_acc = correct / len_testdt  # делим на количество наблюдений в тестовой выборке
    return test_loss, test_acc


#LOOK AT INITIAL_LR, FILENAME, TRANSFORMER, LIST, DATA, NET, NUM_classes, num_epochs
def main():
    # не забывай менять названия файлов и выражение для расчета lr
    torch.set_float32_matmul_precision("medium")  # снижение точности вычислений
    torch.backends.cudnn.benchmark = True
    #log_filename = '/home/mpiscil/cloud2/Yulia/gp_with_neural_network/Log_ep^0,29.txt'
    #log_filename = '/home/mpiscil/cloud2/Yulia/gp_with_neural_network/Log_exp(sin(ep)).txt'
    log_filename = '/home/mpiscil/cloud2/Yulia/gp_with_neural_network/3_sm_expr_google_C10.txt'
    f_log = open(log_filename, 'w', buffering=1)
    f_wr = open('/home/mpiscil/cloud2/Yulia/gp_with_neural_network/3_sm_expr_google_C10_losses.txt', 'w', buffering=1)
    print('start Python')
    f_log.write('start Python\n')
    batch_size = 128
    time_prepar = time.time()
    generator = torch.Generator(device=device)
    transform = transforms.Compose([transforms.ToTensor(), ])  # transforms.ToTensor() автоматически нормализует данные в случае картинок
    """transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])"""
    traindt = CIFAR10(root='data/', train=True, transform=transform, download=True)
    testdt = CIFAR10(root='data/', train=False, transform=transform, download=True)
    train_loader = DataLoader(traindt, batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(testdt, batch_size, shuffle=False, generator=generator)
    print('Time_data_preparation: {:.4f}'.format(time.time() - time_prepar))
    f_log.write('Time_data_preparation: {:.4f}\n'.format(time.time() - time_prepar))
    num_epochs = 100
    num_classes = 10
    #expr = "cos(cos(e))"
    #expr = "(e)**0.29"
    #expr = "exp(sin(e))"
    #expr = 'log(e)'
    #expr = '3.5'
    #expr = "exp(cos(x01-x00))"
    exprs = ["(1.98)**(cos(x00)+x01)", "(x00)**(0.359+x01)", "(x00)**(x01)"]
    for expr in exprs:
        f_log.write(f"Testing expression {expr}\n")
        for l in range(5):
            time_individ = time.time()
            # model = av_Classifier()
            #model = MobileNetV2()
            #model = ResNet18(num_classes)
            #model = PreActResNet18(num_classes)
            model = GoogLeNet()
            model = model.to(device)
            # model = torch.compile(model) кажется без этого лучше, было 21 с стало 27
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            model.train()  # Если нет Dropout, то не имеет смысла, но считается правилом хорошего тона
            f_log.close()
            train_cycle(num_epochs, expr, optimizer, train_loader, model, criterion, log_filename)
            f_log = open(log_filename, 'a', buffering=1)
            model.eval()  # Если нет Dropout, то не имеет смысла, но считается правилом хорошего тона
            len_testdt = len(testdt)
            test_loss, test_acc = test_cycle(test_loader, model, criterion, len_testdt)
            print(
                'TestLoss: {:.4f}, TestAccuracy: {:.4f}, Time_for_individ: {:.4f}'.format(test_loss,
                                                                                                    test_acc,
                                                                                                    time.time() - time_individ))
            f_log.write(
                'TestLoss: {:.4f}, TestAccuracy: {:.4f}, Time_for_individ: {:.4f}\n'.format(test_loss,
                                                                                                    test_acc,
                                                                                                    time.time() - time_individ))
            f_wr.write('{:.4f} {:.4f}\n'.format(test_loss, test_acc))

    f_log.close()
    f_wr.close()


if __name__ == '__main__':
    main()


 
