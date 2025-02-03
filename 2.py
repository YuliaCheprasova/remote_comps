import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10
#import matplotlib.pyplot as plt
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
import math
from torch.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device('cuda')
max_int = sys.maxsize
# min_int = -sys.maxsize-1



def modified_lrs(lrs, ignore, max_int):
    n = len(lrs)
    num_epochs = len(lrs[0])
    if ignore == True:
        for i in range(n):
            for j in range(num_epochs):
                if lrs[i][j] == -50000:
                    for k in range(num_epochs):
                        lrs[i][k] = -50000
                    break
                elif lrs[i][j] > max_int:
                    lrs[i][j] = max_int
    else:
        for i in range(n):
            for j in range(num_epochs):
                if lrs[i][j] == -50000:
                    if j != 0:
                        lrs[i][j] = lrs[i][j-1]
                    else:
                        for k in range(1, num_epochs):
                            if lrs[i][k] != -50000:
                                lrs[i][j] = lrs[i][k]
                                break
                elif lrs[i][j] > max_int:
                    lrs[i][j] = max_int

def check_status():
    status = 0
    f_status = open('/home/mpiscil/cloud2/Yulia/gp_with_neural_network/Status2.txt', 'r')
    while status == 0:
        time.sleep(0.1)
        symbol = f_status.read(1)
        f_status.seek(0)
        if symbol == '0' or symbol == '1':
            status = int(symbol)
        else:
            continue
        # print(status)
    gen = int(f_status.readlines()[1])
    f_status.close()
    return gen


def get_lrs():
    f = open('/home/mpiscil/cloud2/Yulia/gp_with_neural_network/Lrs2.txt', 'r')
    # f = open('C:/Programs/Disk_C/My projects/genetic_programming_with_nn/genetic_programming_with_nn/Lrs.txt', 'r')
    try:
        lines = f.readlines()
        n_individs = len(lines)
        num_epochs = lines[0].count('\t')
        lrs = np.zeros((n_individs, num_epochs))
        for i, line in enumerate(lines):
            line = line.rstrip()
            line = line.split('\t')
            line = list(map(float, line))
            lrs[i] = np.array(line)
        print(lrs)
    finally:
        f.close()
    return n_individs, num_epochs, lrs


def train_cycle(results, num_epochs, lrs, k, optimizer, train_loader, model, criterion, log_filename, gen):
    f_log = open(log_filename, 'a', buffering=1)
    scaler = GradScaler()
    res = False
    for epoch in range(1, num_epochs + 1):
        time_epoch = time.time()
        lr = lrs[k][epoch - 1]
        if lr == -50000.0:
            results[k] = max_int
            res = True
            break
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
        print('Gen {} Tree {}: Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}, Time_for_epoch: {:.4f}'.format(
           gen, k, epoch, num_epochs, lr, running_loss, time.time() - time_epoch))
        f_log.write('Gen {} Tree {}: Epoch [{}/{}], lr:{:.4f}, Loss:{:.4f}, Time_for_epoch: {:.4f}\n'.format(
            gen, k, epoch, num_epochs, lr, running_loss, time.time() - time_epoch))
        if epoch > 1 and (running_loss > 10000 or math.isnan(running_loss)):
            results[k] = max_int
            res = True
            break
    f_log.close()
    return res


def test_cycle(test_loader, model, criterion, len_testdt, results, k):
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # получаем индекс
            correct += torch.sum(predicted == labels).item()

    test_loss /= len(test_loader)  # делим на количество батчей
    test_acc = correct / len_testdt  # делим на количество наблюдений в тестовой выборке
    if (math.isnan(test_loss)):
        results[k] = max_int
    else:
        results[k] = round(test_loss, 8)
    return test_loss, test_acc


def write_losses(n_individs, results):
    f_write = open('/home/mpiscil/cloud2/Yulia/gp_with_neural_network/Losses2.txt', 'w')
    try:
        for i in range(n_individs):
            f_write.write(f"{results[i]:.8f}\n")
    finally:
        f_write.close()


def main():
    torch.set_float32_matmul_precision("medium")# снижение точности вычислений
    torch.backends.cudnn.benchmark = True
    #parallel = True
    log_filename = '/home/mpiscil/cloud2/Yulia/gp_with_neural_network/Log_python2.txt'
    f_log = open(log_filename, 'w', buffering=1)
    print('start Python')
    f_log.write('start Python\n')
    num_generals = 500+1
    ignore = True # если True, то игнорируется целый индивид, если False, то вместо непосчитанного lr подставляется предыдущий или последующий
    batch_size = 128
    max_int = sys.maxsize
    # min_int = -sys.maxsize-1
    num_workers = 0
    time_prepar = time.time()
    generator = torch.Generator(device=device)
    transform = transforms.Compose(
        [transforms.ToTensor(), ])  # transforms.ToTensor() автоматически нормализует данные в случае картинок
    traindt = CIFAR10(root='data/', train=True, transform=transform, download=True)
    testdt = CIFAR10(root='data/', train=False, transform=transform, download=True)
    train_loader = DataLoader(traindt, batch_size, shuffle=True, generator=generator, num_workers=num_workers)
    test_loader = DataLoader(testdt, batch_size, shuffle=False, generator=generator, num_workers=num_workers)
    print('Time_data_preparation: {:.4f}'.format(time.time() - time_prepar))

    #if parallel:
    f_log.write('Time_data_preparation: {:.4f}\n'.format(time.time() - time_prepar))
    for general in range(num_generals):
        start = time.time()
        general = check_status()
        print('General {}\n'.format(general))
        f_log.write('General {}\n'.format(general))
        n_individs, num_epochs, lrs = get_lrs()
        for i in range(n_individs):
            f_log.write(str(lrs[i]))
        #modified_lrs(lrs, ignore, max_int)
        results = np.zeros(n_individs)
        for k in range(n_individs):
            time_individ = time.time()
            #model = av_Classifier()
            model = MobileNetV2()
            model = model.to(device)
            #model = torch.compile(model) кажется без этого лучше десятые части изменяются
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.1)
            model.train() # Если нет Dropout, то не имеет смысла, но считается правилом хорошего тона
            f_log.close()
            res = train_cycle(results, num_epochs, lrs, k, optimizer, train_loader, model, criterion, log_filename, general)
            f_log = open(log_filename, 'a', buffering=1)
            if (res == False):
                model.eval() # Если нет Dropout, то не имеет смысла, но считается правилом хорошего тона
                len_testdt = len(testdt)
                test_loss, test_acc = test_cycle(test_loader, model, criterion, len_testdt, results, k)
                print('Tree {}: TestLoss: {:.4f}, TestAccuracy: {:.4f}, Time_for_individ: {:.4f}'.format(k, test_loss, test_acc, time.time()-time_individ))
                f_log.write('Tree {}: TestLoss: {:.4f}, TestAccuracy: {:.4f}, Time_for_individ: {:.4f}\n'.format(k, test_loss, test_acc, time.time()-time_individ))
            else:
                print('Tree {} is not valid'.format(k))
                f_log.write('Tree {} is not valid\n'.format(k))

        print("Открытие файла для записи losses")
        f_log.write("Открытие файла для записи losses\n")
        write_losses(n_individs, results)
        print("Закрытие файла для записи losses")
        f_log.write("Закрытие файла для записи losses\n")
        f_status = open('/home/mpiscil/cloud2/Yulia/gp_with_neural_network/Status2.txt', 'w')
        f_status.write('0')
        f_status.close()
        print("Time_whole_program: {:.4f} seconds".format(time.time() - start))  # 76 85 почему-то, надо попробовть когда процессор будет не так занят
        f_log.write("Time_whole_program: {:.4f} seconds\n".format(time.time() - start))
    f_log.close()
    #time.sleep(1200)

if __name__ == '__main__':
    main()
