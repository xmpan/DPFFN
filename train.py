import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os
import time
from utils.data import MyDataSetVH, read_data_vh
from utils.model import MyLayer
from utils.loss import FusionLoss,CritionLoss
import argparse

def train_model(model, train_loader, optimizer, scheduler, alpha, device, batch_size, train_labels):
    model.train()
    Loss_temp = torch.zeros(math.ceil(len(train_labels)/batch_size))
    criterion_loss = torch.zeros(math.ceil(len(train_labels)/batch_size))
    fusion_loss = torch.zeros(math.ceil(len(train_labels)/batch_size))
    train_accuracy = 0
    count = 0

    for inputs_hh, inputs_vh, train_label in train_loader:
        inputs_hh, inputs_vh, train_label = inputs_hh.to(device), inputs_vh.to(device), train_label.to(device)

        outputs, global_futrue_hh, detail_futrue_hh, global_futrue_vh, global_futrue_vh = model(inputs_hh, inputs_vh)

        train_out = outputs.argmax(dim=1)
        train_accuracy += (train_out == train_label).sum().item()

        loss_fusion = FusionLoss(global_futrue_hh, global_futrue_vh,detail_futrue_hh, global_futrue_vh)
        loss_criterion = CritionLoss(outputs, train_label.type(torch.long))
        loss = loss_criterion + alpha * loss_fusion

        criterion_loss[count] = loss_criterion
        fusion_loss[count] = alpha * loss_fusion
        Loss_temp[count] = loss
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    Loss_epoch = torch.mean(Loss_temp)
    # print('Training: Loss = {:.6f}, Criterion Loss = {:.6f}, Fusion Loss = {:.6f}, Accuracy = {:.1f}%'.format(
    #     Loss_epoch, torch.mean(criterion_loss), torch.mean(fusion_loss), train_accuracy / len(train_labels)*100))
    return Loss_epoch,criterion_loss,fusion_loss,train_accuracy / len(train_labels)*100

def evaluate_model(model, test_loader, test_labels, device):
    model.eval()
    test_accuracy = 0

    with torch.no_grad():
        for inputs_test_hh, inputs_test_vh, test_label in test_loader:
            inputs_test_hh, inputs_test_vh, test_label = inputs_test_hh.to(device), inputs_test_vh.to(device), test_label.to(device)

            output_test, _, _, _, _ = model(inputs_test_hh, inputs_test_vh)
            test_out = output_test.argmax(dim=1)
            test_accuracy += (test_out == test_label).sum().item()

    test_accuracy = test_accuracy / len(test_labels)
    # print('Evaluation: Accuracy = {:.1f}%'.format(test_accuracy*100))
    return test_accuracy*100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPFN')
    parser.add_argument('-train', type=str, help='the address of train dataset')
    parser.add_argument('-test', type=str, help='the address of test dataset')
    parser.add_argument('-out', type=str, help='the output address')
    parser.add_argument('-epoch', type=int, default=300, help='the training epochs')
    parser.add_argument('-head', type=int, default=10, help='number of heads in Multi-Head Attention')
    parser.add_argument('-Global', type=int, default=10, help='number of Global module')
    parser.add_argument('-Local', type=int, default=10, help='number of Local module')
    parser.add_argument('-batch', type=int, default=1, help='the batch size of train dataset')
    parser.add_argument('-alpha', type=int, default=2.0, help='the hyper-parameter alpha in fusion loss')
    parser.add_argument('-cls', type=int, default=10, help='the number of classes')
    parser.add_argument('-len', type=int, default=512, help='the number of HRRP sequences')
    parser.add_argument('-device', type=str, default='cpu', help='the device to run the model')
    args = parser.parse_args()

    # Parameter initialization
    train_path, test_path, output_path = args.train, args.test, args.out
    epochs, n_heads, n_layers_G, n_layers_L, batch_size = args.epoch, args.head, args.Global, args.Local, args.batch
    alpha, cls, seq_len, device = args.alpha, args.cls, args.len, args.device
    d_model, d_ff, d_k, d_v = 100, 512, 64, 64
    batch_size_test, size_out, num_epochs = 1, 16, 40
    total_time = 0
    best_acc = 0


    # Loading data
    train_data_h, train_data_v, train_labels = read_data_vh(os.path.join(train_path, 'hh/train'), os.path.join(train_path, 'vh/train'))
    train_loader = Data.DataLoader(dataset=MyDataSetVH(train_data_h, train_data_v, train_labels), batch_size=batch_size, shuffle=True)
    samples, _, _ = train_data_h.shape
    del train_data_h, train_data_v
    test_data_h, test_data_v, test_labels = read_data_vh(os.path.join(test_path, 'hh/test'), os.path.join(test_path, 'vh/test'))
    test_loader = Data.DataLoader(dataset=MyDataSetVH(test_data_h, test_data_v, test_labels), batch_size=batch_size_test, shuffle=True)
    del test_data_h, test_data_v


    # Model initialization
    model = MyLayer(batch_size, d_model, n_layers_G, n_layers_L, cls, device, d_ff, seq_len, d_k, d_v, n_heads, size_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=samples // batch_size * num_epochs, gamma=0.8, last_epoch=-1)

    # Create the output address
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for epoch in range(epochs):
        start_time = time.time()
        # training stage
        Loss_epoch,criterion_loss,fusion_loss,train_accuracy = train_model(model, train_loader, optimizer, scheduler, alpha, device, batch_size, train_labels)
        train_time = time.time()

        # Evaluate stage
        test_accuracy = evaluate_model(model, test_loader, test_labels, device)
        eva_time = time.time()


        print('Epoch[{:d}/{:d}]: Training: Loss = {:.6f}, Criterion Loss = {:.6f}, Fusion Loss = {:.6f},  Evaluation: Accuracy = {:.1f}%, Training time is : {:.1f}s, Evaluate time is : {:.1f}s'.format(
                epoch+1,epochs,Loss_epoch, torch.mean(criterion_loss), torch.mean(fusion_loss),test_accuracy,train_time-start_time,eva_time-train_time))
        # Record running time
        if test_accuracy > best_acc:
            file_name = f'epoch={epoch}.pth'
            torch.save(model.state_dict(), os.path.join(output_path, file_name))
            best_acc = test_accuracy
        # Record the running time
        end_time = time.time()
        total_time += end_time - start_time
    print('Totol runtime for {:d} epochs: {:.6f} s'.format(epochs,total_time))
