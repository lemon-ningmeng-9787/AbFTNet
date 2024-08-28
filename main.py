import sys
import random
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import AMRDatasets
from models import cnn, transformer
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from info_nce_loss import InfoNCE
import mltools
import csv
import pickle
import matplotlib.pyplot as plt 
tqdm.disable = True

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default="HisarModdataset/train/", help='dataset path.')
parser.add_argument('--test_path', type=str, default="HisarModdataset/test/", help='dataset path.')
parser.add_argument('--batch_size', type=int, default=400, help='batch_size')
parser.add_argument('--max_epochs', type=int, default=500, help='max_epochs')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device')
parser.add_argument('--model_name', type=str, default='./models/partial_model.pth', help='model path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning_rate')
parser.add_argument('--classes', type=int, default=11, help='classes')
parser.add_argument('--alpha', type=float, default=0.3, help='alpha to ctl')
args = parser.parse_args()

set_random_seed(99003)

def train_model(opt, model):
    model = model.to(opt.device)
    plot_train_loss = []
    plot_train_acc = []
    plot_val_loss = []
    plot_val_acc = []

    best_acc = 0
    length_all = len(train_data)
    length_all_val = len(test_data)

    for epoch in range(opt.max_epochs):
        temp_train_loss = 0
        temp_train_acc = 0
        temp_val_loss = 0
        temp_val_acc = 0

        # train
        model.train()

        for x1, x2, y in tqdm(train_dataloader):
            x1, x2, y = x1.to(opt.device), x2.to(opt.device), y.to(opt.device)

            optimizer.zero_grad()
            cls_out1, cls_out2, cls_y = model((x1, x2))
            loss_cls = loss_func(cls_y, y)
            loss_ctl = loss_ctl_func(cls_out1, cls_out2)
            loss = loss_cls + loss_ctl * opt.alpha
            loss.backward()
            optimizer.step()
            pred = torch.max(cls_y, dim=1)[1]
            true = y
            acc = float(torch.eq(true.cpu(), pred.cpu()).sum() / len(x1))
            temp_train_loss += loss.item() * len(x1)
            temp_train_acc += acc * len(x1)

        # val
        test_steps = len(val_dataloader)
        res = 0.0

        model.eval()
        with torch.no_grad():
            for x1, x2, y in tqdm(test_dataloader):
                optimizer.zero_grad()
                x1, x2, y = x1.to(opt.device), x2.to(opt.device), y.to(opt.device)
                cls_out1, cls_out2, y_1 = model((x1, x2))

                loss_cls = loss_func(y_1, y)
                loss_ctl = loss_ctl_func(cls_out1, cls_out2)
                loss_10 = loss_cls + loss_ctl * opt.alpha

                # Loss_all = loss_func((y_2 + y_1) / 2, y)
                pred = torch.max(y_1, dim=1)[1]
                # true = torch.max(y, dim=1)[1]
                true = y
                acc = float(torch.eq(true.cpu(), pred.cpu()).sum() / len(x1))
                # res += Loss_all.item()
                temp_val_loss += loss_10.item() * len(x1)
                temp_val_acc += acc * len(x1)
            # scheduler.step(res / test_steps)

        print(f'|| {epoch + 1} / {opt.max_epochs} ||'
              f'|| training_loss {temp_train_loss / length_all:.4f} ||'
              f'|| verification_loss {temp_val_loss / length_all_val:.4f} ||')
        print(f'|| training_accuracy {temp_train_acc / length_all:.4f} ||'
              f'|| verification_accuracy {temp_val_acc / length_all_val:.4f} ||')

        plot_train_loss.append(temp_train_loss / length_all)
        plot_train_acc.append(temp_train_acc / length_all)

        plot_val_loss.append(temp_val_loss / length_all_val)
        plot_val_acc.append(temp_val_acc / length_all_val)

        if plot_val_acc[-1] > best_acc:
            best_acc = plot_val_acc[-1]
            if not os.path.exists('./models/'):
                os.mkdir('./models/')
            torch.save(model, opt.model_name)


def confusion_matrix_(preds, labels, matrix, pred_ed=False):
    if not pred_ed:
        preds = torch.argmax(preds, 1)
        # labels = torch.argmax(labels, 1)
    for p, t in zip(preds, labels):
        matrix[p, t] += 1
    return matrix

def get_test_result(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, recall, f1, precision, confusion_mat

def test_model(opt, test_dataloader, num_classes=4):
    model = torch.load(opt.model_name, map_location=opt.device)
    model.eval()
    temp_test_acc = 0
    length = len(test_data)
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        conf_mat = torch.zeros(num_classes, num_classes)
        for x1, x2, y in tqdm(test_dataloader):
            x1, x2, y = x1.to(opt.device), x2.to(opt.device), y.to(opt.device)
            _, _, y_1 = model((x1, x2))
            pred = torch.max(y_1 , dim=1)[1]
            # true = torch.max(y, dim=1)[1]
            true = y
            acc = float(torch.eq(true.cpu(), pred.cpu()).sum() / len(x1))
            y_pred_list.extend(pred.cpu())
            y_true_list.extend(true.cpu())
            conf_mat = confusion_matrix_(y_1, y, conf_mat)
            temp_test_acc += acc * len(x1)
        print(f'|| test_accuracy {temp_test_acc / length:.4f} ||')
        accuracy, recall, f1, precision, confusion_mat = get_test_result(y_true_list, y_pred_list)
        print(f'|| test accuracy {accuracy} |||| recall {recall} |||| f1 {f1} |||| precision {precision} \n confusion_mat : \n {confusion_mat}')
    
    classes = test_dataloader.dataset.mods
    snrs = test_dataloader.dataset.snrs
    lbl = test_dataloader.dataset.lbl
    y_pred_list = np.array(y_pred_list)
    y_true_list = np.array(y_true_list)
    # TODO: add snrs
    confnorm,_,_ = mltools.calculate_confusion_matrix(y_true_list,y_pred_list,classes)
    mltools.plot_confusion_matrix(confnorm, labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'],save_filename='figure/cnn2_total_confusion')

    # Plot confusion matrix
    acc = {}
    acc_mod_snr = np.zeros( (len(classes),len(snrs)) )
    i = 0
    for snr in snrs:

        # extract classes @ SNR
        # test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_SNRs = [lbl[x][1] for x in test_dataloader.dataset.idx]
        y_true_list_i = y_true_list[np.where(np.array(test_SNRs) == snr)]
        y_pred_list_i = y_pred_list[np.where(np.array(test_SNRs) == snr)]

        accuracy, recall, f1, precision, confusion_mat = get_test_result(y_true_list_i, y_pred_list_i)
        print(f'------------------------------{snr}-------------------------------')
        print(f'|| test accuracy {accuracy} |||| recall {recall} |||| f1 {f1} |||| precision {precision} \n confusion_mat : \n {confusion_mat}')

        confnorm_i,cor,ncor = mltools.calculate_confusion_matrix(y_true_list_i,y_pred_list_i,classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        with open('acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])
        mltools.plot_confusion_matrix(confnorm_i, labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'], title="Confusion Matrix",save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr,100.0*acc[snr]))

        acc_mod_snr[:,i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i,axis=1),3)
        i = i +1
    #plot acc of each mod in one picture
    dis_num=11
    for g in range(int(np.ceil(acc_mod_snr.shape[0]/dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g*dis_num
        end_index = np.min([(g+1)*dis_num,acc_mod_snr.shape[0]])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")

        for i in range(beg_index,end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
            # 设置数字标签
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig('figure/acc_with_mod_{}.png'.format(g+1))
        plt.close()
    #save acc for mod per SNR
    fd = open('predictresult/acc_for_mod_on_cnn2.dat', 'wb')
    pickle.dump(('128','cnn2', acc_mod_snr), fd)
    fd.close()

    # Save results to a pickle file for plotting later
    print(acc)
    fd = open('predictresult/cnn2_d0.5.dat','wb')
    pickle.dump( ("CNN2", 0.5, acc) , fd )

    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')
    plt.close()


if __name__ == '__main__':
    print(args)
    classes = args.classes
    
    train_data = AMRDatasets(args.train_path, 'train')
    val_data = AMRDatasets(args.train_path, 'val')
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)

    test_data = AMRDatasets(args.test_path, 'test')
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    print(f'|| train_data || {len(train_data)}'
          f'|| val_data || {len(val_data)}')
    loss_func = nn.CrossEntropyLoss()
    loss_ctl_func = InfoNCE(temperature=0.07)

    model = transformer.Model()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True)
    print('-----train model-----')
    train_model(args, model)
    test_model(args, test_dataloader, num_classes=classes)