import os
import pandas as pd
import time
import numpy as np

import torch
import torchvision
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torch.nn import functional as F
from torch.nn.modules import loss

import argparse
from ZXX_utils.load_csv_data import csv_Dataset
from ZXX_utils.IR_train_MLCL import train_2, test_3
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weightNorm
from ZXX_utils.Gram_schmidt_optimization import Gram_s_optimization
from model_file.resnet import resnet50, resnet101
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = argparse.ArgumentParser(description='Using Resnet50 to AAAI2019 FGIR')

parser.add_argument('--base_lr', dest='base_lr', type=float, default=0.005)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=56)
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-6)
parser.add_argument('--epoches', dest='epoches', type=int, default=100)
parser.add_argument('--model_name', dest='model_name', type=str, default='resnet50')
parser.add_argument('--top_k', dest='top_k', type=int, default=2)
parser.add_argument('--data_name', dest='data_name', type=str, default='CUB', help='CUB or CAR or SOP or INSHOP')
parser.add_argument('--margin', dest='margin', type=float, default=1)
parser.add_argument('--sigma', dest='sigma', type=float, default=100)
parser.add_argument('--Lambda', dest='Lambda', type=float, default=1e-1)
parser.add_argument('--gpu', dest='gpu', type=str, required=True)
#parser.add_argument('--model', dest='model', type=str, default='model_resnet50_new/best.pth')
#parser.add_argument('--save_result_path', dest='save_result_path', type=str, default='model_resnet50_new')
args = parser.parse_args()
print(args)
os.makedirs('results', exist_ok=True)
if args.data_name == 'CAR':
    num_class = 98
    IMAGENET_TRAINSET_SIZE = 8054
    train_csv_path = '/content/datasets/data/CAR196/car_train.csv'
    test_csv_path = '/content/datasets/data/CAR196/car_test.csv'
    query_csv_path = ''
    recall_k = [1, 2, 4, 8, 16, 32]
    precision_k = [1, 5, 10, 100]
    mAP_k = [1, 5, 10, 100]
elif args.data_name == 'CUB':
    num_class = 100
    train_csv_path = '/content/datasets/data/CUB_200_2011/train.csv'
    test_csv_path = '/content/datasets/data/CUB_200_2011/test.csv'
    query_csv_path = ''
    recall_k = [1, 2, 4, 8, 16, 32]
    precision_k = [1, 5, 10, 100]
    mAP_k = [1, 5, 10, 100]
elif args.data_name == 'DOG':
    num_class = 60
    train_csv_path = '/content/datasets/data/Dog/train.csv'
    test_csv_path = '/content/datasets/data/Dog/test.csv'
    query_csv_path = ''
    recall_k = [1, 2, 4, 8, 16, 32]
    precision_k = [1, 5, 10, 100]
    mAP_k = [1, 5, 10, 100]
elif args.data_name == 'FGVC':
    num_class = 50
    train_csv_path = '/content/datasets/data/FGVC/fgvc_trainval_retrieval.csv'
    test_csv_path = '/content/datasets/data/FGVC/fgvc_test_retrieval.csv'
    query_csv_path = ''
    recall_k = [1, 2, 4, 8, 16, 32]
    precision_k = [1, 5, 10, 100]
    mAP_k = [1, 5, 10, 100]
elif args.data_name == 'FLOWERS':
    num_class = 51
    train_csv_path = '/content/datasets/data/Flowers/train.csv'
    test_csv_path = '/content/datasets/data/Flowers/test.csv'
    query_csv_path = ''
    recall_k = [1, 2, 4, 8, 16, 32]
    precision_k = [1, 5, 10, 100]
    mAP_k = [1, 5, 10, 100]
elif args.data_name == 'PETS':
    num_class = 19
    train_csv_path = '../data/Pets/train.csv'
    test_csv_path = '../data/Pets/test.csv'
    query_csv_path = ''
    recall_k = [1, 2, 4, 8, 16, 32]
    precision_k = [1, 5, 10, 100]
    mAP_k = [1, 5, 10, 100]
elif args.data_name == 'SOP':
    num_class = 11318
    train_csv_path = '/content/datasets/SOP/train.csv'
    test_csv_path = '/content/datasets/SOP/test.csv'
    query_csv_path = ''
    recall_k = [1, 10, 20, 30, 40, 50]
    precision_k = [1, 5, 10, 100]
    mAP_k = [1, 5, 10, 100]
elif args.data_name == 'INSHOP':    # unfinish
    num_class = 3997
    train_csv_path = '/content/datasets/INSHOP/train.csv'
    test_csv_path = '/content/datasets/INSHOP/test.csv'
    query_csv_path = '/content/datasets/INSHOP/query.csv'
    recall_k = [1, 10, 100, 1000]
    precision_k = [1, 5, 10, 100]
    mAP_k = [1, 5, 10, 100]


file_top_k = args.top_k if args.top_k!=0 else num_class
args.save_result_path = 'results/%s_%s_%s_%d_%.2f' % (args.model_name, args.data_name, file_top_k, args.sigma, args.Lambda)
args.model = os.path.join(args.save_result_path, 'best.pth')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        self.out_chn = out_chn

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class LearnableMax(nn.Module):
    def __init__(self, out_chn):
        super(LearnableMax, self).__init__()
        self.max1 = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        self.max2 = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        self.out_chn = out_chn*2

    def forward(self, x):
        out = torch.max(x, self.max1.expand_as(x))
        out = torch.max(out, self.max2.expand_as(x))
        return out

class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        if len(norm.size()) == 1:
            x = x / norm.unsqueeze(-1).expand_as(x)
        else:
            [bs, ch, h, w] = x.size()
            norm = norm.view(bs, 1, h, w)
            x = x / norm.expand_as(x)
        return x

class cal_L2Norm(torch.nn.Module):
    def __init__(self):
        super(cal_L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)

        return norm

class Easy_loss(torch.nn.Module):
    def __init__(self):
        super(Easy_loss, self).__init__()
        self.file_top_k = None
    def forward(self, features, labels, k):
        #_assert_no_grad(target)
        #self.file_top_k = k
        batch_size = features.size(0)
        weights = torch.tensor(-1).float().cuda()
        loss_crossentropy = 0
        for i in range(batch_size):
            feature = features[i,:]
            #feature = feature - torch.max(feature.data)
            feat = feature[labels[i]]
            feat_topk = torch.topk(feature, self.file_top_k)[0]

            loss_crossentropy = loss_crossentropy - torch.log(
                    torch.exp(feat) / torch.sum(torch.exp(feat_topk)))

        return loss_crossentropy / batch_size


class ResNet50_IR(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        Resnet50featuremap = resnet50(pretrained=True)
        #Resnet50featuremap = torchvision.models.resnet50(pretrained=True)
        layer_conv1_conv4 = [Resnet50featuremap.conv1,
                             Resnet50featuremap.bn1,
                             Resnet50featuremap.relu,
                             Resnet50featuremap.maxpool,
                             ]

        for i in range(4):
            name = 'layer%d' % (i + 1)
            layer_conv1_conv4.append(getattr(Resnet50featuremap, name))
        #print(layer_conv1_conv4)

        conv1_conv5 = torch.nn.Sequential(*layer_conv1_conv4)
        self.features = conv1_conv5
        self.max_pool = torch.nn.AdaptiveMaxPool2d([1, 1])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d([1, 1])
        #self.alpha = args.sigma
        self.alpha = nn.Parameter(torch.ones(1)*0.1, requires_grad=True)
        
        # self.Lambda = 0.2
        self.fc = weightNorm(nn.Linear(2048 * 2, num_class, bias=False))

        #self.move12 = LearnableBias(2048)
        #self.prelu1 = nn.PReLU(2048)
        #self.max1 = LearnableMax(2048)
        #self.move13 = LearnableBias(2048)

    def forward(self, x, train_flag=False):
        batchsize = x.size(0)
        x = self.features(x)
        #x = self.move12(x)
        #x = self.max1(x)
        #x = self.move13(x)

        if train_flag:

            mx = self.max_pool(x).view(batchsize, -1)
            ax = self.avg_pool(x).view(batchsize, -1)
            # x = mx * ax
            x = torch.cat([mx, ax], 1)
            norm = cal_L2Norm()(x)

            v = self.fc.weight_v
            g = self.fc.weight_g
            v = v / g
            x = x / norm.unsqueeze(1)
            d = distance_matrix_vector(x, v)
            x_ = (10.0/self.alpha)*(args.margin - d)
            #x_ = self.fc(x) * self.alpha / norm.unsqueeze(1)
            #x_ = x_ / g.squeeze(1).unsqueeze(0)

            return x_, v, x, self.alpha

        else:
            '''
            # the step with SCDA
            scda_x = torch.sum(x, 1, keepdim=True)
            mean_x = torch.mean(scda_x.view(scda_x.size(0), -1), 1, True)
            mean_x = mean_x.unsqueeze(2).unsqueeze(3)
            #print(scda_x.size())
            ###############################print(mean_x.size())
            scda_x = scda_x - mean_x
            scda_x = scda_x >0
            scda_x = scda_x.float()
            x = x * scda_x
            '''

            mx = self.max_pool(x).view(batchsize, -1)
            ax = self.avg_pool(x).view(batchsize, -1)
            # x = mx * ax
            x = torch.cat([mx, ax], 1)
            x = L2Norm()(x)
            return x

class ResNet101_IR(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        #Resnet101featuremap = torchvision.models.resnet101(pretrained=True)
        Resnet101featuremap = resnet101(pretrained=True)
        layer_conv1_conv4 = [Resnet101featuremap.conv1,
                             Resnet101featuremap.bn1,
                             Resnet101featuremap.relu,
                             Resnet101featuremap.maxpool,
                             ]

        for i in range(4):
            name = 'layer%d' % (i + 1)
            layer_conv1_conv4.append(getattr(Resnet101featuremap, name))
        #print(layer_conv1_conv4)

        conv1_conv5 = torch.nn.Sequential(*layer_conv1_conv4)
        self.features = conv1_conv5
        self.max_pool = torch.nn.AdaptiveMaxPool2d([1, 1])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d([1, 1])
        self.alpha = args.sigma
        # self.Lambda = 0.2
        self.fc = weightNorm(nn.Linear(2048 * 2, num_class, bias=False))

    def forward(self, x, train_flag=False):
        batchsize = x.size(0)
        x = self.features(x)
        if train_flag:

            mx = self.max_pool(x).view(batchsize, -1)
            ax = self.avg_pool(x).view(batchsize, -1)
            # x = mx * ax
            x = torch.cat([mx, ax], 1)
            norm = cal_L2Norm()(x)

            v = self.fc.weight_v
            g = self.fc.weight_g
            v = v / g
            #x = x / norm.unsqueeze(1)
            # x_ = self.alpha * (-Distance_matrix()(x, v))
            x_ = self.fc(x) * self.alpha / norm.unsqueeze(1)
            #x_ = x_ / g.squeeze(1).unsqueeze(0)

            return x_, v, x

        else:

            # the step with SCDA
            scda_x = torch.sum(x, 1, keepdim=True)
            mean_x = torch.mean(scda_x.view(scda_x.size(0), -1), 1, True)
            mean_x = mean_x.unsqueeze(2).unsqueeze(3)
            #print(scda_x.size())
            ###############################print(mean_x.size())
            scda_x = scda_x - mean_x
            scda_x = scda_x >0
            scda_x = scda_x.float()
            x = x * scda_x

            mx = self.max_pool(x).view(batchsize, -1)
            ax = self.avg_pool(x).view(batchsize, -1)
            # x = mx * ax
            x = torch.cat([mx, ax], 1)
            x = L2Norm()(x)
            return x

class VGG16_IR(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        VGGFeaturemap = torchvision.models.vgg19(pretrained=True)
        self.features = VGGFeaturemap.features
        self.max_pool = torch.nn.AdaptiveMaxPool2d([1, 1])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d([1, 1])
        self.alpha = args.sigma
        # self.Lambda = 0.2
        self.fc = weightNorm(nn.Linear(512 * 2, num_class, bias=False))

    def forward(self, x, train_flag=False):
        batchsize = x.size(0)
        x = self.features(x)
        if train_flag:

            mx = self.max_pool(x).view(batchsize, -1)
            ax = self.avg_pool(x).view(batchsize, -1)
            # x = mx * ax
            x = torch.cat([mx, ax], 1)
            norm = cal_L2Norm()(x)

            v = self.fc.weight_v
            g = self.fc.weight_g
            v = v / g
            #x = x / norm.unsqueeze(1)
            # x_ = self.alpha * (-Distance_matrix()(x, v))
            x_ = self.fc(x) * self.alpha / norm.unsqueeze(1)
            #x_ = x_ / g.squeeze(1).unsqueeze(0)

            return x_, v, x

        else:

            # the step with SCDA
            scda_x = torch.sum(x, 1, keepdim=True)
            mean_x = torch.mean(scda_x.view(scda_x.size(0), -1), 1, True)
            mean_x = mean_x.unsqueeze(2).unsqueeze(3)
            #print(scda_x.size())
            ###############################print(mean_x.size())
            scda_x = scda_x - mean_x
            scda_x = scda_x >0
            scda_x = scda_x.float()
            x = x * scda_x

            mx = self.max_pool(x).view(batchsize, -1)
            ax = self.avg_pool(x).view(batchsize, -1)
            # x = mx * ax
            x = torch.cat([mx, ax], 1)
            x = L2Norm()(x)
            return x


def main():
    print('Come to the main function')

    print(args)

    lr = args.base_lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epoches = args.epoches
    model_path = args.model
    save_model_path = args.save_result_path
    save_train_file = os.path.join(save_model_path, 'train.txt')
    save_test_file = os.path.join(save_model_path, 'test.txt')

    Lambda = args.Lambda

    train_data_list = pd.read_csv(train_csv_path, encoding='gbk')
    test_data_list = pd.read_csv(test_csv_path, encoding='gbk')
    query_data_list = pd.read_csv(query_csv_path, encoding='gbk') if not query_csv_path=='' else None

    if args.model_name == 'resnet50':
        model = ResNet50_IR()
    elif args.model_name == 'resnet101':
        model = ResNet101_IR()
    elif args.model_name == 'vgg':
        model = VGG16_IR()
    model = torch.nn.DataParallel(model).cuda()
    #model.load_state_dict(torch.load(model_path))

    all_parameters = model.parameters()
    other_parameters = []
    for pname, p in model.named_parameters():
        if 'alpha' in pname:
            other_parameters.append(p)
    other_parameters_id = list(map(id, other_parameters))
    weight_parameters = list(filter(lambda p: id(p) not in other_parameters_id, all_parameters))


    #optimizer = torch.optim.Adam(
    #        [{'params' : other_parameters},
    #        {'params' : weight_parameters, 'weight_decay' : weight_decay}],
    #        lr=lr,)

    #model_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/epoches), last_epoch=-1)

    #optimizer = torch.optim.SGD([{'params' : other_parameters, 'weight_decay' : 0},
    #        {'params' : weight_parameters, 'weight_decay' : weight_decay}], lr=lr, momentum=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = sgd_optimizer(model, lr, 0.9, weight_decay)
    #model_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epoches * IMAGENET_TRAINSET_SIZE // args.batch_size)
    #criterion = nn.CrossEntropyLoss()
    criterion = Easy_loss()
    milestones = [100, 200]
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma= 0.1)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imagesize = 280
    train_data = csv_Dataset(train_data_list,
                             transform=transforms.Compose([
                                 transforms.Resize(imagesize),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(imagesize),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    test_data = csv_Dataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize(imagesize),
                                #transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(imagesize),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    if query_data_list is not None:
        query_data = csv_Dataset(query_data_list,
                                transform=transforms.Compose([
                                    transforms.Resize(imagesize),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(imagesize),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
    else:
        query_data = None


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size= 20, shuffle=False, pin_memory=False, num_workers=4)
    if query_data is not None:
        query_loader = DataLoader(query_data, batch_size=20, shuffle=False, pin_memory=False, num_workers=4)
    else:
        query_loader = None


    best_acc = 0.0
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    if not os.path.exists(os.path.join(save_model_path, 'result')):
        os.mkdir(os.path.join(save_model_path, 'result'))

    if not os.path.exists(save_train_file):
        f_train = open(save_train_file, 'w')
        f_train.writelines('Go training\n')
        f_train.close()
    else:
        ctime = os.path.getctime(save_train_file)
        import time
        ctime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(ctime))
        new_train_file = os.path.join(save_model_path, 'train%s.txt' % ctime)
        os.rename(save_train_file, new_train_file)

        f_train = open(save_train_file, 'w')
        f_train.writelines('----------%s------------\n' %  ctime)
        f_train.close()

    if not os.path.exists(save_test_file):
        f_test = open(save_test_file, 'w')
        f_test.writelines('Go testing\n')
        f_test.close()
    else:
        ctime = os.path.getctime(save_test_file)
        import time
        ctime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(ctime))
        new_test_file = os.path.join(save_model_path, 'test%s.txt' % ctime)
        os.rename(save_test_file, new_test_file)
        f_test = open(save_test_file, 'w')
        f_test.writelines('----------%s------------\n' % ctime)
        f_test.close()

    #test_top1, test_top2, test_top4, test_top8, test_top16, test_top32 = test_2(model=model, test_loader=test_loader)
    #print('%d top1: %4.2f%% top2: %4.2f%% top4: %4.2f%% top8: %4.2f%% top16: %4.2f%% top32: %4.2f%%'
    #      % (-1, test_top1, test_top2, test_top4, test_top8, test_top16, test_top32))

    # test for the original pretrained model
    '''
    recall_k_output, precision_k_output, mAP_k_output, np_predict = test_3(model, test_loader=test_loader,
                                                                           query_loader=query_loader,
                                                                           recall=recall_k,
                                                                           precision=precision_k,
                                                                           mAP=mAP_k)
    
    output_string = '%d epoch: ' % -1
    for k in recall_k:
        output_string = output_string + 'top%d:%4.2f, ' % (k, recall_k_output[str(k)] )
    for k in precision_k:
        output_string = output_string + "Precision_%d: %4.2f, " % (k, precision_k_output[str(k)])
    for k in mAP_k:
        output_string = output_string + "mAP_%d: %4.2f" % (k, mAP_k_output[str(k)])

    print(output_string)

    f_test = open(save_test_file, 'a')
    f_test.writelines('%s\n' % output_string)
    f_test.close()
    '''
    for epoch in range(epoches):
        
        k = 1 
        train_acc, train_loss = train_2(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, Lambda=Lambda, k=k)
        print('%d\t%4.3f\t\t%4.2f%%\t' % (epoch, train_loss, train_acc))
        recall_k_output, precision_k_output, mAP_k_output, np_predict = test_3(model, test_loader=test_loader, query_loader=query_loader, recall=recall_k,
                                                                           precision=precision_k,
                                                                           mAP=mAP_k)

        # val_acc = 0.05
        # val_loss = 8.664
        model_scheduler.step()

        if recall_k_output['1'] > best_acc:
          best_acc = recall_k_output['1']
          print('*', end='')
          try:
            torch.save(model.state_dict(),
            os.path.join(save_model_path, 'best.pth'))
            #np.save(os.path.join(save_model_path, 'feature.npy'), np_predict)
            result_name = os.path.join(save_model_path, 'result', 'result_%d_%3.2f' % (epoch, best_acc))

            f_test = open(save_test_file, 'a')
            f_test.writelines('*')
            f_test.close()
          except:
            a=0
            

            #f_train = open(save_train_file, 'a')
            #f_train.writelines('%d %4.3f %4.2f%%\n' % (epoch, train_loss, train_acc))
            #f_train.close()


        output_string = '%d epoch: ' % epoch
        for k in recall_k:
            output_string = output_string + 'top%d:%4.2f, ' % (k, recall_k_output[str(k)] )
        for k in precision_k:
            output_string = output_string + "Precision_%d: %4.2f, " % (k, precision_k_output[str(k)])
        for k in mAP_k:
            output_string = output_string + "mAP_%d: %4.2f" % (k, mAP_k_output[str(k)])

        print(output_string)
        try:
          f_test = open(save_test_file, 'a')
          f_test.writelines('%s\n' % output_string)
          f_test.close()
        except:
          a=0

def sgd_optimizer(model, lr, momentum, weight_decay):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_lr = lr
        apply_wd = weight_decay
        #if 'bias' in key:
        #    apply_lr = 2 * lr       #   Just a Caffe-style common practice. Made no difference.
        #if 'depth' in key:
        #    apply_wd = 0
        if 'alpha' in key:
            apply_lr = 1e-5

        print('set weight decay ', key, apply_wd, apply_lr)
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_wd}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer


if __name__ == '__main__':
    
    main()
    save_model_path = args.save_result_path
    save_train_file = os.path.join(args.save_result_path, 'train.txt')
    save_test_file = os.path.join(args.save_result_path, 'test.txt')
    ctime = os.path.getctime(save_train_file)
    ctime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(ctime))
    new_train_file = os.path.join(save_model_path, 'train%s.txt' % ctime)
    os.rename(save_train_file, new_train_file)

    ctime = os.path.getctime(save_test_file)
    ctime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(ctime))
    new_test_file = os.path.join(save_model_path, 'test%s.txt' % ctime)
    os.rename(save_test_file, new_test_file)







