import torch
from tqdm import tqdm
import numpy as np
import sys
from ZXX_utils.Gram_schmidt_optimization import Gram_s_optim_Layer

def DGLoss(W):
    W_1, W_2 = W.size()
    total = W_1**2 - W_1
    W_mm = torch.mm(W, W.t()) * (torch.ones(W_1) - torch.eye(W_1)).cuda()
    loss = torch.sum(torch.abs(W_mm).view(-1)) / total
    return loss

def DGL2Loss(W):
    W_1, W_2 = W.size()
    total = W_1**2 - W_1
    W_dis = 2-distance_matrix_vector(W, W) * (torch.ones(W_1) - torch.eye(W_1)).cuda()
    loss = torch.sum(W_dis.view(-1)) / total
    return loss



def MLCL_loss(Feat, label):
    batchsize = Feat.size(0)
    ch = Feat.size(1)
    Target = torch.tensor(torch.zeros([batchsize, batchsize]), requires_grad=False).cuda()
    for i in range(batchsize):
        T_i = label[i]
        for k in range(i, batchsize):
            T_k = label[k]
            if T_i == T_k:
                Target[i,k] = 1
                Target[k,i] = 1
    HTH = Feat.mm(Feat.t())
    loss = torch.sum(torch.abs(HTH-Target).view(-1)) / (batchsize**2- batchsize)
    return loss

def cal_topk(all_predict, all_label, num_total):
    num_top1 = 0
    num_top2 = 0
    num_top4 = 0
    num_top8 = 0
    num_top16 = 0
    num_top32 = 0

    for i in tqdm(range(num_total)):
        # print(i)
        query = all_predict[i, :]
        l2norm_query = torch.norm(all_predict - query, p=2, dim=1)
        l2norm_query[i] += 100000
        _, sort_index = torch.sort(l2norm_query)
        sort_label = all_label[sort_index]
        num_top1 += (torch.sum(sort_label[:1] == all_label[i]) > 0).int()
        num_top2 += (torch.sum(sort_label[:2] == all_label[i]) > 0).int()
        num_top4 += (torch.sum(sort_label[:4] == all_label[i]) > 0).int()
        num_top8 += (torch.sum(sort_label[:8] == all_label[i]) > 0).int()
        num_top16 += (torch.sum(sort_label[:16] == all_label[i]) > 0).int()
        num_top32 += (torch.sum(sort_label[:32] == all_label[i]) > 0).int()
    num_total = torch.tensor(num_total).float()
    num_top1 = 100 * num_top1.float() / num_total
    num_top2 = 100 * num_top2.float() / num_total
    num_top4 = 100 * num_top4.float() / num_total
    num_top8 = 100 * num_top8.float() / num_total
    num_top16 = 100 * num_top16.float() / num_total
    num_top32 = 100 * num_top32.float() / num_total
    return num_top1, num_top2, num_top4, num_top8, num_top16, num_top32

def cal_topk_mm(all_predict, all_label, num_total):

    l2norm_query = all_predict.mm(all_predict.t())
    l2norm_query = l2norm_query - 100* torch.eye(num_total)
    _, sort_index = torch.sort(-l2norm_query)
    sort_label = all_label[sort_index]
    flag = (sort_label == all_label.unsqueeze(1)).int()
    num_top1 = torch.sum(torch.sum(flag[:, :1], 1) > 0)
    num_top2 = torch.sum(torch.sum(flag[:, :2], 1) > 0)
    num_top4 = torch.sum(torch.sum(flag[:, :4], 1) > 0)
    num_top8 = torch.sum(torch.sum(flag[:, :8], 1) > 0)
    num_top16 = torch.sum(torch.sum(flag[:, :16], 1) > 0)
    num_top32 = torch.sum(torch.sum(flag[:, :32], 1) > 0)

    num_total = torch.tensor(num_total).float()
    num_top1 = 100 * num_top1.float() / num_total
    num_top2 = 100 * num_top2.float() / num_total
    num_top4 = 100 * num_top4.float() / num_total
    num_top8 = 100 * num_top8.float() / num_total
    num_top16 = 100 * num_top16.float() / num_total
    num_top32 = 100 * num_top32.float() / num_total
    return num_top1, num_top2, num_top4, num_top8, num_top16, num_top32

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def cal_topk_ed(all_predict, all_label, num_total):

    l2norm_query = distance_matrix_vector(all_predict,all_predict)
    l2norm_query = l2norm_query + 100* torch.eye(num_total)
    _, sort_index = torch.sort(l2norm_query)
    sort_label = all_label[sort_index]
    flag = (sort_label == all_label.unsqueeze(1)).int()
    num_top1 = torch.sum(torch.sum(flag[:, :1], 1) > 0)
    num_top2 = torch.sum(torch.sum(flag[:, :2], 1) > 0)
    num_top4 = torch.sum(torch.sum(flag[:, :4], 1) > 0)
    num_top8 = torch.sum(torch.sum(flag[:, :8], 1) > 0)
    num_top16 = torch.sum(torch.sum(flag[:, :16], 1) > 0)
    num_top32 = torch.sum(torch.sum(flag[:, :32], 1) > 0)

    num_total = torch.tensor(num_total).float()
    num_top1 = 100 * num_top1.float() / num_total
    num_top2 = 100 * num_top2.float() / num_total
    num_top4 = 100 * num_top4.float() / num_total
    num_top8 = 100 * num_top8.float() / num_total
    num_top16 = 100 * num_top16.float() / num_total
    num_top32 = 100 * num_top32.float() / num_total
    return num_top1, num_top2, num_top4, num_top8, num_top16, num_top32


def train_2(model, train_loader, criterion, optimizer, Lambda=0, Lambda2 = 0, k=None):

    #if criterion_begin==None:
    #    criterion_begin = criterion

    print('Training...')
    alphas = []

    epoch_loss = []
    num_correct = 0
    num_total = 0

    model.train()

    for i, (images, target) in enumerate(tqdm(train_loader)):

        image_var = torch.tensor(images).cuda()
        label = torch.tensor(target).cuda(non_blocking=True)

        y_pred, W, feat, alpha = model(image_var, train_flag=True)
        loss = criterion(y_pred, label, k) + Lambda * DGLoss(W) 
        epoch_loss.append(loss.item())
        alphas.append(alpha.item())

        # Prediction
        _, prediction = torch.max(y_pred.data, 1)
        num_total += y_pred.size(0)
        num_correct += torch.sum(prediction == label.data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        #if np.isnan(loss.detach().cpu().numpy()):
        #    sys.exit('Loss diverged')
    print(sum(alpha)/len(alpha))
    num_correct = torch.tensor(num_correct).float().cuda()
    num_total = torch.tensor(num_total).float().cuda()

    train_acc = 100 * num_correct / num_total

    return train_acc, sum(epoch_loss) / len(epoch_loss)


class RetricMetric(object):
    def __init__(self, is_query, feats_test, labels_test, feats_query, labels_query):
        self.is_query = is_query
        if not self.is_query:
            self.feats_test = self.feats_query = feats_test
            self.labels_test = self.labels_query = labels_test
        else:
            self.feats_test = feats_test
            self.labels_test = labels_test
            self.feats_query = feats_query
            self.labels_query = labels_query

        self.sim_mat = np.matmul(self.feats_query, np.transpose(self.feats_test))

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0
        for i in range(m):
            pos_sim = self.sim_mat[i][self.labels_test == self.labels_query[i]]
            neg_sim = self.sim_mat[i][self.labels_test != self.labels_query[i]]

            thresh = np.sort(pos_sim)[-2] if not self.is_query else np.max(pos_sim)
            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m

    def precision_at_k(self, k):
        m = len(self.sim_mat)
        accuracy = 0
        for i in range(m):
            knn_i = self.sim_mat[i]
            knn_i_index = np.argsort(-1 * knn_i)
            if self.is_query:
                knn_i_labels_k = self.labels_test[knn_i_index[:k]]
            else:
                knn_i_labels_k = self.labels_test[knn_i_index[1:k+1]]
            accuracy_per_sample = np.sum(knn_i_labels_k == self.labels_query[i])/k
            accuracy += accuracy_per_sample

        return accuracy / m

    def mean_average_precision_at_r(self, k):
        m = len(self.sim_mat)

        k_linspace = np.linspace(1, k, k)
        all_AP = 0

        for i in range(m):
            knn_i = self.sim_mat[i]
            knn_i_index = np.argsort(-1 * knn_i)
            if self.is_query:
                knn_i_labels_k = self.labels_test[knn_i_index[:k]]
            else:
                knn_i_labels_k = self.labels_test[knn_i_index[1:k+1]]

            equality = knn_i_labels_k == self.labels_query[i]
            num_same = np.sum(equality)
            if num_same != 0:
                AP = np.sum(np.cumsum(equality) * equality / k_linspace) / num_same
                all_AP += AP
        return all_AP / m

def test_3(model, test_loader, query_loader, recall, precision=[10], mAP=[100]):

    model.eval()

    epoch_loss = []

    num_total = 0

    for i, (images, target) in enumerate(tqdm(test_loader)):
        # Data.
        image_var = torch.tensor(images).cuda()
        label = torch.tensor(target).cuda(non_blocking=True)
        # Prediction.
        y_pred = model(image_var)
        if i == 0:
            all_predict_test = y_pred.data.cpu().numpy()
            all_label_test = label.data.cpu().numpy()
            #break
        else:
            all_predict_test = np.concatenate([all_predict_test, y_pred.data.cpu().numpy()], 0)
            all_label_test = np.concatenate([all_label_test, label.data.cpu().numpy()], 0)

        num_total += y_pred.size(0)

    if query_loader is not None:
        is_query = True
        for i, (images, target) in enumerate(tqdm(query_loader)):
            # Data.
            image_var = torch.tensor(images).cuda()
            label = torch.tensor(target).cuda(non_blocking=True)
            # Prediction.
            y_pred = model(image_var)
            if i == 0:
                all_predict_query = y_pred.data.cpu()
                all_label_query = label.data.cpu()
                # break
            else:
                all_predict_query = np.concatenate([all_predict_query, y_pred.data.cpu().numpy()], 0)
                all_label_query = np.concatenate([all_label_query, label.data.cpu().numpy()], 0)

            num_total += y_pred.size(0)
    else:
        is_query = False
        all_predict_query = None
        all_label_query = None

    retmatric = RetricMetric(is_query, all_predict_test, all_label_test, all_predict_query, all_label_query)

    recall_output = {}
    for k in recall:
        recall_output['%d' % k] = retmatric.recall_k(k) * 100
    precision_output = {}
    for k in precision:
        precision_output['%d' % k] = retmatric.precision_at_k(k) * 100
    mAP_output = {}
    for k in mAP:
        mAP_output['%d' % k] = retmatric.mean_average_precision_at_r(k) * 100
    all_predict = all_predict_test

    return recall_output, precision_output, mAP_output, all_predict


def train_triplet(model, train_loader, criterion, optimizer, Lambda=0, Lambda2 = 0):

    #if criterion_begin==None:
    #    criterion_begin = criterion

    print('Training...')

    epoch_loss = []
    num_correct = 0
    num_total = 0

    model.train()

    margin = 0.2

    for i, (images, target, pos_images, neg_images) in enumerate(tqdm(train_loader)):

        image_var = torch.tensor(images).cuda()
        label = torch.tensor(target).cuda(non_blocking=True)

        pos_image_var = torch.tensor(pos_images).cuda()
        neg_image_var = torch.tensor(neg_images).cuda()

        feat = model(image_var, train_flag=False)
        pos_feat = model(pos_image_var, train_flag=False)
        neg_feat = model(neg_image_var, train_flag=False)

        loss = criterion(feat, pos_feat, neg_feat)
        epoch_loss.append(loss.item())

        # Prediction
        #_, prediction = torch.max(y_pred.data, 1)
        #num_total += y_pred.size(0)
        #num_correct += torch.sum(prediction == label.data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        #if np.isnan(loss.detach().cpu().numpy()):
        #    sys.exit('Loss diverged')

    num_correct = torch.tensor(num_correct).float().cuda()
    num_total = torch.tensor(num_total).float().cuda()

    train_acc = 100 * num_correct / num_total

    return train_acc, sum(epoch_loss) / len(epoch_loss)

