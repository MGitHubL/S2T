import sys

sys.path.append('../')
import torch
import numpy as np
import random
import math
import time
import argparse
from data_tlp_cite import DataHelper_t
from data_dyn_cite import DataHelper
from torch.utils.data import DataLoader
from model import Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FType = torch.FloatTensor
LType = torch.LongTensor

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    setup_seed(args.seed)
    Data = DataHelper(args.file_path, args.node_feature_path, args.neg_size, args.hist_len, args.directed,
                        tlp_flag=args.tlp_flag)
    loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = Model(args).to(device)
    model.load_state_dict(torch.load(args.model))

    s_emb_list = []
    t_emb_list = []
    dup_s_emb_list = []
    neg_embs_list = []
    loss_list = []
    node_pred_total = []
    node_truth_total = []

    model.eval()
    print('Testing......Waiting one epoch generation......')
    for i_batch, sample_batched in enumerate(loader):
        loss, s_emb, t_emb, dup_s_emb, neg_embs, node_pred, node_truth, s_node, t_node = model.forward(
            sample_batched['s_node'].type(LType).to(device),
            sample_batched['t_node'].type(LType).to(device),
            sample_batched['s_self_feat'].type(FType).reshape(-1, args.feat_dim).to(device),
            sample_batched['s_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to(device),
            sample_batched['s_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to(device),

            sample_batched['t_self_feat'].type(FType).reshape(-1, args.feat_dim).to(device),
            sample_batched['t_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to(device),
            sample_batched['t_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to(device),

            sample_batched['neg_self_feat'].type(FType).reshape(-1, args.feat_dim).to(device),
            sample_batched['neg_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to(device),
            sample_batched['neg_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to(device),

            sample_batched['event_time'].type(FType).to(device),
            sample_batched['s_history_times'].type(FType).to(device),
            sample_batched['s_his_his_times_list'].type(FType).to(device),
            sample_batched['t_history_times'].type(FType).to(device),
            sample_batched['t_his_his_times_list'].type(FType).to(device),
            sample_batched['neg_his_times_list'].type(FType).to(device),
            sample_batched['neg_his_his_times_list'].type(FType).to(device),
            sample_batched['s_edge_rate'].type(FType).to(device),
            sample_batched['t_edge_rate'].type(FType).to(device),
            training=False
        )
        s_emb_list.append(s_emb)
        t_emb_list.append(t_emb)
        dup_s_emb_list.append(dup_s_emb.reshape(-1, args.out_dim))
        neg_embs_list.append(neg_embs.reshape(-1, args.out_dim))
        loss_list.append(loss)
        node_pred_total.append(node_pred)
        node_truth_total.append(node_truth)

    s_emb_list = torch.cat(s_emb_list, dim=0)
    t_emb_list = torch.cat(t_emb_list, dim=0)
    dup_s_emb_list = torch.cat(dup_s_emb_list, dim=0)
    neg_embs_list = torch.cat(neg_embs_list, dim=0)
    truth = torch.ones(s_emb_list.size(0), dtype=torch.int)
    truth_neg = torch.zeros(neg_embs_list.size(0), dtype=torch.int)

    s_list = torch.cat((s_emb_list, dup_s_emb_list), dim=0)
    t_list = torch.cat((t_emb_list, neg_embs_list), dim=0)
    truth_list = torch.cat((truth, truth_neg), dim=0).numpy()

    dif_list = torch.abs(s_list - t_list).numpy()

    x_train, x_test, y_train, y_test = train_test_split(dif_list, truth_list, test_size=1 - args.train_ratio,
                                                        random_state=args.seed, stratify=truth_list)

    lr = LogisticRegression(max_iter=10000)
    lr.fit(x_train, y_train)
    y_test_pred = lr.predict(x_test)
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print('ACC: {}'.format(round(acc, 4)))
    print('F1: {}'.format(round(f1, 4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--emb_path', type=str, default='./data/amazon/emb_%d.txt')
    parser.add_argument('--file_path', type=str, default='./data/amazon/amazon.txt')
    parser.add_argument('--node_feature_path', type=str, default='./data/amazon/feature.txt')
    parser.add_argument('--model', type=str, default='./res/amazon/model.pkl')
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--neg_size', type=int, default=1)
    parser.add_argument('--hist_len', type=int, default=15)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--epoch_num', type=int, default=5, help='epoch number')
    parser.add_argument('--tlp_flag', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ncoef', type=float, default=0.01)
    parser.add_argument('--l2_reg', type=float, default=0.001)

    parser.add_argument('--train_ratio', type=float, default=0.8)

    args = parser.parse_args()

    main(args)
