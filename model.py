import torch
from torch import nn, optim
from dgnn import DGNN
from film import Scale_4, Shift_4
from Emlp import EMLP
from node_relu import Node_edge
from global_prior import Global_emb, Global_w
from loss_w import Loss_weight


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.l2reg = args.l2_reg  # 0.001
        self.ncoef = args.ncoef  # 0.01
        self.EMLP = EMLP(args)  # [1,d],[1]
        self.gnn = DGNN(args)
        self.scale_e = Scale_4(args)
        self.shift_e = Shift_4(args)
        self.node_edge = Node_edge(args)
        self.global_w = Global_w(args)
        self.global_emb = Global_emb(args)
        self.loss_w = Loss_weight(args)

        # self.g_optim = optim.Adam(self.grow_f.parameters(), lr=args.lr)

        self.optim = optim.Adam([{'params': self.gnn.parameters()},
                                 {'params': self.EMLP.parameters()},
                                 {'params': self.scale_e.parameters()},
                                 {'params': self.shift_e.parameters()},
                                 {'params': self.node_edge.parameters()},
                                 {'params': self.global_w.parameters()},
                                 {'params': self.global_emb.parameters()},
                                 {'params': self.loss_w.parameters()},
                                 ], lr=args.lr)

    def forward(self, s_node, t_node, s_self_feat, s_one_hop_feat, s_two_hop_feat,
                t_self_feat, t_one_hop_feat, t_two_hop_feat,
                neg_self_feat, neg_one_hop_feat, neg_two_hop_feat,
                e_time, s_his_time, s_his_his_time,
                t_his_time, t_his_his_time,
                neg_his_time, neg_his_his_time,
                s_edge_rate, t_edge_rate,
                training=True):
        s_gnn = self.gnn(s_self_feat, s_one_hop_feat, s_two_hop_feat,
                         e_time, s_his_time, s_his_his_time)
        t_gnn = self.gnn(t_self_feat, t_one_hop_feat, t_two_hop_feat,
                         e_time, t_his_time, t_his_his_time)
        neg_gnn = self.gnn(neg_self_feat, neg_one_hop_feat, neg_two_hop_feat,
                            e_time, neg_his_time, neg_his_his_time, neg=True)

        s_global_update, s_node_update = self.global_w(s_edge_rate)
        t_global_update, t_node_update = self.global_w(t_edge_rate)

        global_emb = self.global_emb(s_gnn, t_gnn, s_global_update, t_global_update)

        s_emb = s_gnn + s_node_update.unsqueeze(-1) * global_emb
        t_emb = t_gnn + t_node_update.unsqueeze(-1) * global_emb

        neg_embs = neg_gnn + (((s_node_update + t_node_update) / 2).unsqueeze(-1) * global_emb).unsqueeze(1)

        ij_cat = torch.cat((s_emb, t_emb), dim=1)
        alpha_ij = self.scale_e(ij_cat)
        beta_ij = self.shift_e(ij_cat)
        # 公式8、9
        theta_e_new = []
        for s in range(2):
            theta_e_new.append(torch.mul(self.EMLP.parameters()[s], (alpha_ij[s] + 1)) + beta_ij[s])

        p_dif = (s_emb - t_emb).pow(2)
        p_scalar = (p_dif * theta_e_new[0]).sum(dim=1, keepdim=True)
        p_scalar += theta_e_new[1]
        p_scalar_list = p_scalar

        event_intensity = torch.sigmoid(p_scalar_list) + 1e-6  # [b,1]
        log_event_intensity = torch.mean(-torch.log(event_intensity))  # [1]
        # 公式11正样本

        dup_s_emb = s_emb.repeat(1, 1, self.args.neg_size)
        dup_s_emb = dup_s_emb.reshape(s_emb.size(0), self.args.neg_size, s_emb.size(1))

        neg_ij_cat = torch.cat((dup_s_emb, neg_embs), dim=2)
        neg_alpha_ij = self.scale_e(neg_ij_cat)
        neg_beta_ij = self.shift_e(neg_ij_cat)
        neg_theta_e_new = []
        for s in range(2):
            neg_theta_e_new.append(torch.mul(self.EMLP.parameters()[s], (neg_alpha_ij[s] + 1)) + neg_beta_ij[s])

        neg_dif = (dup_s_emb - neg_embs).pow(2)
        neg_scalar = (neg_dif * neg_theta_e_new[0]).sum(dim=2, keepdim=True)
        neg_scalar += neg_theta_e_new[1]
        big_neg_scalar_list = neg_scalar

        neg_event_intensity = torch.sigmoid(- big_neg_scalar_list) + 1e-6

        neg_mean_intensity = torch.mean(-torch.log(neg_event_intensity))

        pos_l2_loss = [torch.norm(s, dim=1) for s in alpha_ij]
        pos_l2_loss = [torch.mean(s) for s in pos_l2_loss]
        pos_l2_loss = torch.sum(torch.stack(pos_l2_loss))
        pos_l2_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=1)) for s in beta_ij]))
        neg_l2_loss = torch.sum(torch.stack([torch.mean(torch.norm(s, dim=2)) for s in neg_alpha_ij]))
        neg_l2_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=2)) for s in neg_beta_ij]))

        l_theta = pos_l2_loss + neg_l2_loss
        delta_e = self.node_edge(s_emb)
        smooth_loss = nn.SmoothL1Loss()
        l_node = smooth_loss(delta_e, s_edge_rate.reshape(s_edge_rate.size(0), 1))
        node_pred = delta_e
        node_truth = s_edge_rate.reshape(s_edge_rate.size(0), 1)

        h_intensity = self.gnn.hawkes(s_self_feat, t_self_feat, s_one_hop_feat, t_one_hop_feat, e_time, s_his_time, t_his_time)
        l_contra = smooth_loss(p_scalar_list, h_intensity)

        l_emb = torch.mean(-torch.log(torch.sigmoid((s_emb - global_emb).pow(2)) + 1e-6)) +\
                torch.mean(-torch.log(torch.sigmoid((t_emb - global_emb).pow(2)) + 1e-6))
        l_hawkes = torch.mean(-torch.log(torch.sigmoid(h_intensity) + 1e-6))

        loss = self.loss_w(l_contra, l_node, l_theta, l_hawkes, l_emb)
        L = log_event_intensity + neg_mean_intensity + loss

        if training == True:
            self.optim.zero_grad()
            L.backward()
            self.optim.step()

        return round((L.detach().clone()).cpu().item(), 4),\
               s_emb.detach().clone().cpu(),\
               t_emb.detach().clone().cpu(),\
               dup_s_emb.detach().clone().cpu(),\
               neg_embs.detach().clone().cpu(),\
               node_pred.detach().clone().cpu(),\
               node_truth.detach().clone().cpu(),\
               s_node.detach().clone().cpu(),\
               t_node.detach().clone().cpu()