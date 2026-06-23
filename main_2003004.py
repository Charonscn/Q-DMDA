"""
EEG conformer 

Test SEED data 1 second
perform strict 5-fold cross validation 
"""

import argparse
import copy
import csv
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第2号物理GPU
import numpy as np
import random
import time
from data import load_mi1, fine_tuning_load_XY_MI

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
import losses as utils
from losses import LabelSmooth
import adversarial as Adver_network
from torch import Tensor
from einops import rearrange
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from core_qnn.quaternion_layers import QuaternionLinear

class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r * torch.tanh(self.bias))
        output = cosine
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class QMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.keys = QuaternionLinear(emb_size, emb_size)
        self.queries = QuaternionLinear(emb_size, emb_size)
        self.values = QuaternionLinear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = QuaternionLinear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling_base = self.head_dim if args.qattn_scale == 'head' else self.emb_size
        scaling = scaling_base ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


def _conv1d_bn_elu(in_channels, out_channels, kernel_size):
    padding = kernel_size // 2
    return [
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ELU()
    ]


class PatchEmbeddingTemporalFeat(nn.Module):
    """
    最简单时间分支：
    在时间轴 T 上做卷积，再映射到 emb_size
    输入:  [B, T, D]
    输出:  [B, T, emb_size]
    """
    def __init__(self, in_dim=300, emb_size=128, residual=True, res_scale_init=1.0, conv_layers=1):
        super().__init__()
        self.residual = residual
        layers = []
        for _ in range(max(1, conv_layers)):
            layers.extend(_conv1d_bn_elu(in_dim, in_dim, 3))
        layers.extend(_conv1d_bn_elu(in_dim, emb_size, 1))
        self.temporal_conv = nn.Sequential(*layers)
        self.shortcut = nn.Linear(in_dim, emb_size) if residual else nn.Identity()
        self.res_scale = nn.Parameter(torch.tensor([res_scale_init], dtype=torch.float32)) if residual else None

    def forward(self, x):  # x: [B, T, D]
        res = self.shortcut(x) if self.residual else None
        x = x.transpose(1, 2)          # [B, D, T]
        x = self.temporal_conv(x)      # [B, emb, T]
        x = x.transpose(1, 2)          # [B, T, emb]
        if self.residual:
            x = x + self.res_scale * res
        return x


class PatchEmbeddingFeatureFeat(nn.Module):
    """
    最简单空间/特征分支：
    先把每个样本的时间窗平均，再在特征维 D 上做卷积，
    最后切成 n_groups 个 token
    输入:  [B, T, D]
    输出:  [B, n_groups, emb_size]
    """
    def __init__(self, feat_dim=300, n_groups=20, emb_size=128, residual=True, res_scale_init=1.0, conv_layers=3):
        super().__init__()
        assert feat_dim % n_groups == 0
        self.feat_dim = feat_dim
        self.n_groups = n_groups
        self.group_dim = feat_dim // n_groups
        self.residual = residual

        kernels = [7, 5, 3, 3, 3, 3]
        layers = []
        in_ch = 1
        for layer_idx in range(max(1, conv_layers)):
            layers.extend(_conv1d_bn_elu(in_ch, 16, kernels[min(layer_idx, len(kernels) - 1)]))
            in_ch = 16
        self.feature_conv = nn.Sequential(*layers)

        self.proj = nn.Linear(16 * self.group_dim, emb_size)
        self.shortcut = nn.Linear(self.group_dim, emb_size) if residual else nn.Identity()
        self.res_scale = nn.Parameter(torch.tensor([res_scale_init], dtype=torch.float32)) if residual else None

    def forward(self, x):  # x: [B, T, D]
        x = x.mean(dim=1)                      # [B, D]
        res = x.view(x.size(0), self.n_groups, self.group_dim)
        x = x.unsqueeze(1)                                         # [B, 1, D]
        x = self.feature_conv(x)                                   # [B, 16, D]
        x = x.view(x.size(0), 16, self.n_groups, self.group_dim)   # [B,16,G,Dg]
        x = x.permute(0, 2, 1, 3).contiguous()                     # [B,G,16,Dg]
        x = x.view(x.size(0), self.n_groups, -1)                   # [B,G,16*Dg]
        x = self.proj(x)                                           # [B,G,emb]
        if self.residual:
            x = x + self.res_scale * self.shortcut(res)
        return x


class DBConformerFeat(nn.Module):
    def __init__(self, feat_dim=1830, n_windows=4, emb_size=128,
                 tem_depth=5, chn_depth=5, n_groups=20, n_classes=2,
                 input_proj_dim=0):
        super().__init__()
        inner_dim = input_proj_dim if input_proj_dim and input_proj_dim > 0 else feat_dim
        self.input_projection = nn.Identity() if inner_dim == feat_dim else nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, inner_dim),
            nn.ELU(),
        )
        self.layernorm = nn.LayerNorm(inner_dim)

        self.temporal_embedding = PatchEmbeddingTemporalFeat(
            in_dim=inner_dim,
            emb_size=emb_size,
            residual=args.branch_residual in [1, 2],
            res_scale_init=args.branch_res_scale_init,
            conv_layers=args.temporal_conv_layers
        )
        self.feature_embedding = PatchEmbeddingFeatureFeat(
            feat_dim=inner_dim,
            n_groups=n_groups,
            emb_size=emb_size,
            residual=args.branch_residual in [1, 3],
            res_scale_init=args.branch_res_scale_init,
            conv_layers=args.spatial_conv_layers
        )
        print("DBConformerFeat init n_windows =", n_windows)
        self.attention = QMultiHeadAttention(emb_size=inner_dim, num_heads=args.raw_attn_heads, dropout=args.raw_attn_dropout)
        self.attn_post = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, inner_dim),
            nn.ELU(),
            nn.Dropout(args.qattn_post_dropout)
        ) if args.qattn_post_ffn else nn.Identity()
        self.classifier = ClassificationHead(
            emb_size * 2, n_classes,
            hidden1=args.cls_hidden1,
            hidden2=args.cls_hidden2,
            drop1=args.cls_drop1,
            drop2=args.cls_drop2
        )
        self.out_dim = emb_size * 2
    def forward(self, x):  # x: [B, T, D] = [B, 4, 300]
        x = self.input_projection(x)
        x0 = x
        x = self.layernorm(x)
        x_attn = self.attention(x)
        x_attn = self.attn_post(x_attn)
        x = x_attn + x0

        x_temporal = self.temporal_embedding(x)
        x_spatial = self.feature_embedding(x)

       
        x_fused = torch.cat([x_temporal.mean(dim=1), x_spatial.mean(dim=1)], dim=-1)

        feat, out = self.classifier(x_fused)
        return feat, out



class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, hidden1=64, hidden2=32, drop1=0.5, drop2=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, hidden1),
            nn.ELU(),
            nn.Dropout(drop1),
            nn.Linear(hidden1, hidden2),
            nn.ELU(),
            nn.Dropout(drop2),
            nn.Linear(hidden2, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Discriminator(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            SLR_layer(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1).float()
        out = self.fc2(x)
        return out





class ExGAN():
    def __init__(self, args, nsub, fold):
        super(ExGAN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.batch_size = 144
        self.n_epochs = args.pretrain_epochs
        self.lr = args.lr_pretrain
        self.lr2 = args.lr_finetune
        self.b1 = 0.5
        self.b2 = 0.999
        self.radius = 10
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.n_windows = args.window_size
        
        self.model = DBConformerFeat(
         feat_dim=args.model_feat_dim,     # 300
         n_windows=self.n_windows,   # 4
         emb_size=args.emb_size,
         tem_depth=2,
         chn_depth=2,
         n_groups=args.n_groups,
         n_classes=2,
         input_proj_dim=args.input_proj_dim).float().cuda()
        self.num_source_domains = args.num_source_domains
        self.domain_Discriminator = Discriminator(
            in_dim=self.model.out_dim,
            n_classes=self.num_source_domains
        ).to(self.device).float()
        self.criterion = LabelSmooth(num_class=args.num_class).cuda()
        
    def schedule_lambda(self, epoch, total_epochs, max_lambda=0.6, k=5):
        p = epoch / total_epochs  # 归一化到 [0,1]
        return max_lambda * (2. / (1. + np.exp(-k * p)) - 1)


    def get_source_data(self, feature="de_LDS"):
        if self.args.dataset == "seed":
            datasets, dataset_test, X_subjects, Y_subjects = load_mi1(
                args,
                path=args.feature_path,
                n_windows=args.feature_windows,
                k=args.feature_k
            )
        return datasets, dataset_test, X_subjects, Y_subjects

    def get_source_data_for_fine(self, X, Y):
        if self.args.dataset == "seed":
            dset_loaders = fine_tuning_load_XY_MI(self.args, X, Y)
        return dset_loaders

    def test_suda(self, loader, model):
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.type(torch.FloatTensor).cuda()
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # 自动计算 62×5=310 [批次，3，310]
                labels = labels
                _, outputs = model(inputs.float())
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        _, predictions = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])
        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)
    
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)
    
        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        matrix = confusion_matrix(y_true, y_pred)
    
        return accuracy, f1, auc, matrix

    def _to_tensor(self, x, device, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=dtype)
        return x

    def train(self, fold):
        
        train_dataset, test_dataset, X, Y = self.get_source_data(feature="de_LDS")
    
        self.optimizer = torch.optim.SGD(
            list(self.model.parameters()) + list(self.domain_Discriminator.parameters()),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.args.weight_decay
        )
    
        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0
        epochs_acc = []
    
        B = self.args.batch_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        for e in range(self.n_epochs):
            self.model.train()
            self.domain_Discriminator.train()
    
            for i, data in enumerate(train_dataset):
                x_src = [data[f"Sx{idx}"] for idx in range(1, self.num_source_domains + 1)]
                y_src = [data[f"Sy{idx}"] for idx in range(1, self.num_source_domains + 1)]

                img = torch.cat(x_src, dim=0).to(device, non_blocking=True).float()
                label = torch.cat(y_src, dim=0).to(device, non_blocking=True).long()
    
                x_trg = data["Tx"].to(device, non_blocking=True).float()
                img = img.view(img.size(0), img.size(1), -1)
                x_trg = x_trg.view(x_trg.size(0), x_trg.size(1), -1)
                domain_label = torch.arange(self.num_source_domains, device=device, dtype=torch.long).repeat_interleave(B)
                tok, outputs = self.model(img)            # tok: [14B, feat], outputs: [14B, C]
                tok_target, outputs_target = self.model(x_trg)  # [B, feat], [B, C]
                pre_target = torch.softmax(outputs_target, dim=1)  # [B, C]
                tok_s = tok.view(self.num_source_domains, B, -1)
                lab_s = label.view(self.num_source_domains, B)
                tgt_tok_eq = tok_target          # [B, feat]
                tgt_prob_eq = pre_target         # [B, C]
    
                mmd_b_vals, mmd_t_vals = [], []
                for d in range(self.num_source_domains):
                    src_tok_d = tok_s[d]                 # [B, feat]
                    src_lab_d = lab_s[d].reshape(B, 1)   # [B, 1]
    
                    mb = utils.marginal(src_tok_d, tgt_tok_eq)
                    mt = utils.conditional(
                        src_tok_d,
                        tgt_tok_eq,
                        src_lab_d,
                        tgt_prob_eq,
                        0.5,
                        5,
                        None
                    )
                    mb = self._to_tensor(mb, outputs.device)
                    mt = self._to_tensor(mt, outputs.device)
                    mmd_b_vals.append(mb)
                    mmd_t_vals.append(mt)
                mmd_b_stack = torch.stack(mmd_b_vals)
                mmd_t_stack = torch.stack(mmd_t_vals)
                selected_domains = None
                if 0 < self.args.source_select_topk < mmd_b_stack.numel():
                    if self.args.source_select_metric == 'total':
                        source_metric = mmd_b_stack.detach().float() + mmd_t_stack.detach().float()
                    else:
                        source_metric = mmd_b_stack.detach().float()
                    selected_domains = torch.topk(
                        source_metric,
                        k=self.args.source_select_topk,
                        largest=False
                    ).indices
                    mmd_b_loss = mmd_b_stack[selected_domains].mean()
                    mmd_t_loss = mmd_t_stack[selected_domains].mean()
                else:
                    mmd_b_loss = mmd_b_stack.mean()
                    mmd_t_loss = mmd_t_stack.mean()
                MMD_loss = self.args.mmd_marginal_weight*mmd_b_loss + self.args.mmd_conditional_weight*mmd_t_loss
    
                lambda_adv = self.schedule_lambda(e, self.n_epochs, max_lambda=self.args.lambda_adv_max)
                if selected_domains is not None:
                    outputs_for_cls = outputs.view(self.num_source_domains, B, -1)[selected_domains].reshape(-1, outputs.size(-1))
                    labels_for_cls = lab_s[selected_domains].reshape(-1)
                    tok_for_adv = tok.view(self.num_source_domains, B, -1)[selected_domains].reshape(-1, tok.size(-1))
                    domain_label_for_adv = domain_label.view(self.num_source_domains, B)[selected_domains].reshape(-1)
                else:
                    outputs_for_cls = outputs
                    labels_for_cls = label
                    tok_for_adv = tok
                    domain_label_for_adv = domain_label
                features_s_Adver = Adver_network.ReverseLayerF.apply(tok_for_adv, lambda_adv)

                outputs_D = self.domain_Discriminator(features_s_Adver.float())
                Adver_domain_labels_loss = self.criterion(outputs_D, domain_label_for_adv)
                slc_loss = self.criterion(outputs_for_cls, labels_for_cls)
                loss = slc_loss + MMD_loss + Adver_domain_labels_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

            if (e + 1) % 1 == 0:
                start_test = True
                with torch.no_grad():        
                    self.model.eval()
        
                    for batch_idx, tar_data in enumerate(test_dataset):
                        Tx = tar_data['Tx']
                        Ty = tar_data['Ty']
                        Tx = Tx.float().cuda()
                        Tx = Tx.view(Tx.size(0), Tx.size(1), -1)  # 自动计算 62×5=310 [批次，3，310]
                        Tok, Cls = self.model(Tx)
                        if start_test:
                            all_output = Cls.float().cpu()
                            all_label = Ty.float()
                            start_test = False
                        else:
                            all_output = torch.cat((all_output, Cls.float().cpu()), 0)
                            all_label = torch.cat((all_label, Ty.float()), 0)
                        loss_test = self.criterion_cls(Cls.float().cpu(), Ty.long())
                torch.cuda.empty_cache()  # 清理GPU缓存
                y_pred = torch.max(all_output, 1)[1]
                acc = float((y_pred == all_label).cpu().numpy().astype(int).sum()) / float(all_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                epochs_acc.append(acc)
                print('Epoch:', e,
                      '  Train loss: %.4f' % loss.item(),
                      '  cls: %.4f' % slc_loss.detach().cpu().numpy(),
                      '  MMD: %.4f' % MMD_loss.item(),
                      '  adv: %.4f' % Adver_domain_labels_loss.detach().cpu().numpy(),
                      '  lambda_adv: %.4f' % lambda_adv,
                      '  Train acc: %.4f' % train_acc,
                      '  Test acc: %.4f' % acc)
             
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = Ty
                    Y_pred = y_pred

        averAcc = averAcc / num
        print('The average accuracy of n_epochs%d is:' %(e+1), averAcc)
        print('The best accuracy of n_epochs%d is:' %(e+1), bestAcc)
     
        return bestAcc, averAcc, Y_true, Y_pred, X, Y, self.model, epochs_acc


    def fine_tuning(self, args, X, Y, model):
        dset_loaders = self.get_source_data_for_fine(X, Y)
        parameter_model = model.parameters()
        self.optimizer = torch.optim.Adam(parameter_model, lr=self.lr2, betas=(self.b1, self.b2))

        len_train_source = len(dset_loaders["source"])
        len_train_target = len(dset_loaders["target"])
        final_acc = 0
        final_f1 = 0
        final_auc = 0
        final_mat = []

        # 新增：记录该受试者每一轮微调的测试结果
        iter_acc_list = []
        iter_f1_list = []
        iter_auc_list = []

        for i in range(args.max_iter2):
            if i % 1 == 0:
                with torch.no_grad():
                    model.eval()
                    eval_acc, eval_f1, eval_auc, eval_mat = self.test_suda(dset_loaders, model)

                    # 记录当前这一轮的结果
                    iter_acc_list.append(eval_acc)
                    iter_f1_list.append(eval_f1)
                    iter_auc_list.append(eval_auc)

                    final_acc = eval_acc
                    final_f1 = eval_f1
                    final_auc = eval_auc
                    final_mat = eval_mat

                    if i == 0:
                        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(
                            i, eval_acc, eval_f1, eval_auc
                        )
                    else:
                        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f} \t loss: {:.4f}".format(
                            i, eval_acc, eval_f1, eval_auc, total_loss.item()
                        )
                    print(log_str)

            model.train()
            if i % len_train_source == 0:
                iter_source = iter(dset_loaders["source"])
            if i % len_train_target == 0:
                iter_target = iter(dset_loaders["target"])

            inputs_source_, labels_source = next(iter_source)
            inputs_target_, _ = next(iter_target)

            inputs_source_ = inputs_source_.type(torch.FloatTensor)
            labels_source = labels_source.type(torch.LongTensor)
            inputs_target_ = inputs_target_.type(torch.FloatTensor)

            inputs_source, labels_source = inputs_source_.cuda(), labels_source.cuda()
            inputs_target = inputs_target_.cuda()

            inputs_source = inputs_source.view(inputs_source.size(0), inputs_source.size(1), -1)
            inputs_target = inputs_target.view(inputs_target.size(0), inputs_target.size(1), -1)

            _, outputs_source = model(inputs_source)
            _, outputs_target = model(inputs_target)

            classifier_loss = self.criterion_cls(outputs_source, labels_source.flatten())
            total_loss = classifier_loss

            probs_target = torch.softmax(outputs_target, dim=1)
            if args.ft_entropy_weight > 0:
                entropy_loss = -(probs_target * torch.log(probs_target.clamp_min(1e-8))).sum(dim=1).mean()
                total_loss = total_loss + args.ft_entropy_weight * entropy_loss

            pseudo_conf, pseudo_label = probs_target.max(dim=1)
            pseudo_mask = pseudo_conf >= args.ft_pseudo_threshold
            if pseudo_mask.any():
                pseudo_loss = self.criterion_cls(outputs_target[pseudo_mask], pseudo_label[pseudo_mask])
                total_loss = total_loss + args.ft_pseudo_weight * pseudo_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            model.eval()
            final_acc, final_f1, final_auc, final_mat = self.test_suda(dset_loaders, model)
            iter_acc_list.append(final_acc)
            iter_f1_list.append(final_f1)
            iter_auc_list.append(final_auc)
            print("final_after_update: accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(
                final_acc, final_f1, final_auc
            ))

        return final_acc, final_f1, final_auc, final_mat, model, iter_acc_list, iter_f1_list, iter_auc_list

    def export_test_predictions(self, args, X, Y, model, writer):
        dset_loaders = self.get_source_data_for_fine(X, Y)
        model.eval()
        sample_index = 0
        with torch.no_grad():
            for inputs, labels in dset_loaders["test"]:
                inputs = inputs.type(torch.FloatTensor).cuda()
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
                _, outputs = model(inputs.float())
                preds = torch.argmax(outputs.float().cpu(), dim=1).numpy().astype(int)
                y_true = labels.cpu().numpy().astype(int)
                for yt, yp in zip(y_true, preds):
                    writer.writerow({
                        "dataset": "2003004",
                        "seed": int(args.seed),
                        "target": int(args.target),
                        "sample_index": int(sample_index),
                        "y_true": int(yt),
                        "y_pred": int(yp),
                    })
                    sample_index += 1

def run_single_seed(args):
    pre_train = []
    tuning = []
    result_write = open(f"./snapshot_seed{args.seed}.txt", "w")

    total_acc = []

    # 新增：保存 9 个受试者在每个 fine-tuning iter 上的结果
    all_subject_ft_acc = []
    all_subject_ft_f1 = []
    all_subject_ft_auc = []

    pred_fh = None
    pred_writer = None
    if args.prediction_csv:
        pred_dir = os.path.dirname(args.prediction_csv)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        pred_fh = open(args.prediction_csv, "w", newline="", encoding="utf-8")
        pred_writer = csv.DictWriter(
            pred_fh,
            fieldnames=["dataset", "seed", "target", "sample_index", "y_true", "y_pred"]
        )
        pred_writer.writeheader()

    target_list = _parse_targets(args.targets)
    for i, target in enumerate(target_list):
        args.target = target
        seed_n = args.seed

        result_write.write('--------------------------------------------------\n')
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Target subject %d' % args.target)
        result_write.write('Target subject ' + str(args.target) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")

        ba = 0
        aa = 0
        pre_train_Acc = 0
        averAcc = 0

        exgan = ExGAN(args, i + 1, 1)

        ba, aa, _, _, X, Y, model, epochs_acc = exgan.train(1)
        total_acc.append(epochs_acc)

        final_acc, final_f1, final_auc, final_mat, model, iter_acc_list, iter_f1_list, iter_auc_list = exgan.fine_tuning(args, X, Y, model)

        if pred_writer is not None:
            exgan.export_test_predictions(args, X, Y, model, pred_writer)
            pred_fh.flush()

        if args.checkpoint_dir and str(args.checkpoint_dir).lower() not in ["none", "null", "false"]:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.trial_id}_target_{args.target:02d}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "target": int(args.target),
                "subject_loop_index": int(i + 1),
                "seed": int(seed_n),
                "pretrain_best_acc": float(ba),
                "finetune_final_acc": float(final_acc),
                "finetune_final_f1": float(final_f1),
                "finetune_final_auc": float(final_auc),
                "feature_path": args.feature_path,
                "feature_windows": int(args.feature_windows),
                "feature_k": int(args.feature_k),
                "model_feat_dim": int(args.model_feat_dim),
                "n_groups": int(args.n_groups),
                "raw_attn_heads": int(args.raw_attn_heads),
                "branch_residual": int(args.branch_residual),
                "branch_res_scale_init": float(args.branch_res_scale_init),
                "max_iter2": int(args.max_iter2),
                "ft_entropy_weight": float(args.ft_entropy_weight),
                "trial_id": args.trial_id,
            }, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

        # 新增：收集该受试者的整条 fine-tuning 曲线
        all_subject_ft_acc.append(iter_acc_list)
        all_subject_ft_f1.append(iter_f1_list)
        all_subject_ft_auc.append(iter_auc_list)

        result_write.write('pre_training acc is:' + str(ba) + "\n")
        result_write.write('fine_tuning acc is:' + str(final_acc) + "\n")

        pre_train_Acc = ba
        tuning_Acc = final_acc

        pre_train.append(pre_train_Acc)
        tuning.append(tuning_Acc)

        print('pre_training acc is:', pre_train)
        print('fine_tuning acc is:', tuning)

        del exgan, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # =======================
    # 预训练平均最好 epoch
    # =======================
    total_acc = np.array(total_acc)
    epoch_mean_acc = np.mean(total_acc, axis=0)
    print(f"所有epochs的平均准确率: {epoch_mean_acc}")

    best_epoch = np.argmax(epoch_mean_acc) + 1
    best_epoch_acc = epoch_mean_acc[best_epoch - 1]
    print(f"\n最佳epoch为: {best_epoch}，对应平均准确率 = {best_epoch_acc:.4f}")

    # =======================
    # 微调平均最好 iter
    # =======================
    all_subject_ft_acc = np.array(all_subject_ft_acc)   # [9, max_iter2]
    all_subject_ft_f1 = np.array(all_subject_ft_f1)
    all_subject_ft_auc = np.array(all_subject_ft_auc)

    mean_ft_acc = np.mean(all_subject_ft_acc, axis=0)   # [max_iter2]
    mean_ft_f1 = np.mean(all_subject_ft_f1, axis=0)
    mean_ft_auc = np.mean(all_subject_ft_auc, axis=0)

    best_ft_iter = np.argmax(mean_ft_acc) + 1
    best_ft_acc = mean_ft_acc[best_ft_iter - 1]
    best_ft_f1 = mean_ft_f1[best_ft_iter - 1]
    best_ft_auc = mean_ft_auc[best_ft_iter - 1]
    best_ft_idx = best_ft_iter - 1  # 数组索引
    
    # 取出9个受试者在“全局最佳微调iter”上的结果
    subject_best_iter_acc = all_subject_ft_acc[:, best_ft_idx]
    subject_best_iter_f1  = all_subject_ft_f1[:, best_ft_idx]
    subject_best_iter_auc = all_subject_ft_auc[:, best_ft_idx]
    print("\n================= Fine-tuning平均结果 =================")
    print(f"每个微调iter在9个受试者上的平均准确率: {mean_ft_acc}")
    print(f"最佳微调iter为: {best_ft_iter}")
    print(f"该iter的平均准确率 = {best_ft_acc:.4f}")
    print(f"该iter的平均F1 = {best_ft_f1:.4f}")
    print(f"该iter的平均AUC = {best_ft_auc:.4f}")
    print("\n================= 每位受试者在最佳微调iter上的结果 =================")
    for subj in range(len(target_list)):
        print(
            f"Target {target_list[subj]}: "
            f"acc = {subject_best_iter_acc[subj]:.4f}, "
            f"f1 = {subject_best_iter_f1[subj]:.4f}, "
            f"auc = {subject_best_iter_auc[subj]:.4f}"
        )
        pre_ave = sum(pre_train) / len(pre_train)
        tuning_ave = sum(tuning) / len(tuning)

    print('------------------------pre-training result--------------------------', pre_train)
    print('------------------------fin-tuning result--------------------------', tuning)
    print('------------------------pre-training average result--------------------------', pre_ave)
    print('------------------------fin-tuning average result--------------------------', tuning_ave)

    result_write.write('--------------------------------------------------\n')
    result_write.write(f"All accuracy is: {pre_train}\n")
    result_write.write(f"All subject Aver accuracy is: {tuning}\n")
    result_write.write(f"Best fine-tuning iter across 9 subjects: {best_ft_iter}\n")
    result_write.write(f"Best fine-tuning mean acc: {best_ft_acc:.4f}\n")
    result_write.write(f"Best fine-tuning mean f1: {best_ft_f1:.4f}\n")
    result_write.write(f"Best fine-tuning mean auc: {best_ft_auc:.4f}\n")
    result_write.close()
    if pred_fh is not None:
        pred_fh.close()

    return {
        "seed": int(args.seed),
        "pre_train": pre_train,
        "fine_tuning": tuning,
        "pretrain_mean": float(pre_ave),
        "finetune_mean": float(tuning_ave),
        "best_ft_iter": int(best_ft_iter),
        "best_ft_acc": float(best_ft_acc),
        "best_ft_f1": float(best_ft_f1),
        "best_ft_auc": float(best_ft_auc),
    }


def _parse_seeds(seed_text, fallback_seed):
    if not seed_text:
        return [int(fallback_seed)]
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def _parse_targets(target_text):
    if not target_text:
        return list(range(7, 0, -1))
    return [int(item.strip()) for item in target_text.split(",") if item.strip()]


def main(args):
    seeds = _parse_seeds(args.seeds, args.seed)
    if len(seeds) == 1:
        args.seed = seeds[0]
        return run_single_seed(args)

    base_trial_id = args.trial_id
    base_prediction_csv = args.prediction_csv
    seed_results = []
    for seed in seeds:
        seed_args = copy.deepcopy(args)
        seed_args.seed = int(seed)
        seed_args.trial_id = f"{base_trial_id}_seed{seed}"
        if base_prediction_csv:
            root, ext = os.path.splitext(base_prediction_csv)
            seed_args.prediction_csv = f"{root}_seed{seed}{ext or '.csv'}"

        print(f"\n================= Running seed {seed} =================")
        seed_results.append(run_single_seed(seed_args))

    finetune_means = np.asarray([item["finetune_mean"] for item in seed_results], dtype=np.float64)
    pretrain_means = np.asarray([item["pretrain_mean"] for item in seed_results], dtype=np.float64)

    print("\n================= 5-seed summary =================")
    print("Seeds:", seeds)
    print("Pre-training means:", pretrain_means.tolist())
    print("Fine-tuning means:", finetune_means.tolist())
    print("Pre-training mean +/- std: %.4f +/- %.4f" % (
        float(pretrain_means.mean()),
        float(pretrain_means.std(ddof=1)) if len(seeds) > 1 else 0.0,
    ))
    print("Fine-tuning mean +/- std: %.4f +/- %.4f" % (
        float(finetune_means.mean()),
        float(finetune_means.std(ddof=1)) if len(seeds) > 1 else 0.0,
    ))

    with open("five_seed_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "pretrain_mean",
                "finetune_mean",
                "best_ft_iter",
                "best_ft_acc",
                "best_ft_f1",
                "best_ft_auc",
            ],
        )
        writer.writeheader()
        for item in seed_results:
            writer.writerow({
                "seed": item["seed"],
                "pretrain_mean": item["pretrain_mean"],
                "finetune_mean": item["finetune_mean"],
                "best_ft_iter": item["best_ft_iter"],
                "best_ft_acc": item["best_ft_acc"],
                "best_ft_f1": item["best_ft_f1"],
                "best_ft_auc": item["best_ft_auc"],
            })

    return {
        "seeds": seeds,
        "pretrain_mean": float(pretrain_means.mean()),
        "pretrain_std": float(pretrain_means.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "finetune_mean": float(finetune_means.mean()),
        "finetune_std": float(finetune_means.std(ddof=1)) if len(seeds) > 1 else 0.0,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-DMDA 2003004 cross-subject EEG classification")
    parser.add_argument("--dataset", type=str, default="seed")
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--targets", type=str, default="5,4,3,2,1", help="Comma-separated target subjects.")
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, e.g. 1,2,3,4,5")
    parser.add_argument("--num_source_domains", type=int, default=4)

    parser.add_argument("--feature_path", type=str, default=r"features\2003004")
    parser.add_argument("--feature_file_prefix", type=str, default="2003004")
    parser.add_argument("--feature_mode", type=str, default="spat")
    parser.add_argument("--feature_windows", type=int, default=4)
    parser.add_argument("--feature_shape_windows", type=int, default=4)
    parser.add_argument("--feature_k", type=int, default=25)
    parser.add_argument("--feature_use_gfk", action="store_true", default=True)
    parser.add_argument("--no_feature_use_gfk", action="store_false", dest="feature_use_gfk")
    parser.add_argument("--feature_path_extra", type=str, default="")
    parser.add_argument("--feature_mode_extra", type=str, default="spat")
    parser.add_argument("--feature_windows_extra", type=int, default=4)
    parser.add_argument("--feature_shape_windows_extra", type=int, default=4)
    parser.add_argument("--feature_k_extra", type=int, default=25)
    parser.add_argument("--feature_use_gfk_extra", action="store_true", default=True)
    parser.add_argument("--no_feature_use_gfk_extra", action="store_false", dest="feature_use_gfk_extra")

    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--batch_size_fine", type=int, default=144)
    parser.add_argument("--pretrain_epochs", type=int, default=40)
    parser.add_argument("--max_iter2", type=int, default=20)
    parser.add_argument("--lr_pretrain", type=float, default=0.004)
    parser.add_argument("--lr_finetune", type=float, default=0.00008)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--mmd_marginal_weight", type=float, default=0.4)
    parser.add_argument("--mmd_conditional_weight", type=float, default=0.4)
    parser.add_argument("--lambda_adv_max", type=float, default=0.5)
    parser.add_argument("--source_select_topk", type=int, default=2)
    parser.add_argument("--source_select_metric", type=str, default="total", choices=["marginal", "total"])
    parser.add_argument("--ft_pseudo_weight", type=float, default=1.5)
    parser.add_argument("--ft_pseudo_threshold", type=float, default=0.85)
    parser.add_argument("--ft_entropy_weight", type=float, default=0.0)

    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--model_feat_dim", type=int, default=512)
    parser.add_argument("--input_proj_dim", type=int, default=0)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--n_groups", type=int, default=32)
    parser.add_argument("--raw_attn_heads", type=int, default=8)
    parser.add_argument("--raw_attn_dropout", type=float, default=0.0)
    parser.add_argument("--qattn_scale", type=str, default="emb", choices=["emb", "head"])
    parser.add_argument("--qattn_post_ffn", type=int, default=0)
    parser.add_argument("--qattn_post_dropout", type=float, default=0.0)
    parser.add_argument("--branch_residual", type=int, default=0)
    parser.add_argument("--branch_res_scale_init", type=float, default=1.0)
    parser.add_argument("--temporal_conv_layers", type=int, default=1)
    parser.add_argument("--spatial_conv_layers", type=int, default=3)
    parser.add_argument("--cls_hidden1", type=int, default=256)
    parser.add_argument("--cls_hidden2", type=int, default=128)
    parser.add_argument("--cls_drop1", type=float, default=0.1)
    parser.add_argument("--cls_drop2", type=float, default=0.05)

    parser.add_argument("--checkpoint_dir", type=str, default=r"checkpoints\2003004_lraugqk_logeuclid_compact512_seed1")
    parser.add_argument("--trial_id", type=str, default="2003004_lraugqk_logeuclid_compact512_seed1")
    parser.add_argument("--prediction_csv", type=str, default=r"logs\2003004_lraugqk_logeuclid_compact512_seed1_predictions.csv")

    args = parser.parse_args()

    main(args)
