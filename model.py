import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import  GraphConv
import numpy as np, itertools, random, copy, math
from model_GCN import GCN_2Layers, GCNLayer1, GCNII, TextCNN
from model_hyper import emoGen,emoAgent
from nncore.nn import (build_model, FeedForwardNetwork, MultiHeadAttention,
                       Parameter, build_norm_layer)
def print_grad(grad):
    print('the grad is', grad[2][0:5])
    return grad

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits,-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1,2,0)
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha



class GRUModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(GRUModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        
    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()  
        )
        
    def forward(self, x, residual):
        combined = torch.cat([x, residual], dim=-1)
        gate_weight = self.gate(combined)
        return (x + gate_weight * residual)  
class Global(nn.Module):
    def __init__(self,
                 dims,
                 heads=1,
                 ratio=2,
                 p=0.1,
                 layer=1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='gelu'),
                 mem_size=20):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.mem_size = mem_size
        self.memory_net = MemoryModule(dims)  
        self.gated_residual = GatedResidual(dims)

    def _process_modality(self, da,db,dt,dg):
        a,b,t,g= self.memory_net(da,db,dt,dg)
        a = a.transpose(1, 2)  + da
        b = b.transpose(1, 2) + db
        t = t.transpose(1, 2) + dt
        g = g.transpose(1, 2) + dg
        a = self.gated_residual(da.transpose(1, 2), a.transpose(1, 2))
        b = self.gated_residual(db.transpose(1, 2), b.transpose(1, 2))
        t = self.gated_residual(dt.transpose(1, 2), t.transpose(1, 2))
        g = self.gated_residual(dg.transpose(1, 2), g.transpose(1, 2))
        
        return a,b,t,g

    def forward(self, da, db, dt, dg):
        for i in range(len(self.transencoders['a'])):
            da, db, dt, dg = self._process_modality(da, db, dt, dg)
            da = da.transpose(1, 2)
            db = db.transpose(1, 2)
            dt = dt.transpose(1, 2)
            dg = dg.transpose(1, 2)
        return da, db, dt, dg

class MemoryModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.feature_enhancer = nn.Sequential(
            nn.Conv1d(dim, dim*2, kernel_size=1), 
            nn.ReLU(),
            nn.Conv1d(dim*2, dim, kernel_size=1),
            
        )
        self.ln = nn.LayerNorm(dim)
    def _temporal_pooling(self, x):
        return x.mean(2).unsqueeze(2)


    def forward(self, a,b,t,g):
        a = self.ln(self.feature_enhancer(a).transpose(1, 2))  
        b = self.ln(self.feature_enhancer(b).transpose(1, 2))
        t = self.ln(self.feature_enhancer(t).transpose(1, 2))
        g = self.ln(self.feature_enhancer(g).transpose(1, 2))


        xx = self._temporal_pooling(a)  
        yy = self._temporal_pooling(b)
        zz = self._temporal_pooling(t)
        gg = self._temporal_pooling(g)
        visual_sim = torch.matmul(a.transpose(1, 2), xx) 
        text_sim = torch.matmul(a.transpose(1, 2), yy) 
        audio_sim = torch.matmul(a.transpose(1, 2),zz) 
        agent_sim = torch.matmul(a.transpose(1, 2),gg) 
        

        visual_att = torch.softmax(visual_sim, dim=-1) 
        text_att = torch.softmax(text_sim, dim=-1)
        audio_att = torch.softmax(audio_sim, dim=-1)
        agent_attn = torch.softmax(agent_sim, dim=-1)
        a = torch.matmul(visual_att, xx.transpose(1, 2)).transpose(1, 2) + torch.matmul(text_att, yy.transpose(1, 2)).transpose(1, 2) + torch.matmul(audio_att, zz.transpose(1, 2)).transpose(1, 2) + torch.matmul(agent_attn, gg.transpose(1, 2)).transpose(1, 2)
        
        visual_sim = torch.matmul(b.transpose(1, 2), xx)  
        text_sim = torch.matmul(b.transpose(1, 2), yy) 
        audio_sim = torch.matmul(b.transpose(1, 2),zz) 
        agent_sim = torch.matmul(b.transpose(1, 2),gg) 
        

        visual_att = torch.softmax(visual_sim, dim=-1) 
        text_att = torch.softmax(text_sim, dim=-1)
        audio_att = torch.softmax(audio_sim, dim=-1)
        agent_attn = torch.softmax(agent_sim, dim=-1)
        b = torch.matmul(visual_att, xx.transpose(1, 2)).transpose(1, 2) +torch.matmul(text_att, yy.transpose(1, 2)).transpose(1, 2) +torch.matmul(audio_att, zz.transpose(1, 2)).transpose(1, 2) +torch.matmul(agent_attn, gg.transpose(1, 2)).transpose(1, 2)
        
        
        visual_sim = torch.matmul(t.transpose(1, 2), xx)  
        text_sim = torch.matmul(t.transpose(1, 2), yy)  
        audio_sim = torch.matmul(t.transpose(1, 2),zz) 
        agent_sim = torch.matmul(t.transpose(1, 2),gg) 
        
        visual_att = torch.softmax(visual_sim, dim=-1) 
        text_att = torch.softmax(text_sim, dim=-1)
        audio_att = torch.softmax(audio_sim, dim=-1)
        agent_attn = torch.softmax(agent_sim, dim=-1)
        t = torch.matmul(visual_att, xx.transpose(1, 2)).transpose(1, 2) +torch.matmul(text_att, yy.transpose(1, 2)).transpose(1, 2) +torch.matmul(audio_att, zz.transpose(1, 2)).transpose(1, 2) +torch.matmul(agent_attn, gg.transpose(1, 2)).transpose(1, 2)
        
        visual_sim = torch.matmul(g.transpose(1, 2), xx)  
        text_sim = torch.matmul(g.transpose(1, 2), yy)  
        audio_sim = torch.matmul(g.transpose(1, 2),zz) 
        agent_sim = torch.matmul(g.transpose(1, 2),gg) 
        
        visual_att = torch.softmax(visual_sim, dim=-1) 
        text_att = torch.softmax(text_sim, dim=-1)
        audio_att = torch.softmax(audio_sim, dim=-1)
        agent_attn = torch.softmax(agent_sim, dim=-1)
        g = torch.matmul(visual_att, xx.transpose(1, 2)).transpose(1, 2) +torch.matmul(text_att, yy.transpose(1, 2)).transpose(1, 2) +torch.matmul(audio_att, zz.transpose(1, 2)).transpose(1, 2) +torch.matmul(agent_attn, gg.transpose(1, 2)).transpose(1, 2)
                
        return a,b,t,g

class Model(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, n_speakers, max_seq_len, window_past, window_future, 
                 n_classes=7, listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5, nodal_attention=True, avec=False, 
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6, graph_construct='direct', use_GCN=False,use_residue=True,
                 dynamic_edge_w=False,D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False,Deep_GCN_nlayers = 64, dataset='IEMOCAP',
                 use_speaker=True, use_modal=False, norm='LN2', num_L = 3, num_K = 4):
        
        super(Model, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type=graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

        self.normLNa = nn.LayerNorm(1024, elementwise_affine=True)
        self.normLNb = nn.LayerNorm(1024, elementwise_affine=True)
        self.normLNc = nn.LayerNorm(1024, elementwise_affine=True)
        self.normLNd = nn.LayerNorm(1024, elementwise_affine=True)
        self.norm_strategy = norm
        if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.use_bert_seq = False
        self.dataset = dataset

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    if self.use_bert_seq:
                        self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                    else:
                        self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            #self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                if self.av_using_lstm:
                    self.gru_a = nn.GRU(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                if self.av_using_lstm:
                    self.gru_v = nn.GRU(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                hidden_l = D_g
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.gru_l = nn.GRU(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'Transformer':
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                self.trans_a = nn.TransformerEncoderLayer(d_model=hidden_a, nhead=4)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                self.trans_v = nn.TransformerEncoderLayer(d_model=hidden_v, nhead=4)
            if 'l' in self.modals:
                hidden_l = D_g
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.trans_l = nn.TransformerEncoderLayer(d_model=hidden_l, nhead=4)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError 



        if self.graph_type=='hyper':
            self.EA_fusion= emoAgent(a_dim=D_g, v_dim=D_g, l_dim=D_g, n_dim=D_g, nlayers=64, nhidden=graph_hidden_size, nclass=n_classes, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals, use_speaker=self.use_speaker, use_modal=self.use_modal, num_L=num_L, num_K=num_K)
            print("construct emotion fusion")
            self.emotion_modality = emoGen(a_dim=D_g, v_dim=D_g, l_dim=D_g, n_dim=D_g, nlayers=64, nhidden=graph_hidden_size, nclass=n_classes, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals, use_speaker=self.use_speaker, use_modal=self.use_modal, num_L=num_L, num_K=num_K)
            print("construct emotion_modality")
        elif self.graph_type=='None':
            if not self.multi_modal:
                self.graph_net = nn.Linear(2*D_e, n_classes)
            else:
                if 'a' in self.modals:
                    self.graph_net_a = nn.Linear(2*D_e, graph_hidden_size)
                if 'v' in self.modals:
                    self.graph_net_v = nn.Linear(2*D_e, graph_hidden_size)
                if 'l' in self.modals:
                    self.graph_net_l = nn.Linear(2*D_e, graph_hidden_size)
            print("construct Bi-LSTM")
        else:
            print("There are no such kind of graph")

        edge_type_mapping = {} 
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping
        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            elif self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size), n_classes)
                else:
                    self.smax_fc = nn.Linear(graph_hidden_size*(len(self.modals)+1), n_classes)
                    
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)
        self.Global = Global(dims=graph_hidden_size)

        self.gobal = MultiHeadSelfAttention(graph_hidden_size, graph_hidden_size, num_heads = 2) 
        self.fc = nn.Linear(graph_hidden_size * 8, graph_hidden_size * 2)  
    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):
        #=============roberta features
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()
        if self.norm_strategy == 'LN':
            r1 = self.normLNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normLNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normLNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normLNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        elif self.norm_strategy == 'BN':
            r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        elif self.norm_strategy == 'LN2':
            norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
            r1 = norm2(r1.transpose(0, 1)).transpose(0, 1)
            r2 = norm2(r2.transpose(0, 1)).transpose(0, 1)
            r3 = norm2(r3.transpose(0, 1)).transpose(0, 1)
            r4 = norm2(r4.transpose(0, 1)).transpose(0, 1)
        else:
            pass
        U = r1 + r2 + r3 + r4
        if self.base_model == 'LSTM':
            if not self.multi_modal:
                U = self.linear_(U)
                emotions, hidden = self.lstm(U)
            else:
                if 'a' in self.modals:
                    U_a = self.linear_a(U_a)
                    if self.av_using_lstm:
                        emotions_a, hidden_a = self.lstm_a(U_a)
                    else:
                        emotions_a = U_a
                if 'v' in self.modals:
                    U_v = self.linear_v(U_v)
                    if self.av_using_lstm:
                        emotions_v, hidden_v = self.lstm_v(U_v)
                    else:
                        emotions_v = U_v
                if 'l' in self.modals:
                    if self.use_bert_seq:
                        U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                        U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
                    else:
                        U = self.linear_l(U)
                    emotions_l, hidden_l = self.lstm_l(U)

        elif self.base_model == 'GRU':
            #emotions, hidden = self.gru(U)
            if 'a' in self.modals:
                U_a = self.linear_a(U_a)
                if self.av_using_lstm:
                    emotions_a, hidden_a = self.gru_a(U_a)
                else:
                    emotions_a = U_a
            if 'v' in self.modals:
                U_v = self.linear_v(U_v)
                if self.av_using_lstm:
                    emotions_v, hidden_v = self.gru_v(U_v)
                else:
                    emotions_v = U_v
            if 'l' in self.modals:
                if self.use_bert_seq:
                    U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                    U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
                else:
                    if self.dataset=='MELD':
                        pass
                    else:
                        U = self.linear_l(U)
                #self.gru_l.flatten_parameters()
                emotions_l, hidden_l = self.gru_l(U)

        elif self.base_model == 'Transformer':
            if 'a' in self.modals:
                U_a = self.linear_a(U_a)
                if self.av_using_lstm:
                    emotions_a = self.trans_a(U_a)
                else:
                    emotions_a = U_a
            if 'v' in self.modals:
                U_v = self.linear_v(U_v)
                if self.av_using_lstm:
                    emotions_v = self.trans_v(U_v)
                else:
                    emotions_v = U_v
            if 'l' in self.modals:
                if self.use_bert_seq:
                    U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                    U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
                else:
                    U = self.linear_l(U)
                emotions_l = self.trans_l(U)
        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        if not self.multi_modal:
            features, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
            else:
                features_a = []
            if 'v' in self.modals:
                features_v, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []
            if 'l' in self.modals:
                features_l, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            else:
                features_l = []
        if self.graph_type=='GCN3' or self.graph_type=='DeepGCN':
            if self.use_topic:
                topicLabel = [] 
            else:
                topicLabel = []
            if not self.multi_modal:
                log_prob = self.graph_net(features, seq_lengths, qmask)
            else:
                emotions_a = self.graph_net_a(features_a, seq_lengths, qmask) if 'a' in self.modals else []
                emotions_v = self.graph_net_v(features_v, seq_lengths, qmask) if 'v' in self.modals else []
                emotions_l = self.graph_net_l(features_l, seq_lengths, qmask) if 'l' in self.modals else []

                if self.att_type == 'concat_subsequently':                
                    emotions = []
                    if len(emotions_a) != 0:
                        emotions.append(emotions_a)
                    if len(emotions_v) != 0:
                        emotions.append(emotions_v)
                    if len(emotions_l) != 0:
                        emotions.append(emotions_l)
                    emotions_feat = torch.cat(emotions, dim=-1)
                else:
                    print("There is no such attention mechnism")

                emotions_feat = self.dropout_(emotions_feat)
                emotions_feat = nn.ReLU()(emotions_feat)
                log_prob = F.log_softmax(self.hidfc(self.smax_fc(emotions_feat)), 1)
        elif self.graph_type=='hyper':
            agent = self.emotion_modality(features_a, features_v, features_l, seq_lengths, qmask, epoch)

            emotions_feat = torch.cat((features_a, features_v, features_l, agent),dim=-1)
            position = emotions_feat.shape[1]//4
            maxlength = max(seq_lengths)
            a = torch.zeros(16, 512, maxlength, device="cuda" ) # 16 = batch_size
            b = torch.zeros(16, 512, maxlength, device="cuda" )
            c = torch.zeros(16, 512, maxlength, device="cuda" )
            d = torch.zeros(16, 512, maxlength, device="cuda" )
            start_idx = 0
            for i, length in enumerate(seq_lengths):
                tensor = emotions_feat[start_idx:start_idx + length, :]
                start_idx = start_idx +length
                chunks = torch.chunk(tensor, 4, dim=1)
                a[i, :, :length] = chunks[0].T
                b[i, :, :length] = chunks[1].T
                c[i, :, :length] = chunks[2].T
                d[i, :, :length] = chunks[3].T
            emotions_a, emotions_v, emotions_l, emotions_g = self.Global(a,b,c,d)
            emotions_a = emotions_a.permute(0, 2, 1)
            emotions_v = emotions_v.permute(0, 2, 1)
            emotions_l = emotions_l.permute(0, 2, 1)
            emotions_g = emotions_g.permute(0, 2, 1)
            start_idx = 0
            output_a=None
            for i, length in enumerate(seq_lengths):
                a_part = emotions_a[i, :length, : ]
                b_part = emotions_v[i, :length, : ]
                c_part = emotions_l[i, :length, : ]
                d_part = emotions_g[i, :length, : ]
                if output_a==None: 
                    output_a = a_part
                    output_v = b_part
                    output_l = c_part
                    output_g = d_part
                else:
                    output_a = torch.cat((output_a, a_part),dim=0)
                    output_v = torch.cat((output_v, b_part),dim=0)
                    output_l = torch.cat((output_l, c_part),dim=0)
                    output_g = torch.cat((output_g, d_part),dim=0)
                start_idx = start_idx +1
            emotions_feat = self.EA_fusion(output_a, output_v, output_l, output_g, seq_lengths, qmask, epoch)
            emotions_feat = self.dropout_(emotions_feat)
            emotions_feat = nn.ReLU()(emotions_feat)
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        else:
            print("There are no such kind of graph")        
        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths, emotions_feat
