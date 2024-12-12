from torch import nn
import torch
import os
import json

from data.fwf_dataset import get_label_structure



class FGFeatNetwork (nn.Module):        
    def __init__(self,
                 cfg,
                 num_input_feats,
                 label_structure = {'labels_0':3, 'labels_1':10, 'labels_2':12, 'labels_3':18},
                 dropout_prob = 0.2,
                 global_constraint = False,
                 ):
        super(FGFeatNetwork, self).__init__()
        
        self.label_structure = get_label_structure(cfg)
        self.global_constraint = global_constraint
        # Pointwise feats
        self.mlp1 = nn.Sequential(
            nn.Linear(num_input_feats, 64), nn.ReLU(), nn.Dropout(p=dropout_prob),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(p=dropout_prob),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=dropout_prob),
        )
        self.wf_conv = nn.Sequential(
            # FIXME: check conv and maxpool kernel sizes and strides
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3), nn.RReLU(),nn.BatchNorm1d(16), nn.MaxPool1d(2), # activations 32 -> 16
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3), nn.RReLU(),nn.BatchNorm1d(32), nn.MaxPool1d(2),# activations 16 -> 8
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3), nn.RReLU(),nn.BatchNorm1d(32), nn.MaxPool1d(2),# activations 8 -> 4
            # after concat. activation shape = 4 * 64 = 256
        )
        # TODO: Try out inverse bottleneck?
        # TODO: Network size might be an overkill / unfeasable for the task
        
        # MLP after concat with WFM feats
        self.mlp2 = nn.Sequential(
            nn.Linear(192, 512), nn.ReLU(), nn.Dropout(p=dropout_prob), 
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=dropout_prob),
        )
        
        # decoder
        self.mlp3 = nn.Sequential(
            nn.Linear(704, 512), nn.ReLU(), nn.Dropout(p=dropout_prob),  # joined shape (point + neibors)
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=dropout_prob),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=dropout_prob),
        )
        
        # classifier
        self.classifier = nn.ModuleDict({k:nn.Linear(128,v) for k,v in self.label_structure.items()})
        
    def forward(self, x):
        # handle neibor features
        pw_feats_neib = self.mlp1(x['features_neibors'])
        batch, neibor, signal_length = x['wfm_neibors'].shape
        wf_feats_neib = self.wf_conv(x['wfm_neibors'].view((batch * neibor, 1, signal_length))) # batch, neibor, signal_length -> batch * neibor, 1,  signal_length
        wf_feats_neib = wf_feats_neib.view((batch, neibor,-1))
        joined_feats_neib = torch.cat([pw_feats_neib, wf_feats_neib], dim=-1)
        
        # handle the point features
        pw_feats_point = self.mlp1(x['features_point'])
        batch, neibor, signal_length = x['wfm_point'].shape
        wf_feats_point = self.wf_conv(x['wfm_point'].view((batch * neibor, 1, signal_length))) ## batch, 1, signal_length -> batch , 1,  signal_length
        wf_feats_point = wf_feats_point.view((batch, neibor,-1))
        joined_feats_point = torch.cat([pw_feats_point, wf_feats_point], dim=-1)
        
        # continue processing neibor feats
        
        joined_feats_neib = self.mlp2(joined_feats_neib)
        joined_feats_neib = torch.max(joined_feats_neib, dim=1)[0]
        
        # join neibor with skip connection to point features
        global_feat = torch.cat([joined_feats_neib, torch.squeeze(joined_feats_point)],dim=-1)
        if self.global_constraint:
            global_feat = global_feat / (torch.norm(global_feat, p=2, dim=1, keepdim=True)+1e-6)

        global_feat = self.mlp3(global_feat)
        
        result = {k: self.classifier[k](global_feat) for k in self.classifier.keys()}
        
        
        
        return result

