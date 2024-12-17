import torch
from torch import nn

class Conv1DModel(nn.Module):
    def __init__(self, cfg):
        super(Conv1DModel, self).__init__()

        self.waveform_encoder = self.build_waveform_encoder(cfg) if cfg.model.waveform_encoder.use else None
        self.additional_feats_mlp = self.build_additional_feats_mlp(cfg) if cfg.model.additional_feats_mlp.use else None
        self.classifier_mlp, self.classifier_head = self.build_classifier(cfg)


    def forward(self, x, cfg):
        # encode the waveform and additional features
        encoded_waveform = self.waveform_encoder(x['waveform'])\
              if self.waveform_encoder is not None else None
        encoded_additional = self.additional_feats_mlp(x['feats'][:,cfg.model.additional_feats_mlp.feats_idx])\
              if self.additional_feats_mlp is not None else None
        
        # Concatenate the encoded features
        if encoded_waveform is not None and encoded_additional is not None:
            x = torch.cat([encoded_waveform, encoded_additional], dim=1)
        else:
            x = encoded_waveform if encoded_waveform is not None else encoded_additional

        if self.classifier_mlp is not None:
            x = self.classifier_mlp(x)
            
        # pass the concatenated features through the classifier
        output = dict()
        for level in cfg.train.train_on_label_levels:
            output[str(level)] = self.classifier_head[str(level)](x)
        return output
    
    def build_waveform_encoder(self, cfg):
        encoder = nn.Sequential()
        for i, layer in enumerate(cfg.model.waveform_encoder.architecture):
            print(f"Adding layer {i}: {layer}")
            if layer.type not in ["Conv1d","MaxPool1d","BatchNorm1d","ReLU","Flatten","Linear", "Dropout"]:
                raise ValueError(f"Unknown layer type {layer.type}")
            
            # populate the encoder with the defined layers            
            encoder.add_module(name = f"WF_Encoder_{i:03}", module=getattr(nn,layer.type)(**layer.params))

        return encoder
    
    def build_additional_feats_mlp(self, cfg):
        mlp = nn.Sequential()
        for i, layer in enumerate(cfg.model.additional_feats_mlp.architecture):
            print(f"Adding layer {i}: {layer}")
            if layer.type not in ["BatchNorm1d","ReLU","Flatten","Linear", "Dropout"]:
                raise ValueError(f"Unknown layer type {layer.type}")
            
            # populate the mlp with the defined layers            
            mlp.add_module(name = f"Feat_MLP_{i:03}", module=getattr(nn,layer.type)(**layer.params))
        return mlp
            

    def build_classifier(self, cfg):
        classifier_mlp = None
        for i, layer in enumerate(cfg.model.classifier.architecture):
            print(f"Adding layer {i}: {layer}")
            if layer.type =='head':
                head = layer
                break # stop adding layers if the head is reached
            if classifier_mlp is None:
                classifier_mlp = nn.Sequential()
            if layer.type not in ["BatchNorm1d","ReLU","Flatten","Linear", "Dropout"]:
                raise ValueError(f"Unknown layer type {layer.type}")
            
            # populate the mlp with the defined layers            
            classifier_mlp.add_module(name = f"Classifier_MLP_{i:03}", module=getattr(nn,layer.type)(**layer.params))
            
        # build the head
        classifier_head = nn.ModuleDict()
        for level in cfg.label_schema:
            if level not in cfg.train.train_on_label_levels:
                continue
            classifier_head[str(level)] = nn.Linear(head.in_features, len(cfg.label_schema[level]))

        return classifier_mlp, classifier_head