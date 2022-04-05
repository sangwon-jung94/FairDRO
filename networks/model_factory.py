import torch.nn as nn

from networks.resnet import resnet10, resnet12,resnet18, resnet34, resnet50, resnet101
from networks.mlp import MLP
from networks.cifar_net import Net

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(target_model, n_classes=2, img_size=224, pretrained=False, n_groups=2):

        if target_model == 'mlp': 
            return MLP(feature_size=img_size, hidden_dim=64, n_classes=n_classes)
        
        elif target_model == 'lr': 
            return MLP(feature_size=img_size, hidden_dim=64, n_classes=n_classes, n_layer=1)



        elif 'resnet' in target_model:
            model_class = eval(target_model)
            if pretrained:
                model = model_class(pretrained=True, img_size=img_size)
                model.fc = nn.Linear(in_features=model.fc.weight.shape[1], out_features=n_classes, bias=True)
            else:
                model = model_class(pretrained=False, n_classes=n_classes, n_groups=n_groups, img_size=img_size)
            return model

        elif target_model == 'cifar_net':
            return Net(n_classes=n_classes)
        
        elif target_model == 'bert':
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=n_classes)
            return model


        else:
            raise NotImplementedError


