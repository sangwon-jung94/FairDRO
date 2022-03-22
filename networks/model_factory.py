import torch.nn as nn

from networks.vgg import vgg19, vgg16, vgg13, vgg11, vgg8, vgg6, vgg6_s, vgg5_s
from networks.resnet import resnet10, resnet12,resnet18, resnet34, resnet50, resnet101
from networks.resnet_dropout import resnet18_dropout
from networks.mobilenet import mobilenet_v2
from networks.shufflenet import shufflenet_v2_x1_0
from networks.cifar_net import Net
from networks.mlp import MLP
from networks.wide_resnet import Wide_ResNet
from networks.autoencoder import ImageTransformNet

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(target_model, num_classes=2, img_size=224, pretrained=False, num_groups=2):

        if target_model == 'mlp': 
            return MLP(feature_size=img_size, hidden_dim=64, num_classes=num_classes)

        elif 'resnet' in target_model:
            model_class = eval(target_model)
            if pretrained:
                model = model_class(pretrained=True, img_size=img_size)
                model.fc = nn.Linear(in_features=model.fc.weight.shape[1], out_features=num_classes, bias=True)
            else:
                model = model_class(pretrained=False, num_classes=num_classes, num_groups=num_groups, img_size=img_size)
            return model

        elif target_model == 'cifar_net':
            return Net(num_classes=num_classes)

        elif target_model == 'shufflenet':
            if pretrained:
                model = shufflenet_v2_x1_0(pretrained=True, img_size=img_size)
                model.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
            else:
                model = shufflenet_v2_x1_0(pretrained=False, num_classes=num_classes, img_size=img_size)
            return model
        
        elif target_model.startswith('wrn'):
            depth = int(target_model[3:5])
            wide_factor = int(target_model[-1])
            return Wide_ResNet(depth, wide_factor, 0.3, num_classes=num_classes, image_size=img_size)        

        elif target_model == 'mobilenet':
            if pretrained:
                model = mobilenet_v2(pretrained=True, img_size=img_size)
                model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
            else:
                model = mobilenet_v2(pretrained=False, num_classes=num_classes, img_size=img_size)
            return model

        elif target_model == 'image_transformer':
            model = ImageTransformNet()
            return model
        
        elif target_model == 'bert':
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                'bert-based-uncased',
                num_labels=num_classes)


        else:
            raise NotImplementedError

