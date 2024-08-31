
from torch import nn
from typing import Literal
from models_train.inceptionresnet import BasicConv2d, inceptionresnetv2

from models_train.baseIQAmodel import IQA

_BASE_MODELS = Literal["resnet", "inceptionresnet"]

from icecream import ic
ic.disable()

class KonCept(IQA):
    # @staticmethod
    # def get_base_model(base_model_name: _BASE_MODELS):
    #     if "inceptionresnet" in base_model_name:
    #         base_model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    #     elif "resnet" in base_model_name:
    #         base_model = models.__dict__["resnet50"](pretrained=True)
    #     else:
    #         raise NameError(f"No base model {base_model_name}")
    #     return base_model
    def __init__(self, arch="resnet50", activation="relu",**kwargs):
        ic("KONCEPT")
        super(KonCept, self).__init__(arch)
        num_classes=1
        # self.base_model_name = base_model_name
        # base_model = self.get_base_model(base_model_name)
        
        # print(base_model.children())
        # self.base= nn.Sequential(*list(base_model.children())[:-1])
        in_features, self.features = self.get_features(self._base_model_features)
        # base_model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        # self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        Activ = self.get_activation_module(activation)
        # ic(list(self.features.children()))
        self.adapter = BasicConv2d(in_features[1], 1536, kernel_size=1, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(1536, 2048),
            Activ(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            Activ(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            Activ(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self,x):
        ic(x.shape)
        x = self.features(x)
        ic(x.shape)
        if self.arch != "inceptionresnet":
            x = self.adapter(x)
        ic(x.shape)
        x = x.view(x.size(0), -1)
        ic(x.shape)
        x = self.fc(x)

        return x 
