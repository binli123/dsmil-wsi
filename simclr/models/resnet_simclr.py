import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()

        from conch.open_clip_custom import create_model_from_pretrained
        conch_v1, _ = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="key")
        conch_v1.forward = partial(conch_v1.encode_image, proj_contrast=False, normalize=False)

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "conch": conch_v1}

        resnet = self._get_basemodel(base_model)
        if resnet ==conch_v1:
            num_ftrs = 512
            self.features = resnet
        else:
            num_ftrs = resnet.fc.in_features
            self.features = nn.Sequential(*list(resnet.children())[:-1])


        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
