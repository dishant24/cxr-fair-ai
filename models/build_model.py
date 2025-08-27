import torchvision
import torch.nn as nn
import torch


class DenseNet_Model(nn.Module):
     def __init__(self, weights, out_feature):
          super().__init__()
          self.weight = weights
          self.out_feature = out_feature
          self.encoder = torchvision.models.densenet121(weights=weights) # Adapt the architecture to initial paper: The limits of fair medical imaging and almost all other papers
          self.relu = nn.ReLU()
          self.clf = nn.Linear(1000, out_feature)

     def encode(self, x):
          return self.encoder(x)

     def forward(self, x):
          z = self.encode(x)
          z = self.relu(z)
          return self.clf(z)

     def predict_proba(self, x):
          return torch.sigmoid(self(x))


# Help to build model for race prediction training
def model_transfer_learning(path, model, device, gradcam):

     state_dict = torch.load(path, map_location=device, weights_only=True)
     state_dict.pop("clf.weight", None)
     state_dict.pop("clf.bias", None)
     
     model.load_state_dict(state_dict, strict=False)

     if not gradcam:
          for params in model.encoder.parameters():
               params.requires_grad = False

     return model