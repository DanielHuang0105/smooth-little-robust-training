import torch
from robustness import model_utils
class Net(torch.nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        self.model_layer = torch.nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.model_layer(x)
        return x

# make a model with a pretrained weight
def generate_model(arch,ds,resume_path):
    model,_ = model_utils.make_and_restore_model(arch=arch, 
                                                dataset=ds, 
                                                resume_path=resume_path, 
                                                parallel=False)
    model = Net(model)
    return model