import torch

import yacs

"""
Submodule code for a classification head.
"""

class MlpHead(torch.nn.Module):
    """
    Multilayer Perceptron Classifcation Head
    """
    def __init__(self, cfg: yacs.config.CfgNode, in_dim: int=96):
        """
        Take the config file and initilize the model accordingly.
        """
        super().__init__()
        
        self.cfg = cfg

        # Validation
        assert self.cfg.HEAD.TYPE.lower() == 'linear', f'HEAD.TYPE is expected to be linear!'

        self.in_dim = in_dim
        self.out_dim = self.cfg.HEAD.NUM_CLASSES
        self.nodes_by_layer = self.cfg.HEAD.LAYERS # [300, 300]

        self.nodes_by_layer.insert(0, self.in_dim)
        self.nodes_by_layer.append(self.out_dim) # [96, 300, 300, 2]

        if self.cfg.HEAD.LAYER_ACT_FUNC.lower() == 'relu':
            layer_act_class = torch.nn.ReLU
        else:
            raise NotImplementedError(f'{self.cfg.HEAD.LAYER_ACT_FUNC.lower()} is not supported!')

        print(self.nodes_by_layer)

        layer_modules = []
        for i in range(1, len(self.nodes_by_layer)):
            fc_in = self.nodes_by_layer[i-1]
            fc_out = self.nodes_by_layer[i]
            layer_modules.append(torch.nn.Linear(fc_in, fc_out))

            if i != len(self.nodes_by_layer) - 1:
                layer_modules.append(layer_act_class())
                layer_modules.append(torch.nn.Dropout(p=self.cfg.HEAD.DROPOUT))

        self.layers = torch.nn.Sequential(*layer_modules)


    def forward(self, x):
        """
        Forward pass for the MLP Head

        Args:
            x: input features. Expected shape: B D.
                where B is the batch size and D is the feature dimension.
        """

        x = self.layers(x) # Shape: B C
        return x
