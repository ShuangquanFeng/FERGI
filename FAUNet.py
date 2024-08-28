import torch
import torch.nn as nn
# import torch.nn.functional as F

class FAUNet(nn.Module):
    def __init__(self, preprocess, layer_sizes, activation):
        super(FAUNet, self).__init__()
        
        # Preprocess
        self.preprocess = nn.ModuleList()
        for _ in range(layer_sizes[0]):
            node_preprocess = nn.ModuleList()
            for step in preprocess:
                if step == 'linear':
                    linear_layer = nn.Linear(1, 1)
                    nn.init.constant_(linear_layer.weight, 1)
                    nn.init.constant_(linear_layer.bias, -1)
                    node_preprocess.append(linear_layer)
                elif step == 'sigmoid':
                    node_preprocess.append(nn.Sigmoid())
                else:
                    raise ValueError ('Preprocess step not recognized')
            self.preprocess.append(node_preprocess)
        
        # Dense hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.hidden_layers.append(nn.Linear(sum(layer_sizes[0:(i+1)]), layer_sizes[i+1]))
            
        # Activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation function not recognized.')
        
        # Output layer
        self.output_layer = nn.Linear(sum(layer_sizes[0:(len(layer_sizes)-1)]), layer_sizes[-1])

    def forward(self, x):
        # Preprocess
        preprocessed_x = []
        for i, node_preprocess in enumerate(self.preprocess):
            node = x[:, i:i+1]
            for step in node_preprocess:
                node = step(node)
            preprocessed_x.append(node)
        
        x = torch.cat(preprocessed_x, dim=1)
        
        # Dense connectivity forward pass
        for hidden_layer in self.hidden_layers:
            x = torch.cat([x, hidden_layer(x)], dim=1)
            x = self.activation(x)
        
        out = self.output_layer(x)
                
        return out


# import torch
# import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, layer_sizes, dropout=0.5):
#         super(MLP, self).__init__()

#         self.layers = nn.ModuleList()
#         for i in range(len(layer_sizes) - 1):
#             self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
#             if i < len(layer_sizes) - 2:  # No batchnorm/dropout/ReLu for last layer
#                 self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
#                 self.layers.append(nn.ReLU())
#                 self.layers.append(nn.Dropout(dropout))

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x