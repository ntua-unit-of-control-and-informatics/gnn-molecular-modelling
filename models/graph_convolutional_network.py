import torch.nn as nn
from typing import Optional, Iterable, Union
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.init as init

class GraphConvolutionalNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Iterable[int],
                 output_dim: Optional[int] = 1,
                 activation: Optional[nn.Module] = nn.ReLU(),
#                  final_activation: Optional[Union[str, None]] = None,
                 dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: Optional[bool] = False,
                 pooling: Optional[str] = 'mean',
                 *args,
                 **kwargs):
        
        super(GraphConvolutionalNetwork, self).__init__()
        
        # Input types check
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be of type int")
        
        if not isinstance(hidden_dims, Iterable):
            raise TypeError("hidden_dims must be an Iterable")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")
        if not all(isinstance(hidden_dim, int) for hidden_dim in hidden_dims):
            raise TypeError("hidden_dims must only contain integers")
        
        if not isinstance(output_dim, int):
            raise TypeError("output_dim must be of type int")
        
        if not isinstance(activation, nn.Module):
            raise TypeError("activation must be a torch.nn.Module")
            
#         if not (final_activation is None or isinstance(final_activation, str)):
#             raise TypeError("final_activation must be of type str or None")            
        
        if not (isinstance(dropout, float) or isinstance(dropout, Iterable)):
            raise TypeError("dropout must be either of type float or Iterable")
        if isinstance(dropout, float):
            if not 0 <= dropout <= 1:
                raise ValueError("dropout probability must be between 0 and 1")
        if isinstance(dropout, Iterable):
            for item in dropout:
                if not isinstance(item, float):
                    raise TypeError("dropout list must only contain floats")
                if not 0 <= item <= 1:
                    raise ValueError("Each element in the dropout list must be between 0 and 1")
            if len(dropout) != len(hidden_dims):
                raise ValueError("hidden_dims and dropout must be of same size")        
        
        if not isinstance(graph_norm, bool):
            raise TypeError("graph_norm must be of type bool")  
        
        if pooling is not None and pooling not in ['mean', 'add', 'max']:
            raise NotImplementedError(f"Pooling operation '{self.pooling}' is not supported")
        
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
#         self.final_activation = final_activation
        self.dropout_probabilities = [dropout]*len(hidden_dims) if isinstance(dropout, float) else dropout
        self.graph_norm = graph_norm
        self.pooling = pooling
        
        
        # Initialise GCNConv Layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dims[0]))
            
        for i in range(len(hidden_dims) - 1):
            hidden_layer = GCNConv(hidden_dims[i], hidden_dims[i+1])
            self.conv_layers.append(hidden_layer)
        
        
        # Initialise Graph Norm Layers
        if graph_norm:
            self.gn_layers = nn.ModuleList()
            for hidden_dim in hidden_dims:
                gn_layer = GraphNorm(hidden_dim)
                self.gn_layers.append(gn_layer)
            
        # Initialise Fully Connected Layer
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        
        # Apply Xavier initialization to fc
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
        
        self.fc1 = nn.Linear(hidden_dims[-1], 128)
        self.fc2 = nn.Linear(128, output_dim)
        
        
        
          
        
    def forward(self, x, edge_index, batch):

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if self.graph_norm:
                x = self.gn_layers[i](x, batch)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_probabilities[i], training=self.training)
        
        x = self._pooling_function(x, batch)
        x = self.fc(x)
        
#         x = self.fc1(x)
#         x = self.fc2(x)
        
#         if self.final_activation is None:
#             pass
#         if self.final_activation=='sigmoid':
#             x = F.sigmoid(x)
#         elif self.final_activation=='softmax':
#             x = F.softmax(x, dim=-1)
#         else:
#             raise NotImplementedError(f'"{self.final_activation}" is not supported as a final activation function.')
        
        return x
    
    def _pooling_function(self, x, batch):

        if self.pooling == 'add':
            return global_add_pool(x, batch)
        elif self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        else:
            raise NotImplementedError(f"Pooling operation '{self.pooling}' is not supported")

    