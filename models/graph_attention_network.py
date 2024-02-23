import torch.nn as nn
from typing import Optional, Iterable, Union
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.init as init



class GraphAttentionNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Iterable[int],
                 heads: int = Union[int, Iterable[int]],
                 output_dim: Optional[int] = 1,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: Optional[bool] = False,
                 pooling: Optional[str] = 'mean',
                 *args,
                 **kwargs):
    
        super(GraphAttentionNetwork, self).__init__()
                
        # Input types check
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be of type int")
        
        if not isinstance(hidden_dims, Iterable):
            raise TypeError("hidden_dims must be an Iterable")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")
        if not all(isinstance(hidden_dim, int) for hidden_dim in hidden_dims):
            raise TypeError("hidden_dims must only contain integers")
        
        if not (isinstance(heads, int) or isinstance(heads, Iterable)):
            raise TypeError("heads must be either of type int or Iterable")
        if isinstance(heads, int):
            if heads <= 0:
                raise ValueError("heads must be between greater than 0")
        if isinstance(heads, Iterable):
            for item in heads:
                if not isinstance(item, int):
                    raise TypeError("heads list must only contain integers")
                if item <= 0:
                    raise ValueError("Each element in the heads list must be between greater than 0")
            if len(heads) != len(hidden_dims):
                raise ValueError("hidden_dims and heads must be of same size") 

        if not isinstance(output_dim, int):
            raise TypeError("output_dim must be of type int")
        
        if not isinstance(activation, nn.Module):
            raise TypeError("activation must be a torch.nn.Module")

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
        self.heads = [heads]*len(hidden_dims) if isinstance(heads, int) else heads
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_probabilities = [dropout]*len(hidden_dims) if isinstance(dropout, float) else dropout
        self.graph_norm = graph_norm
        self.pooling = pooling

        # Initialise GATConv Layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(input_dim, hidden_dims[0], self.heads[0]))

        for i in range(len(hidden_dims) - 1):
            hidden_layer = GATConv(hidden_dims[i]*self.heads[i], hidden_dims[i+1], self.heads[i+1])
            self.conv_layers.append(hidden_layer)


        # Initialise Graph Norm Layers
        if graph_norm:
            self.gn_layers = nn.ModuleList()
            for hidden_dim in hidden_dims:
                gn_layer = GraphNorm(hidden_dim)
                self.gn_layers.append(gn_layer)
        
        # Initialise Fully Connected Layer
        self.fc = nn.Linear(hidden_dims[-1]*self.heads[-1], output_dim)

        # Apply Xavier initialization to fc
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)


    def forward(self, x, edge_index, batch):

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if self.graph_norm:
                x = self.gn_layers[i](x, batch)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_probabilities[i], training=self.training)
        
        x = self._pooling_function(x, batch)
        x = self.fc(x)

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

    
    