import torch.nn as nn
from typing import Optional, Iterable, Union
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GraphNorm, BatchNorm, GraphSizeNorm, InstanceNorm, LayerNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.init as init
from torch import Tensor
from torch_geometric.typing import OptTensor


class GraphTransformerBlock(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 heads: Optional[int] = 1,
                 edge_dim: Optional[int] = None,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 dropout_probability: float = 0.5,
                 graph_norm: Optional[bool] = False,
                 jittable: Optional[bool] = True,
                 *args,
                 **kwargs):
        
        super(GraphTransformerBlock, self).__init__()
        
        self.jittable = jittable

        self.hidden_layer = TransformerConv(input_dim, hidden_dim, heads, edge_dim=edge_dim)

        if jittable:
            self.hidden_layer = self.hidden_layer.jittable()

        self.graph_norm = graph_norm
        if self.graph_norm:
            self.gn_layer = GraphNorm(hidden_dim*heads)
        else:
            self.gn_layer = None
            
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)
        
    
    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                batch: Optional[Tensor],
                edge_attr: OptTensor = None) -> Tensor:

        x = self.hidden_layer(x, edge_index, edge_attr=edge_attr)
        if self.gn_layer is not None:
            x = self.gn_layer(x, batch)
        x = self.activation(x)
        x = self.dropout(x)

        return x



class GraphTransformerNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Iterable[int],
                 heads: int = Union[int, Iterable[int]],
                 edge_dim: Optional[int] = None,
                 output_dim: Optional[int] = 1,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 dropout: Union[float, Iterable[float]] = 0.5,
                 graph_norm: Optional[bool] = False,
                 pooling: Optional[str] = 'mean',
                 jittable: Optional[bool] = True,
                 *args,
                 **kwargs):
    
        super(GraphTransformerNetwork, self).__init__()
                
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
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.dropout_probabilities = [dropout]*len(hidden_dims) if isinstance(dropout, float) else dropout
        self.graph_norm = graph_norm
        self.pooling = pooling
        self.jittable = jittable


        self.graph_layers = nn.ModuleList() 
        graph_layer = GraphTransformerBlock(input_dim, hidden_dims[0],
                                          heads=self.heads[0],
                                          activation=activation,
                                          edge_dim=edge_dim,
                                          dropout_probability=self.dropout_probabilities[0],
                                          graph_norm=graph_norm,
                                          jittable=jittable)
        self.graph_layers.append(graph_layer)

        
        for i in range(len(hidden_dims) - 1):
            graph_layer = GraphTransformerBlock(hidden_dims[i]*self.heads[i], hidden_dims[i+1],
                                              heads=self.heads[i+1],
                                              edge_dim=edge_dim,
                                              activation=activation,
                                              dropout_probability=self.dropout_probabilities[i],
                                              graph_norm=graph_norm,
                                              jittable=jittable)
            self.graph_layers.append(graph_layer)
    
        
        # Initialise Fully Connected Layer
        self.fc = nn.Linear(hidden_dims[-1]*self.heads[-1], output_dim)

        # Apply Xavier initialization to fc
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)
    
    
    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                batch: Optional[Tensor],
                edge_attr: OptTensor = None) -> Tensor:
        
        x = self._forward_graph(x, edge_index, batch, edge_attr=edge_attr)
        x = self.fc(x)
        return x
    

    def _forward_graph(self,
                       x: Tensor,
                       edge_index: Tensor,
                       batch: Optional[Tensor],
                       edge_attr: OptTensor = None) -> Tensor:

        for graph_layer in self.graph_layers:
            x = graph_layer(x, edge_index, batch=batch, edge_attr=edge_attr)
        x = self._pooling_function(x, batch)
        return x
    

    def _pooling_function(self,
                          x: Tensor, 
                          batch: Optional[Tensor]) -> Tensor:

        if self.pooling == 'add':
            return global_add_pool(x, batch)
        elif self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        else:
            raise NotImplementedError(f"Pooling operation '{self.pooling}' is not supported")
             