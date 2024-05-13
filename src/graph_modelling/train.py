from tqdm import tqdm
from models.graph_attention_network import GraphAttentionNetwork
from models.graph_transformer_network import GraphTransformerNetwork
import torch

# Train function
def train(epoch, n_epochs, loader, model, loss_fn, optimizer, device, use_tqdm=True):
    
    running_loss = 0
    total_samples = 0
    
    tqdm_loader = tqdm(loader, desc=f'Epoch {epoch}/{n_epochs}') if use_tqdm else loader
    
    model.train()
    for _, data in enumerate(tqdm_loader):
        
        data = data.to(device)
        
        optimizer.zero_grad()
        if isinstance(model, GraphAttentionNetwork) or isinstance(model, GraphTransformerNetwork):
            outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(-1)
        else:
            outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)
        loss = loss_fn(outputs.float(), data.y.float())
    

        running_loss += loss.item() * data.y.size(0)
        total_samples += data.y.size(0)
        
        loss.backward()
        optimizer.step()
        
        if use_tqdm:
            # Update tqdm description with additional information
            tqdm_loader.set_postfix(loss=running_loss/total_samples)
        
    avg_loss = running_loss / len(loader.dataset)
    
    if use_tqdm:
        tqdm_loader.set_postfix(loss=running_loss)
        tqdm_loader.close()
            
    return avg_loss


    