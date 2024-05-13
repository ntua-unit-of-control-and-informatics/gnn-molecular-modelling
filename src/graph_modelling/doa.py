from abc import ABC, abstractmethod
import torch
import pickle

class GraphEmbeddingSpaceDoA(ABC):

    def __init__(self,
                #  doa: str,
                 *args,
                 **kwargs):
        pass
        # self.doa_type = doa
        # self._XTX = None
    
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_applicability_domain_with_embedding(self, embeddings):
        pass

    @abstractmethod
    def check_applicability_domain_with_model(self, data_loader, model, device):
        pass


    def save(self, filepath='doa.pkl'):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)


class Leverage(GraphEmbeddingSpaceDoA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._XTX_inv = None
        self.n = None
        self.k = None
        self.threshold = None
    
    def fit(self, data_loader, model, device, *args, **kwargs):

        initial_training_state = model.training
        
        try:
            embedding_size = model.hidden_dims[-1] * model.heads[-1]
        except AttributeError:
            embedding_size = model.hidden_dims[-1]

        XTX = torch.zeros((embedding_size, embedding_size)).to(device)
        self.n = len(data_loader.dataset)
        self.k = embedding_size
        self.threshold = 3 * self.k / self.n

        # print(self.threshold)

        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(data_loader):
                data = data.to(device)
                embeddings = model._forward_graph(x=data.x, edge_index=data.edge_index, batch=data.batch)
                XTX += embeddings.T @ embeddings
        # print(XTX.shape)
        # print(embeddings[:, 17])
        # print(embeddings[0])
        try:
            self._XTX_inv = torch.linalg.inv(XTX)
        except torch._C._LinAlgError:
            self._XTX_inv = torch.linalg.pinv(XTX)
        
        if initial_training_state:
            model.train()
        else:
            model.eval()
        
        return self
    
    # def (self, data_loader, model, device):
    
    def compute_leverage_with_embedding(self, embeddings):
        # x = (batch_size, embedding_size)
        if self._XTX_inv is None or self.n is None or self.k is None or self.threshold is None:
            raise RuntimeError("Necessary attributes are not initialized. Please call fit method first.")

        h = embeddings @ self._XTX_inv @ embeddings.T

        return h
    
    def compute_leverage_with_model(self, data_loader, model, device):

        initial_training_state = model.training
        
        h = []
        model = model.to(device)
        model.eval()
        # with torch.no_grad():
        for _, data in enumerate(data_loader):
            data = data.to(device)
            embeddings = model._forward_graph(x=data.x, edge_index=data.edge_index, batch=data.batch)
            h_batch = (embeddings.unsqueeze(1) @ self._XTX_inv @ embeddings.unsqueeze(2)).squeeze().tolist()
            h.extend(h_batch)
        
        if initial_training_state:
            model.train()
        else:
            model.eval()
        
        return torch.tensor(h)
        
    


    def check_applicability_domain_with_embedding(self, embeddings):
        h = self.compute_leverage_with_embedding(embeddings)
        return h.lt(self.threshold)

    def check_applicability_domain_with_model(self, data_loader, model, device):
        h = self.compute_leverage_with_model(data_loader, model, device)
        is_in_doa = h.lt(self.threshold)
        # print(sum(is_in_doa)/len(is_in_doa))    
        return is_in_doa


    
    
# it is true or false showing if the initial model state was training or eval respectively
    
    # it is true or false showing if the initial model state was training or eval respectively