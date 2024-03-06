import torch.nn.functional as F
import sklearn.metrics as metrics
import torch


def test(loader, model, loss_fn, device, task='binary', binary_decision_threshold=0.5):

    if task=='binary':
        return test_binary(loader, model, loss_fn, device, decision_threshold=binary_decision_threshold)
    elif task=='regression':
        return test_regression()
    else:
        raise ValueError(f"Unsupported task type '{task}'")


def test_binary(loader, model, loss_fn, device, decision_threshold=0.5):
    
    running_loss = 0
    total_samples = 0
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):
        
            data = data.to(device)
            
            outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)
            
            probs = F.sigmoid(outputs)
            preds = (probs > decision_threshold).int()
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(data.y.tolist())
            
            loss = loss_fn(outputs.float(), data.y.float())
            
            running_loss += loss.item() * data.y.size(0)
            total_samples += data.y.size(0)
        
        avg_loss = running_loss / len(loader.dataset)
    
    metrics_dict = compute_metrics(all_labels, all_preds, task='binary')
    metrics_dict['roc_auc'] = metrics.roc_auc_score(all_labels, all_probs)
    metrics_dict['loss'] = avg_loss
    conf_mat = metrics.confusion_matrix(all_labels, all_preds).ravel()

#     tn, fp, fn, tp = metrics.confusion_matrix(all_labels, all_preds).ravel()
#     conf_mat = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
     
    return avg_loss, metrics_dict, conf_mat
                                                                         

def test_regression(loader, model, loss_fn, device):
    
    running_loss = 0
    total_samples = 0
    
    all_preds = []
    all_true = []

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):
        
            data = data.to(device)
            
            outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)

            all_preds.extend(outputs.tolist())
            all_true.extend(data.y.tolist())
            
            loss = loss_fn(outputs.float(), data.y.float())
            
            running_loss += loss.item() * data.y.size(0)
            total_samples += data.y.size(0)
        
        avg_loss = running_loss / len(loader.dataset)
    
    metrics_dict = compute_metrics(all_true, all_preds)
    metrics_dict['loss'] = avg_loss
        
    return avg_loss, metrics_dict


def compute_metrics(y_true, y_pred, task='binary'):
    
    if task=='binary':
        accuracy = metrics.accuracy_score(y_true, y_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        mcc = metrics.matthews_corrcoef(y_true, y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc
        }
    elif task=='regression':
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = metrics.root_mean_squared_error(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)

        metrics_dict = {
            'explained_variance': explained_variance,
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    return metrics_dict
