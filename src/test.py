from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from train_multi_linear import SourcePECMAEModule, FusionModel, FeatureDataset

class TestDataset(Dataset):
    """Dataset for test data loading"""
    def __init__(self, base_path, sources):
        self.base_path = Path(base_path)
        self.sources = sources
        self.genre_map = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 
                         'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
        self.inv_genre_map = {v: k for k, v in self.genre_map.items()}
        
        first_source = sources[0]
        self.metadata = pd.read_csv(self.base_path / f"feats_{first_source}/test_metadata.csv")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        features = []
        
        for source in self.sources:
            feat_path = self.base_path / f"feats_{source}/gtzan/diffusion_4s" / row['feature']
            feat = np.load(feat_path)
            features.append(torch.FloatTensor(feat))
        
        label = self.genre_map[row['label']]
        return features, label, row['feature']

def evaluate_model(model, dataloader, device, is_source_model=True):
    """
    Evaluate model on given dataloader
    is_source_model: True for source models (returns 2 values), False for fusion model (returns 3 values)
    """
    model.eval()
    predictions_list = []
    labels_list = []
    files_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if is_source_model:
                # source model dataloader returns (features, labels)
                features = batch['feature'].to(device)
                labels = batch['label'].to(device)
                files = [""] * len(labels)  
            else:
                # fusion model dataloader returns (features, labels, files)
                features, labels, files = batch
                if isinstance(features, list):
                    features = [f.to(device) for f in features]
                else:
                    features = features.to(device)
                labels = labels.to(device)
            
            outputs = model(features)
            batch_preds = outputs.argmax(dim=1)
            
            predictions_list.append(batch_preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            files_list.extend(files)
    
    # Concatenate 
    final_predictions = np.concatenate(predictions_list)
    final_labels = np.concatenate(labels_list)
    
    return final_predictions, final_labels, files_list

def test_models(data_path, model_path, sources=['bass', 'drums', 'other', 'vocals'], batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = Path(data_path)
    model_path = Path(model_path)
    
    source_models = []
    for source in sources:
        print(f"\nTesting {source} source model:")
        try:
            model = SourcePECMAEModule()
            model.load_state_dict(torch.load(model_path / f"model_{source}.pt"))
            model = model.to(device)
            source_models.append(model)
            
            test_dataset = FeatureDataset(
                data_path / f"feats_{source}/gtzan/diffusion_4s",
                data_path / f"feats_{source}/test_metadata.csv"
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            preds, labels, _ = evaluate_model(model, test_loader, device, is_source_model=True)
            
            accuracy = accuracy_score(labels, preds)
            print(f"{source.capitalize()} Model Test Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(labels, preds, 
                                     target_names=['Blues', 'Classical', 'Country', 'Disco', 
                                                 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'],
                                     zero_division=0))
            
            print(f"Number of predictions: {len(preds)}")
            print(f"Number of labels: {len(labels)}")
            
        except Exception as e:
            print(f"Error evaluating {source} model:", str(e))
            print("Shapes of arrays:")
            print(f"Predictions shape: {preds.shape if 'preds' in locals() else 'Not available'}")
            print(f"Labels shape: {labels.shape if 'labels' in locals() else 'Not available'}")
            continue
    

    fusion_model = FusionModel(source_models)
    fusion_model.load_state_dict(torch.load(model_path / "fusion_model.pt"))
    fusion_model = fusion_model.to(device)
    
    test_dataset = TestDataset(data_path, sources)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    

    preds, labels, files = evaluate_model(fusion_model, test_loader, device, is_source_model=False)

    accuracy = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    
    print(f"\nFusion Model Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, 
                              target_names=['Blues', 'Classical', 'Country', 'Disco', 
                                          'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'],
                              zero_division=0))
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    results_df = pd.DataFrame({
        'File': files,
        'True_Label': [test_dataset.inv_genre_map[l] for l in labels],
        'Predicted_Label': [test_dataset.inv_genre_map[p] for p in preds],
        'Correct': labels == preds
    })
    results_df.to_csv(model_path / 'test_results.csv', index=False)
    print(f"\nDetailed results saved to {model_path / 'test_results.csv'}")

if __name__ == "__main__":
    data_path = Path("/content/pecmae/features")
    model_path = Path("output")
    test_models(data_path, model_path)