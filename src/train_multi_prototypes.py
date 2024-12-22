from pathlib import Path
import numpy as np
import torch
from torch import nn
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from models import PrototypeNet

from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_shape(features_dir):
    """Get the shape of features from first valid file"""
    for genre_dir in features_dir.glob("*"):
        if genre_dir.is_dir():
            for file_path in genre_dir.glob("*.npy"):
                feat = np.load(file_path)
                return feat.shape
    raise ValueError("Could not find any valid feature files")

def train_source_model(data_path, source):
    """Train a single source model using the existing PECMAE implementation"""
    from train_protos import train as train_pecmae, create_protos
    
    print(f"\nStarting training for {source} source model...")
    
    features_dir = Path(data_path) / f"feats_{source}/gtzan/diffusion_4s"
    groundtruth_dir = Path('/content/MLSP_Project/groundtruth/gtzan')
    
    feature_shape = get_feature_shape(features_dir)
    print(f"Feature shape for {source}: {feature_shape}")
    
    timestamps = feature_shape[0]  # First dimension is time
    feat_dim = feature_shape[1]    # Second dimension is features
    n_labels = 10  # Number of genres in GTZAN
    n_protos_per_label = 5
    time_summarization = "transformer" 
    
    protos = create_protos(
        None,  
        "random",
        (timestamps, feat_dim),
        n_protos_per_label=n_protos_per_label,
        labels=list(range(n_labels))
    )
    
    model = PrototypeNet(
        protos=protos,
        time_dim=timestamps,
        feat_dim=feat_dim,
        n_labels=n_labels,
        batch_size=32,
        total_steps=10000,
        temp=0.1,
        alpha=0.7,
        max_lr=1e-3,
        proto_loss="l2",
        proto_loss_samples="class",
        use_discriminator=False,
        discriminator_type="mlp",
        time_summarization=time_summarization
    )
    
    _ = train_pecmae(
        data_dir=features_dir,
        data_dir_test=features_dir,
        metadata_file_train=groundtruth_dir / "groundtruth_train.tsv",
        metadata_file_val=groundtruth_dir / "groundtruth_val.tsv",
        metadata_file_test=groundtruth_dir / "groundtruth_test.tsv",
        protos_init="kmeans-centers",
        n_protos_per_label=n_protos_per_label,
        batch_size=32,
        dataset="gtzan",
        time_summarization=time_summarization,
        max_lr=1e-3,
        temp=0.1,
        alpha=0.7,
        proto_loss="l2",
        proto_loss_samples="class",
        use_discriminator=False,
        discriminator_type="mlp",
        gpu_id=0,
        timestamps=timestamps,
        total_steps=10000
    )
    
    checkpoints_dir = sorted(Path("tb_logs/zinemanet").glob("version_*"))[-1] / "checkpoints"
    best_checkpoint = sorted(checkpoints_dir.glob("*.ckpt"))[-1]
    print(f"Loading best checkpoint: {best_checkpoint}")
    
    model = PrototypeNet.load_from_checkpoint(
        best_checkpoint,
        protos=protos,
        time_dim=timestamps,
        feat_dim=feat_dim,
        n_labels=n_labels,
        batch_size=32,
        total_steps=10000,
        temp=0.1,
        alpha=0.7,
        max_lr=1e-3,
        proto_loss="l2",
        proto_loss_samples="class",
        use_discriminator=False,
        discriminator_type="mlp",
        time_summarization=time_summarization,  
        labels=list(range(n_labels)) 
    )
    
    return model

class MultiSourceDataset(Dataset):
    def __init__(self, base_path, sources, split='train'):
        self.base_path = Path(base_path)
        self.sources = sources
        
        self.label_map = {
            'blu': ('blues', 0),
            'cla': ('classical', 1),
            'cou': ('country', 2),
            'dis': ('disco', 3),
            'hip': ('hiphop', 4),
            'jaz': ('jazz', 5),
            'met': ('metal', 6),
            'pop': ('pop', 7),
            'reg': ('reggae', 8),
            'roc': ('rock', 9)
        }
        
        groundtruth_file = Path('/content/pecmae/groundtruth/gtzan') / f'groundtruth_{split}.tsv'
        self.metadata = pd.read_csv(groundtruth_file, sep='\t', header=None, 
                                  names=['label_short', 'path'])
        
        valid_rows = []
        for idx, row in self.metadata.iterrows():
            valid = True
            full_label, _ = self.label_map[row['label_short']]
            filename = Path(row['path']).name  
            
            for source in sources:
                feat_path = self.base_path / f"feats_{source}/gtzan/diffusion_4s/{full_label}/{filename}"
                if not feat_path.exists():
                    print(f"Missing file: {feat_path}")
                    valid = False
                    break
            if valid:
                valid_rows.append(idx)
        
        self.metadata = self.metadata.iloc[valid_rows].reset_index(drop=True)
        print(f"Found {len(self.metadata)} valid examples for {split} set")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        features = []
        
        full_label, label_idx = self.label_map[row['label_short']]
        filename = Path(row['path']).name  
        
        for source in self.sources:
            feat_path = self.base_path / f"feats_{source}/gtzan/diffusion_4s/{full_label}/{filename}"
            feat = np.load(feat_path)
            features.append(torch.FloatTensor(feat))
        
        return features, label_idx

class FusionModel(L.LightningModule):
    def __init__(self, source_models, hidden_dim=512, num_classes=10):
        super().__init__()
        self.source_models = nn.ModuleList(source_models)
        
        for model in self.source_models:
            for param in model.parameters():
                param.requires_grad = False
                
        total_protos = sum(model.n_protos for model in source_models)
        self.fusion = nn.Sequential(
            nn.Linear(total_protos, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.all_preds = []
        self.all_labels = []
        
        self.class_names = [
            'blues', 'classical', 'country', 'disco', 'hiphop', 
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
    
    def forward(self, x):
        similarities = []
        for i, model in enumerate(self.source_models):
            # Get prototype similarities
            sim = model.get_similarities(x[i])
            similarities.append(sim)
            
        # Concatenate similarities 
        combined = torch.cat(similarities, dim=1)
        
        return self.fusion(combined)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(dim=-1) == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(dim=-1) == y).float().mean()
        
        preds = y_hat.argmax(dim=-1).cpu().numpy()
        self.all_preds.extend(preds)
        self.all_labels.extend(y.cpu().numpy())
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        preds = y_hat.argmax(dim=-1).cpu().numpy()
        self.all_preds.extend(preds)
        self.all_labels.extend(y.cpu().numpy())

    def on_test_end(self):
        # Convertir les listes en numpy arrays
        all_preds = np.array(self.all_preds)
        all_labels = np.array(self.all_labels)
        
        # Générer et imprimer le rapport de classification
        print("\nClassification Report:")
        print(classification_report(
            all_labels, 
            all_preds, 
            target_names=self.class_names, 
            digits=4
        ))
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=self.class_names, 
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_preds, 
            average=None, 
            labels=range(len(self.class_names))
        )
        
        import pandas as pd
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print("\nDetailed Metrics:")
        print(metrics_df)
        
        metrics_df.to_csv('detailed_metrics.csv', index=False)
        
        self.all_preds = []
        self.all_labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def train_multi_source_model(data_path, output_path, sources=['bass', 'drums', 'other', 'vocals']):
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    source_models = []
    for source in sources:
        print(f"\nTraining model for {source} source...")
        model = train_source_model(data_path, source)
        source_models.append(model)
        torch.save(model.state_dict(), output_path / f"model_{source}.pt")

    train_dataset = MultiSourceDataset(data_path, sources, split='train')
    val_dataset = MultiSourceDataset(data_path, sources, split='val')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    test_dataset = MultiSourceDataset(data_path, sources, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32)


    fusion_model = FusionModel(source_models)
    logger = TensorBoardLogger("tb_logs", name="fusion_model")
    
    trainer = L.Trainer(
        max_epochs=500,
        accelerator='gpu',
        devices=[0],
        logger=logger,
        callbacks=[
            L.callbacks.ModelCheckpoint(
                monitor='val_accuracy',
                mode='max',
                filename='best-fusion-{epoch:02d}-{val_accuracy:.4f}'
            )
        ]
    )
    
    trainer.fit(fusion_model, train_loader, val_loader)
    torch.save(fusion_model.state_dict(), output_path / "fusion_model.pt")
    trainer.test(fusion_model, test_loader)

if __name__ == "__main__":
    data_path = Path("/content/MLSP_Project/features")
    output_path = Path("out_data")
    train_multi_source_model(data_path, output_path)