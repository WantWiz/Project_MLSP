from pathlib import Path
import numpy as np
import torch
from torch import nn
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from sklearn.model_selection import train_test_split

class FeatureDataset(Dataset):
    """Dataset class for loading features with optimized statistics calculation"""
    def __init__(self, features_dir, metadata_file, batch_size=32):
        self.features_dir = Path(features_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.genre_map = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 
                         'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
        
        # Calculate dataset statistics using batched processing
        sum_features = None
        sum_squared = None
        total_samples = 0
        
        for i in range(0, len(self.metadata), batch_size):
            batch_metadata = self.metadata.iloc[i:i + batch_size]
            batch_features = []
            
            for _, row in batch_metadata.iterrows():
                feat = np.load(self.features_dir / row['feature'])
                batch_features.append(feat)
            
            batch_features = np.concatenate(batch_features, axis=0)
            
            if sum_features is None:
                sum_features = np.zeros_like(batch_features[0])
                sum_squared = np.zeros_like(batch_features[0])
            
            sum_features += np.sum(batch_features, axis=0)
            sum_squared += np.sum(np.square(batch_features), axis=0)
            total_samples += batch_features.shape[0]
            
            print(f"\rProcessed {min(i + batch_size, len(self.metadata))}/{len(self.metadata)} files", 
                  end="", flush=True)
        
        # Compute mean and standard deviation
        self.mean = sum_features / total_samples
        variance = (sum_squared / total_samples) - np.square(self.mean)
        self.std = np.sqrt(np.maximum(variance, 1e-8))  # small epsilon to avoid division by zero
        
        print("Dataset initialization complete")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        feature = np.load(self.features_dir / row['feature'])
        feature = (feature - self.mean) / (self.std + 1e-8)  # normalization
        label = self.genre_map[row['label']]
        return {
            'feature': torch.FloatTensor(feature),
            'label': label
        }

def create_metadata_files(data_path, source, test_size=0.2, val_size=0.1):
    """Create metadata files for training, validation and testing"""
    features_path = Path(data_path) / f"feats_{source}/gtzan/diffusion_4s"
    
    data = []
    for genre_path in features_path.glob("*"):
        if genre_path.is_dir():
            genre = genre_path.name
            for file_path in genre_path.glob("*.npy"):
                data.append({
                    'file': file_path.stem,
                    'label': genre,
                    'feature': str(file_path.relative_to(features_path))
                })
    
    df = pd.DataFrame(data)
    
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size), stratify=train_val_df['label'], random_state=42)
    
    output_dir = Path(data_path) / f"feats_{source}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train_metadata.csv", index=False)
    val_df.to_csv(output_dir / "val_metadata.csv", index=False)
    test_df.to_csv(output_dir / "test_metadata.csv", index=False)
    
    return output_dir

class SourcePECMAEModule(L.LightningModule):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def _shared_step(self, batch, batch_idx, step_type):
        x = batch['feature']
        y = batch['label']
        
        if len(x.shape) == 3:  # If shape is [batch, time, features]
            x = x.mean(dim=1)  # Average over time dimension
        
        logits = self(x)
        
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        
        self.log(f'{step_type}_loss', loss, prog_bar=True)
        self.log(f'{step_type}_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class MultiSourceDataset(Dataset):
    def __init__(self, base_path, sources):
        self.base_path = Path(base_path)
        self.sources = sources
        self.genre_map = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 
                         'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
        
        first_source = sources[0]
        self.metadata = pd.read_csv(self.base_path / f"feats_{first_source}/train_metadata.csv")
        
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
        return features, label

class FusionModel(L.LightningModule):
    def __init__(self, source_models, hidden_dim=512, num_classes=10):
        super().__init__()
        self.source_models = nn.ModuleList(source_models)
        
        self.total_dim = hidden_dim * len(source_models)
        
        # Freeze source models
        for model in self.source_models:
            for param in model.parameters():
                param.requires_grad = False
                
        self.fusion = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        embeddings = []
        for i, model in enumerate(self.source_models):
            if len(x[i].shape) == 3: 
                x[i] = x[i].mean(dim=1)  
                
            src_features = model.encoder(x[i])
            embeddings.append(src_features)
            
        # concatenate embeddings along feature dimension
        combined = torch.cat(embeddings, dim=1)
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
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def train_source_model(data_path, source):
    """Train a single source model"""
    print(f"\nStarting training for {source} source model...")
    
    features_dir = Path(data_path) / f"feats_{source}/gtzan/diffusion_4s"
    metadata_train = Path(data_path) / f"feats_{source}/train_metadata.csv"
    metadata_val = Path(data_path) / f"feats_{source}/val_metadata.csv"
    
    print(f"Loading {source} datasets...")
    train_dataset = FeatureDataset(features_dir, metadata_train)
    val_dataset = FeatureDataset(features_dir, metadata_val)
   

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    model = SourcePECMAEModule()
    
    logger = TensorBoardLogger(
        "tb_logs",
        name=f"source_model_{source}",
        default_hp_metric=False
    )

    trainer = L.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=[0],
        logger=logger,
        callbacks=[
            L.callbacks.ModelCheckpoint(
                monitor='val_accuracy',
                mode='max',
                filename=f'best-{source}-{{epoch:02d}}-{{val_accuracy:.4f}}'
            ),
            L.callbacks.RichProgressBar()
        ]
    )
    
    print(f"\nTraining {source} model:")
    print("=" * 50)
    trainer.fit(model, train_loader, val_loader)
    print(f"\nFinished training {source} model.")
    print(f"Best validation accuracy: {trainer.callback_metrics.get('val_accuracy', 0):.4f}")
    print("=" * 50)
    
    return model

def train_multi_source_model(data_path, output_path, sources=['bass', 'drums', 'other', 'vocals']):
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for source in sources:
        create_metadata_files(data_path, source)

    source_models = []
    for source in sources:
        print(f"\nTraining model for {source} source...")
        model = train_source_model(data_path, source)
        source_models.append(model)
        torch.save(model.state_dict(), output_path / f"model_{source}.pt")


    dataset = MultiSourceDataset(data_path, sources)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    fusion_model = FusionModel(source_models)
    logger = TensorBoardLogger("tb_logs", name="fusion_model")
    
    trainer = L.Trainer(
        max_epochs=150,
        accelerator='gpu',
        devices=[0],
        logger=logger,
        callbacks=[
            L.callbacks.ModelCheckpoint(
                monitor='val_accuracy',
                mode='max',
                filename='best-fusion-{epoch:02d}-{val_accuracy:.4f}'
            ),
            L.callbacks.RichProgressBar()
        ]
    )
    
    trainer.fit(fusion_model, train_loader, val_loader)
    torch.save(fusion_model.state_dict(), output_path / "fusion_model.pt")
if __name__ == "__main__":
    data_path = Path("/content/MLSP_Project/features")
    output_path = Path("output")
    train_multi_source_model(data_path, output_path)