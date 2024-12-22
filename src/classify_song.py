from pathlib import Path
import torch
import numpy as np
import librosa
import subprocess
import tempfile
import shutil
from encodecmae_to_wav.hub import load_model
from torch.nn import functional as F

from train_multi_linear import SourcePECMAEModule, FusionModel

def separate_sources(input_file, output_dir):
    """Separate audio sources using demucs"""
    print("Separating sources with demucs")
    cmd = [
        "demucs",
        str(input_file),
        "-o", str(output_dir)
    ]
    subprocess.run(cmd, check=True)
    return output_dir / "htdemucs" / input_file.stem

def extract_features(audio_file, encoder, device="cuda:0", sr=24000):
    """Extract features from audio file using EnCodecMAE"""
    print(f"Extracting features from {audio_file}")
    audio, _ = librosa.load(audio_file, sr=sr)
    
    with torch.no_grad():
        features = encoder.encode(audio)
    
    return features.detach().cpu().numpy()

def load_models(model_dir, sources=['bass', 'drums', 'other', 'vocals']):
    """Load all trained models"""
    print("Loading models")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    source_models = []
    for source in sources:
        model = SourcePECMAEModule()
        model.load_state_dict(torch.load(model_dir / f"model_{source}.pt", weights_only=True))
        model = model.to(device)
        model.eval()
        source_models.append(model)
    
    # load fusion model
    fusion_model = FusionModel(source_models)
    fusion_model.load_state_dict(torch.load(model_dir / "fusion_model.pt", weights_only=True))
    fusion_model = fusion_model.to(device)
    fusion_model.eval()
    
    return source_models, fusion_model

def classify_song(input_file, model_dir, genre_map=None):
    """Classify a single song using the trained models"""
    if genre_map is None:
        genre_map = {
            0: 'blues', 1: 'classical', 2: 'country', 3: 'disco',
            4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sources = ['bass', 'drums', 'other', 'vocals']
    
    # create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        separated_dir = separate_sources(input_file, temp_dir)
        
        encoder = load_model("DiffTransformerAE2L8L1CLS-4s", device=device)
        
        source_features = []
        for source in sources:
            source_file = separated_dir / f"{source}.wav"
            features = extract_features(source_file, encoder, device)
            features = torch.FloatTensor(features).to(device)
            source_features.append(features)
            print(f"{source}: {features.shape} segments")
        
        source_models, fusion_model = load_models(model_dir, sources)
        
        source_embeddings = []
        source_predictions = []
        
        with torch.no_grad():
            for source_idx, (model, features) in enumerate(zip(source_models, source_features)):
                #get embeddings from encoder part of the source model
                embeddings = model.encoder(features)  # [n_segments, hidden_dim]
                source_embeddings.append(embeddings)
                
                output = model.classifier(embeddings)  # [n_segments, n_classes]
                probs = F.softmax(output, dim=1)
                
                # calculate average probabilities over segments
                avg_probs = probs.mean(dim=0)
                pred = avg_probs.argmax().item()
                source_predictions.append({
                    'source': sources[source_idx],
                    'prediction': genre_map[pred],
                    'confidence': avg_probs[pred].item(),
                    'segment_predictions': probs.argmax(dim=1).cpu().numpy()
                })
                print(f"{sources[source_idx]} prediction: {genre_map[pred]} ({avg_probs[pred]:.3f})")
        
        with torch.no_grad():
            fusion_output = fusion_model.fusion(torch.cat(source_embeddings, dim=1))
            fusion_probs = F.softmax(fusion_output, dim=1)
            
            # average probabilities over segments
            avg_fusion_probs = fusion_probs.mean(dim=0)
            fusion_pred = avg_fusion_probs.argmax().item()
            
            top3_probs, top3_indices = torch.topk(avg_fusion_probs, 3)
        
        print("\nClassification Results:")
        print("-" * 50)
        print(f"Final prediction (Fusion model): {genre_map[fusion_pred]}")
        print(f"Confidence: {avg_fusion_probs[fusion_pred]:.3f}")
        
        print("\nTop 3 genres (Fusion model):")
        for prob, idx in zip(top3_probs.cpu().numpy(), top3_indices.cpu().numpy()):
            print(f"{genre_map[idx]}: {prob:.3f}")
        
        print("\nPredictions by source:")
        for pred_info in source_predictions:
            print(f"{pred_info['source']}: {pred_info['prediction']} "
                  f"(confidence: {pred_info['confidence']:.3f})")
        
        segment_predictions = fusion_probs.argmax(dim=1).cpu().numpy()
        n_segments = len(segment_predictions)
        segment_counts = np.bincount(segment_predictions, minlength=len(genre_map))
        segment_percentages = segment_counts / n_segments * 100
        
        print(f"\nSegment analysis (total {n_segments} segments of 4 seconds each):")
        for genre_idx, percentage in enumerate(segment_percentages):
            if percentage > 0:
                print(f"{genre_map[genre_idx]}: {percentage:.1f}%")
        
        return {
            'final_prediction': genre_map[fusion_pred],
            'confidence': avg_fusion_probs[fusion_pred].item(),
            'top3_genres': [(genre_map[idx], prob) for prob, idx in zip(top3_probs.cpu().numpy(), top3_indices.cpu().numpy())],
            'source_predictions': source_predictions,
            'segment_distribution': {genre_map[i]: p for i, p in enumerate(segment_percentages) if p > 0},
            'n_segments': n_segments
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify a music file using trained models')
    parser.add_argument('input_file', type=Path, help='Input music file (mp3)')
    parser.add_argument('model_dir', type=Path, help='Directory containing trained models')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed predictions')
    
    args = parser.parse_args()
    
    results = classify_song(args.input_file, args.model_dir)
    
    if args.verbose:
        print("\nDetailed segment analysis for each source:")
        for source_pred in results['source_predictions']:
            print(f"\n{source_pred['source'].capitalize()}:")
            segment_counts = np.bincount(source_pred['segment_predictions'])
            for genre_idx, count in enumerate(segment_counts):
                if count > 0:
                    percentage = count / results['n_segments'] * 100
                    print(f"{genre_map[genre_idx]}: {percentage:.1f}%")