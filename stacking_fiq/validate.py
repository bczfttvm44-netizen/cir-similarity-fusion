import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import extract_index_features, collate_fn
from data_utils import CIRDataset
from models import CIRPlus


def compute_fiq_val_metrics(relative_val_dataset: CIRDataset, model, index_features: torch.tensor,
                            index_names: List[str], device=torch.device('cpu')) -> Tuple[float, float]:
    # Generate predictions
    target_names, refer_names = generate_fiq_val_predictions(model, relative_val_dataset,
                                                                                 index_names, index_features, device)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    dress_types = relative_val_dataset.dress_types[0]
    clip_fea = torch.load('feature_fiq/clip/val_feature_{}.pt'.format(dress_types))
    clip_fea = itemgetter(*range(len(refer_names)))(clip_fea)
    clip_fea = torch.stack(clip_fea, dim=0)
    tgcir_fea = torch.load('feature_fiq/tgcir/val_feature_{}.pt'.format(dress_types))
    tgcir_fea = itemgetter(*range(len(refer_names)))(tgcir_fea)
    tgcir_fea = torch.stack(tgcir_fea, dim=0)
    blip_fea = torch.load('feature_fiq/blip/val_feature_{}.pt'.format(dress_types))
    blip_fea = itemgetter(*range(len(refer_names)))(blip_fea)
    blip_fea = torch.stack(blip_fea, dim=0)
    blip2_fea = torch.load('feature_fiq/blip2/val_feature_{}.pt'.format(dress_types))
    blip2_fea = itemgetter(*range(len(refer_names)))(blip2_fea)
    blip2_fea = torch.stack(blip2_fea, dim=0)
    query_features = torch.stack([clip_fea, tgcir_fea, blip_fea, blip2_fea], dim=1)
    distances = -model.net(query_features)
    
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    N = distances.shape[0]
    recall_at10 = 0
    recall_at50 = 0
    for i in range(N):
        sorted_index_name = sorted_index_names[i]
        if target_names[i] in sorted_index_name[:10]:
            recall_at10 += 1
            recall_at50 += 1
        elif target_names[i] in sorted_index_name[:50]:
            recall_at50 += 1

    # Compute the metrics
    recall_at10 = (recall_at10 / N) * 100
    recall_at50 = (recall_at50 / N) * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(model, relative_val_dataset: CIRDataset, index_names: List[str],
                                 index_features: torch.tensor, device=torch.device('cuda')) -> \
        Tuple[torch.tensor, List[str], List[str]]:
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    target_names = []
    refer_names = []
    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data
        target_names.extend(batch_target_names)
        refer_names.extend(reference_names)

    return target_names, refer_names


def fashioniq_val_retrieval(dress_type: str, model):
    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRDataset('fiq', 'val', 'classic', args.data_path, [dress_type])
    index_features, index_names = extract_index_features(classic_val_dataset, model)
    relative_val_dataset = CIRDataset('fiq', 'val', 'relative', args.data_path, [dress_type])

    return compute_fiq_val_metrics(relative_val_dataset, model, index_features, index_names)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()

    model = CIRPlus('fiq')
    model.eval()
    model.load_ckpt()

    average_recall10_list = []
    average_recall50_list = []

    shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', model)
    average_recall10_list.append(shirt_recallat10)
    average_recall50_list.append(shirt_recallat50)

    dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', model)
    average_recall10_list.append(dress_recallat10)
    average_recall50_list.append(dress_recallat50)

    toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', model)
    average_recall10_list.append(toptee_recallat10)
    average_recall50_list.append(toptee_recallat50)

    print(f"{dress_recallat10 = }")
    print(f"{dress_recallat50 = }")

    print(f"\n{shirt_recallat10 = }")
    print(f"{shirt_recallat50 = }")

    print(f"{toptee_recallat10 = }")
    print(f"{toptee_recallat50 = }")

    print(f"average recall10 = {mean(average_recall10_list)}")
    print(f"average recall50 = {mean(average_recall50_list)}")
        
    
