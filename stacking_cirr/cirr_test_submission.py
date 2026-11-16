import json
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRDataset, base_path
from utils import extract_index_features
from models import CIRPlus


def generate_cirr_test_submissions(file_name: str, model: CIRPlus):
    # Define the dataset and extract index features
    classic_test_dataset = CIRDataset('cirr', 'test1', 'classic', args.data_path)
    index_features, index_names = extract_index_features(classic_test_dataset, model)
    relative_test_dataset = CIRDataset('cirr', 'test1', 'relative', args.data_path)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset, model,
                                                                                  index_features, index_names)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = base_path / "submission" 
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print("Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRDataset, model: CIRPlus, index_features: torch.tensor,
                             index_names: List[str]) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    # Generate predictions
    reference_names, group_members, pairs_id = \
        generate_cirr_test_predictions(model, relative_test_dataset, index_names,
                                       index_features)

    print("Compute CIRR prediction dicts")
    
    pairs_id_int = [int(t.item()) for t in pairs_id]
    clip_fea = torch.load('feature_cirr/clip/test_feature.pt')
    clip_fea = itemgetter(*pairs_id_int)(clip_fea)
    clip_fea = torch.stack(clip_fea, dim=0)
    tgcir_fea = torch.load('feature_cirr/tgcir/test_feature.pt')
    tgcir_fea = itemgetter(*pairs_id_int)(tgcir_fea)
    tgcir_fea = torch.stack(tgcir_fea, dim=0)
    blip_fea = torch.load('feature_cirr/blip/test_feature.pt')
    blip_fea = itemgetter(*pairs_id_int)(blip_fea)
    blip_fea = torch.stack(blip_fea, dim=0)
    blip2_fea = torch.load('feature_cirr/blip2/test_feature.pt')
    blip2_fea = itemgetter(*pairs_id_int)(blip2_fea)
    blip2_fea = torch.stack(blip2_fea, dim=0)
    query_features = torch.stack([clip_fea, tgcir_fea, blip_fea, blip2_fea], dim=1)
    distances = -model.net(query_features)

    # Compute the distances and sort the results
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(model: CIRPlus, relative_test_dataset: CIRDataset,
                                   index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    print("Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=True)

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(
            relative_test_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    return reference_names, group_members, pairs_id


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="submission file name")
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()
    model = CIRPlus('cirr')
    model.eval()
    model.load_ckpt()
    generate_cirr_test_submissions(args.submission_name, model)
