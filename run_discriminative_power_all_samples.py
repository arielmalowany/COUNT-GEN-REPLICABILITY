#!/usr/bin/env python3
"""
Script to compute Discriminative Power metric for all 21 counterfactual samples
"""

import argparse
import pickle as pkl
import torch
import numpy as np
import json
import torch.utils.data as data
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# custom scripts
from my_attgan import AttGAN
from my_mivolo_inference import mivolo_inference
from cf_utils import *
from data import CelebA_HQ_custom

# ============================================================================
# SETUP
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--test_set_size',   type=int,   default=100,    help='Total number of test samples per front')
parser.add_argument('--distance_metric', type=str,   default='lpips', choices=['lpips', 'pixel_wise', 'ssim'], help='Distance metric for image comparisons')
parser.add_argument('--distance_img_size', type=int, default=384,    help='Image size used inside distances_between_image_sets')
args = parser.parse_args()

print("Loading models and data...")

# Load settings
image_size = 384
with open('./384_shortcut1_inject1_none_hq/setting.txt', 'r') as f:
    gan_args = json.load(f)

training_set_size = 1000
training_set_image_size = 384
test_set_size    = args.test_set_size
distance_metric  = args.distance_metric
distance_img_size = args.distance_img_size

# Load AttGAN
attgan = AttGAN(gan_args)
attgan.load('./384_shortcut1_inject1_none_hq/weights.149.pth')
attgan.eval()

# Load CelebaHQ training set
celeba_path = './celeba_hq_dataset/CelebA-HQ-img'
atts_path = './celeba_hq_dataset/CelebAMask-HQ-attribute-anno.txt'
base_attrs = gan_args.get('attrs')

sample_celeba_data = CelebA_HQ_custom(
    data_path=celeba_path,
    attr_path=atts_path,
    selected_attrs=base_attrs,
    image_size=training_set_image_size,
    mode='train'
)

sample_celeba_dataloader = data.DataLoader(
    sample_celeba_data, 
    batch_size=training_set_size, 
    num_workers=gan_args.get('num_workers'),
    shuffle=True, 
    drop_last=False
)

# Load batch of N random images
data_iterator = iter(sample_celeba_dataloader)
training_set_images, training_set_attributes, training_set_names = next(data_iterator)

print(f"Loaded training set with {len(training_set_images)} images")

# Balance training set by gender using the Male attribute annotation
male_attr_idx = base_attrs.index('Male')
male_mask = training_set_attributes[:, male_attr_idx] > 0

male_idx   = male_mask.nonzero(as_tuple=True)[0]
female_idx = (~male_mask).nonzero(as_tuple=True)[0]

n_per_class = min(len(male_idx), len(female_idx))
balanced_idx = torch.cat([male_idx[:n_per_class], female_idx[:n_per_class]])

training_set_images     = training_set_images[balanced_idx]
training_set_attributes = training_set_attributes[balanced_idx]

coverage = test_set_size / len(training_set_images)

print(f"Balanced training set: {n_per_class} male + {n_per_class} female = {len(training_set_images)} images")
print(f"Coverage: {coverage:.4f} → ~{test_set_size} test samples per front")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def gender_training_set(training_set=training_set_images, threshold=0.5):
    """Predict gender labels for training set"""
    training_set_gender = []
    
    for sample in training_set: 
        gender_prob, factual_age = mivolo_inference(sample, True)
        if gender_prob >= 0.5:
            pred_gender = 1
        else:
            pred_gender = 0
        training_set_gender.append(pred_gender)
    
    return training_set_gender


def gender_test_set(training_images, training_labels, factual_image, coverage=0.10, img_size=384, distance_metric='lpips'):
    """Create test set for discriminative power evaluation"""

    # Create distance matrices
    factual_to_training_distances, factual_to_training_indices = distances_between_image_sets(
        factual_image,
        img_size=img_size,
        training_set=training_images,
        distance_metric=distance_metric
    )
    
    # Select k examples from positive and negative classes
    n = len(training_labels)
    n_female = np.sum(training_labels)
    n_male = n - n_female
    
    m_male = round(n_male * coverage)
    m_female = round(n_female * coverage)
    
    # FIXED: Look up the label for each index from the sorted distance list
    female_indices = [i for i in factual_to_training_indices if training_labels[i] == 1][0:m_female]
    male_indices = [i for i in factual_to_training_indices if training_labels[i] == 0][0:m_male]
    
    male_samples = torch.index_select(training_images, 0, torch.tensor(male_indices))
    female_samples = torch.index_select(training_images, 0, torch.tensor(female_indices))
    
    test_set_samples = torch.cat((female_samples, male_samples), dim=0)
    test_set_gender = torch.cat((torch.ones(m_female, 1), torch.zeros(m_male, 1)))
    
    closest_data_idx_male = male_indices[0]
    closest_data_idx_female = female_indices[0]
    
    return test_set_samples, test_set_gender, closest_data_idx_male, closest_data_idx_female


def predict_gender(image, training_set_knn, training_set_gender_knn, img_size=384, distance_metric='lpips'):
    """Predict gender using 1-NN classifier"""
    distances, indices = distances_between_image_sets(
        image,
        img_size=img_size,
        training_set=training_set_knn,
        distance_metric=distance_metric
    )
    nearest_gender = training_set_gender_knn[indices[0]]
    if nearest_gender >= 0.5:
        p = 1
    else:
        p = 0
    return p


def compute_discriminative_power(folder, training_set_gender, coverage, img_size=384, distance_metric='lpips', front=0):
    """Compute discriminative power metric for a single sample"""
    
    # Load counterfactual data
    path = os.path.join('./Counterfactuals', f'Front_{folder}_{front}.pkl')
    
    with open(path, 'rb') as f:
        pareto_front = pkl.load(f)
        factual_img = pkl.load(f)
        factual_atts = pkl.load(f)
        runtime_in_seconds = pkl.load(f)
        experiment_metadata = pkl.load(f)
    
    raw_x_data, raw_y_data, raw_z_data, new_preds, new_attributes, generated_cfs, dominance_ranking, crowding_distances = unpack_front(pareto_front)
    
    # Get valid CFs
    valid_cfs_idx = [i for i, y in enumerate(raw_y_data) if y < 0.5]
    valid_cf_images = torch.stack([generated_cfs[i] for i in valid_cfs_idx]).squeeze(1)
    count_valid_cfs = len(valid_cfs_idx)
    
    if count_valid_cfs == 0:
        return {
            'folder': folder,
            'count_valid_cfs': 0,
            'accuracy': None,
            'error': 'No valid CFs'
        }
    
    # Define factual image
    factual_image = factual_img
    factual_gender_prob, factual_age = mivolo_inference(factual_image, True)
    
    # Create test set
    test_set_samples, test_set_gender, closest_data_idx_male, closest_data_idx_female = gender_test_set(
        training_set_images,
        training_set_gender,
        factual_image,
        coverage=coverage,
        img_size=img_size,
        distance_metric=distance_metric
    )
    
    # Training set for 1-NN Classifier
    training_set_knn = torch.concat([valid_cf_images, factual_image])
    training_set_gender_knn = gender_training_set(valid_cf_images)
    training_set_gender_knn.append(factual_gender_prob)
    
    # Predict Gender on test set
    test_set_predictions = []
    for sample in test_set_samples:
        p = predict_gender(sample, training_set_knn, training_set_gender_knn, img_size=img_size, distance_metric=distance_metric)
        test_set_predictions.append(p)
    
    test_set_predictions = torch.tensor(test_set_predictions).unsqueeze(1)
    
    # Calculate accuracy
    accuracy = torch.sum(test_set_predictions == test_set_gender) / test_set_gender.shape[0]
    
    return {
        'folder': folder,
        'count_valid_cfs': count_valid_cfs,
        'accuracy': round(accuracy.item(), 3),
        'test_set_size': len(test_set_gender),
        'factual_gender_prob': round(factual_gender_prob, 3)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# List of all 21 sample folders
sample_folders = [
    '10131', '1088', '11382', '11462', '1306', '13245', '13407',
    '14218', '16255', '17488', '19256', '20632', '2162', '21724',
    '22092', '22252', '22899', '23164', '24975', '25613', '9731'
]

print("Computing training set gender labels...")
training_set_gender = gender_training_set()
print(f"Training set gender labels computed ({len(training_set_gender)} labels)")

print(f"\nProcessing {len(sample_folders)} samples...\n")

results = []

for folder in tqdm(sample_folders, desc="Computing discriminative power"):
    try:
        result = compute_discriminative_power(folder, training_set_gender, coverage, img_size=distance_img_size, distance_metric=distance_metric)
        results.append(result)
        print(f"✓ Sample {folder}: Accuracy = {result['accuracy']}, Valid CFs = {result['count_valid_cfs']}")
    except Exception as e:
        error_result = {
            'folder': folder,
            'count_valid_cfs': None,
            'accuracy': None,
            'error': str(e)
        }
        results.append(error_result)
        print(f"✗ Sample {folder}: Error - {str(e)}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Create DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
output_csv = 'discriminative_power_results.csv'
df_results.to_csv(output_csv, index=False)
print(f"\n✓ Results saved to {output_csv}")

# Save to Excel
output_xlsx = 'discriminative_power_results.xlsx'
df_results.to_excel(output_xlsx, index=False, sheet_name='Discriminative Power')
print(f"✓ Results saved to {output_xlsx}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total samples processed: {len(results)}")
valid_results = df_results[df_results['accuracy'].notna()]
if len(valid_results) > 0:
    print(f"Mean accuracy: {valid_results['accuracy'].mean():.3f}")
    print(f"Std accuracy: {valid_results['accuracy'].std():.3f}")
    print(f"Min accuracy: {valid_results['accuracy'].min():.3f}")
    print(f"Max accuracy: {valid_results['accuracy'].max():.3f}")
    print(f"Mean valid CFs: {valid_results['count_valid_cfs'].mean():.1f}")
print("\nDetailed results:")
print(df_results.to_string(index=False))
print("="*60)
