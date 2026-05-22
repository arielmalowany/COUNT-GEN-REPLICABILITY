"""
Batch processing script to run CF evaluation metrics on all pickle files
in the Counterfactuals folder.
"""

import pickle as pkl
import torch
import numpy as np
import json
import torch.utils.data as data
import os
import sys
from pathlib import Path
import pandas as pd
import traceback
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
import argparse
from tqdm import tqdm
import logging

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

from my_attgan import AttGAN
from my_mivolo_inference import mivolo_inference
from cf_utils import *
from generate_gender_cfs import AttGanPlausibleCounterfactualProblem
from data import CelebA_HQ_custom


def load_models_and_data(gan_checkpoint_path='./384_shortcut1_inject1_none_hq', training_set_size=1000):
    """Load AttGAN model and training data"""
    print("Loading models and data...")

    # Load settings
    image_size = 384
    with open(f'{gan_checkpoint_path}/setting.txt', 'r') as f:
        gan_args = json.load(f)

    # Load AttGAN
    attgan = AttGAN(gan_args)
    attgan.load(f'{gan_checkpoint_path}/weights.149.pth')
    attgan.eval()

    # Load CelebA-HQ samples
    celeba_path = './celeba_hq_dataset/CelebA-HQ-img'
    atts_path = './celeba_hq_dataset/CelebAMask-HQ-attribute-anno.txt'
    base_attrs = gan_args.get('attrs')

    sample_celeba_data = CelebA_HQ_custom(
        data_path=celeba_path,
        attr_path=atts_path,
        selected_attrs=base_attrs,
        image_size=image_size,
        mode='train'
    )

    sample_celeba_dataloader = data.DataLoader(
        sample_celeba_data,
        batch_size=training_set_size,
        num_workers=gan_args.get('num_workers'),
        shuffle=True,
        drop_last=False
    )

    # Get one batch of training data
    data_iterator = iter(sample_celeba_dataloader)
    training_set_images, training_set_attributes, training_set_names = next(data_iterator)

    print("Models and data loaded successfully!")
    return attgan, gan_args, training_set_images, training_set_attributes


def load_counterfactuals(pkl_path):
    """Load counterfactuals from pickle file"""
    with open(pkl_path, 'rb') as f:
        pareto_front = pkl.load(f)
        factual_img = pkl.load(f)
        factual_atts = pkl.load(f)
        runtime_in_seconds = pkl.load(f)
        experiment_metadata = pkl.load(f)

    raw_x_data, raw_y_data, raw_z_data, new_preds, new_attributes, generated_cfs, dominance_ranking, crowding_distances = unpack_front(pareto_front)

    return {
        'pareto_front': pareto_front,
        'factual_img': factual_img,
        'factual_atts': factual_atts,
        'runtime_in_seconds': runtime_in_seconds,
        'experiment_metadata': experiment_metadata,
        'raw_x_data': raw_x_data,
        'raw_y_data': raw_y_data,
        'raw_z_data': raw_z_data,
        'new_preds': new_preds,
        'new_attributes': new_attributes,
        'generated_cfs': generated_cfs,
        'dominance_ranking': dominance_ranking,
        'crowding_distances': crowding_distances
    }


def compute_evaluation_metrics(cf_data, attgan, gan_args, training_set_images, training_set_attributes):
    """Compute all evaluation metrics for a set of counterfactuals"""

    # Extract valid CFs
    valid_cfs_idx = [i for i, y in enumerate(cf_data['raw_y_data']) if y < 0.5]

    if len(valid_cfs_idx) == 0:
        print("  Warning: No valid counterfactuals found!")
        return None

    valid_cf_images = torch.stack([cf_data['generated_cfs'][i] for i in valid_cfs_idx]).squeeze(1)
    count_valid_cfs = len(valid_cfs_idx)
    valid_cf_atts = torch.stack([cf_data['new_attributes'][i] for i in valid_cfs_idx])

    raw_x_data = [cf_data['raw_x_data'][i] for i in valid_cfs_idx]
    raw_y_data = [cf_data['raw_y_data'][i] for i in valid_cfs_idx]
    raw_z_data = [cf_data['raw_z_data'][i] for i in valid_cfs_idx]

    # Define factual image
    factual_image = cf_data['factual_img']
    factual_atts = cf_data['factual_atts']
    factual_gender_prob, factual_age = mivolo_inference(factual_image, True)
    factual_gender_binary = 1 if factual_gender_prob > 0.5 else 0

    # Compute LPIPS reconstruction score for factual image
    factual_img_recons = attgan.G(factual_image, factual_atts)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none')
    lpips_recons = lpips_metric(factual_image, factual_img_recons).item()

    # Initialize metrics dictionary
    evaluation_metrics = {}
    pop_size = len(cf_data['generated_cfs'])

    # SIZE
    size = count_valid_cfs / pop_size
    evaluation_metrics["Size"] = {
        "size_pct": {"value": size, "definition": "Available CFs wrt required CFs"},
        "size_count": {"value": count_valid_cfs, "definition": "Available CFs"},
        "required_cfs": {"value": pop_size, "definition": "Required CFs"}
    }

    # PROXIMITY
    avg_distance = np.mean(raw_x_data)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none')
    lpips_score = 0
    for cf in valid_cf_images:
        lpips_score += lpips(factual_image, cf.unsqueeze(0))
    lpips_score /= count_valid_cfs

    # MINIMALITY
    features_changed = torch.abs(valid_cf_atts - factual_atts) > 0.1
    avg_features_changed = torch.sum(torch.sum(features_changed, axis=1)) / (features_changed.shape[0] * features_changed.shape[1])

    evaluation_metrics["Dissimilarity"] = {
        "proximity": {
            "avg_distance_features": {"value": round(avg_distance.item(), 3), "definition": "Average distance between the factual image features and valid CFs features"},
            "avg_distance_images": {"value": round(lpips_score.item(), 3), "definition": "Average LPIPS score between the factual image and valid CFs"}
        },
        "minimality": {"avg_features_changed": {"value": round(avg_features_changed.item(), 3), "definition": "Average number of features changed for valid CFs"}}
    }

    # DIVERSITY
    distances = torch.sqrt(((valid_cf_atts - factual_atts) ** 2).sum(-1))
    mean_distance = torch.mean(distances)

    attribute_deviations = torch.std(valid_cf_atts, dim=0)
    attribute_means = torch.mean(valid_cf_atts, dim=0)
    std_attribute_value = torch.mean(attribute_deviations)

    average_lpips_distance = 0
    for cf in valid_cf_images:
        average_lpips_distance += torch.sum(distances_between_image_sets(cf, 128, valid_cf_images, 'lpips')[0][1:])
    average_lpips_distance /= (count_valid_cfs * (count_valid_cfs - 1))

    mean_cf_pixelwise_distance = 0
    for cf in valid_cf_images:
        mean_cf_pixelwise_distance += torch.sum(distances_between_image_sets(cf, 128, valid_cf_images, 'pixel_wise')[0][1:])
    mean_cf_pixelwise_distance /= (count_valid_cfs * (count_valid_cfs - 1))

    evaluation_metrics["Diversity"] = {
        "avg_distance_features_L2": {"value": round(mean_distance.item(), 3), "definition": "Average distance between the factual image features and valid CFs features (L2 distance)"},
        "mean_cf_pixelwise_distance": {"value": round(mean_cf_pixelwise_distance.item(), 3), "definition": "Average pixel-wise distance of the CF set"},
        "mean_cf_lpips_distance": {"value": round(average_lpips_distance.item(), 3), "definition": "Average LPIPS distance of the CF set"},
        "mean_std": {"value": round(std_attribute_value.item(), 3), "definition": "Std deviation across all CF's attributes"},
        "attribute_means": {"value": attribute_means, "definition": "Average value of CF's attributes"},
        "attribute_changes": {"value": torch.sum(features_changed, axis=0) / count_valid_cfs, "definition": "Number of CFs that had changed the attribute's value"},
        "attribute_deviations": {"value": attribute_deviations, "definition": "Standard deviation of CF's attributes"},
    }

    # DISCRIMINATIVE POWER
    def gender_training_set(training_set, threshold=0.5):
        training_set_gender = []
        for sample in training_set:
            gender_prob, _ = mivolo_inference(sample, True)
            pred_gender = 1 if gender_prob >= 0.5 else 0
            training_set_gender.append(pred_gender)
        return training_set_gender

    def gender_test_set(training_images, training_labels, coverage=0.30):
        factual_to_training_distances, factual_to_training_indices = distances_between_image_sets(
            factual_image, img_size=128, training_set=training_images, distance_metric='pixel_wise'
        )

        n = len(training_labels)
        n_female = np.sum(training_labels)
        n_male = n - n_female

        m_male = round(n_male * coverage)
        m_female = round(n_female * coverage)

        female_indices = [i for i in factual_to_training_indices if training_labels[i] == 1][:m_female]
        male_indices = [i for i in factual_to_training_indices if training_labels[i] == 0][:m_male]

        male_samples = torch.index_select(training_images, 0, torch.tensor(male_indices))
        female_samples = torch.index_select(training_images, 0, torch.tensor(female_indices))

        test_set_samples = torch.cat((female_samples, male_samples), dim=0)
        test_set_gender = torch.cat((torch.ones(m_female, 1), torch.zeros(m_male, 1)))

        closest_data_idx_male = male_indices[0]
        closest_data_idx_female = female_indices[0]
        return test_set_samples, test_set_gender, closest_data_idx_male, closest_data_idx_female

    training_set_gender = gender_training_set(training_set_images)
    test_set_samples, test_set_gender, closest_data_idx_male, closest_data_idx_female = gender_test_set(
        training_set_images, training_set_gender
    )

    training_set_knn = torch.concat([valid_cf_images, factual_image])
    training_set_gender_knn = gender_training_set(valid_cf_images)
    training_set_gender_knn.append(factual_gender_prob)

    def predict_gender(image, training_set_knn):
        distances, indices = distances_between_image_sets(image, img_size=128, training_set=training_set_knn, distance_metric='pixel_wise')
        nearest_gender = training_set_gender_knn[indices[0]]
        return 1 if nearest_gender >= 0.5 else 0

    test_set_predictions = []
    for sample in test_set_samples:
        p = predict_gender(sample, training_set_knn)
        test_set_predictions.append(p)

    test_set_predictions = torch.tensor(test_set_predictions).unsqueeze(1)
    accuracy = torch.sum(test_set_predictions == test_set_gender) / test_set_gender.shape[0]
    evaluation_metrics["Discriminative Power"] = {
        "1nn_accuracy": {"value": round(accuracy.item(), 3), "definition": "Accuracy of a 1-NN classifier trained on CFs + 1F"}
    }

    # RUNTIME
    evaluation_metrics["Runtime"] = {
        "execution_time": {"value": round(cf_data['runtime_in_seconds'], 3), "definition": "Algorithm's Runtime"}
    }

    # IMPLAUSIBILITY
    min_distances = torch.tensor(0.0)
    for cf in valid_cf_images:
        distances_result, _ = distances_between_image_sets(cf, img_size=128, training_set=training_set_images, distance_metric='lpips')
        min_d = distances_result[0]
        min_distances = min_distances + min_d
    average_min_distance = min_distances / valid_cf_images.shape[0]

    evaluation_metrics["Implausibility"] = {
        "average_min_distance_training": {"value": round(average_min_distance.item(), 3), "definition": "Average distance of the CFs from their closest instance in the training set"}
    }

    # INSTABILITY
    sample_idx = closest_data_idx_female if factual_gender_binary == 1 else closest_data_idx_male
    instability_image = training_set_images[sample_idx].unsqueeze(0)
    instability_gender_prob, _ = mivolo_inference(instability_image, True)
    instability_gender_binary = 1 if instability_gender_prob > 0.5 else 0

    desired_pred = 1 - instability_gender_binary
    max_evals = 500
    pop_size = 100

    problem = AttGanPlausibleCounterfactualProblem(
        image=instability_image,
        code=factual_atts,
        decoder=attgan.G,
        discriminator=attgan.D,
        classifier=mivolo_inference,
        original_pred=instability_gender_prob,
        original_discriminator_score=lpips_recons,
        desired_pred=desired_pred,
        use_lpips=True
    )

    algorithm = NSGAII(
        problem=problem,
        population_size=pop_size,
        offspring_population_size=pop_size,
        mutation=PolynomialMutation(
            probability=1 / problem._number_of_variables,
            distribution_index=20
        ),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evals)
    )

    algorithm.run()
    pareto_front = algorithm.result()

    instability_attributes_set = []
    for sol in pareto_front:
        x = float(sol.objectives[0])
        y = float(sol.objectives[1])
        z = float(sol.objectives[2])
        new_code = torch.tensor(sol.variables)
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            if y < 0.5:
                instability_attributes_set.append(new_code)

    if len(instability_attributes_set) > 0:
        instability_attributes_set = torch.stack(instability_attributes_set)
        d, _ = distances_between_image_sets(instability_image, img_size=128, training_set=factual_image, distance_metric='pixel_wise')
        instability_score = 0
        for cf in valid_cf_atts:
            distance_to_cf = torch.sqrt(((cf - instability_attributes_set) ** 2).sum(-1))
            instability_score += distance_to_cf
        instability_score = torch.sum(instability_score) / (instability_attributes_set.shape[0] * valid_cf_atts.shape[0]) * 1 / (1 + d)

        evaluation_metrics["Instability"] = {
            "instability": {"value": round(instability_score.item(), 3), "definition": "Distance between CF sets of similar instances"}
        }
    else:
        evaluation_metrics["Instability"] = {
            "instability": {"value": None, "definition": "Could not compute instability (no valid CFs for comparison image)"}
        }

    # FID SCORE
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(normalize_0_1(training_set_images), real=True)
    fid.update(normalize_0_1(valid_cf_images), real=False)
    fid_score = fid.compute()

    evaluation_metrics["FID_score"] = {
        "fid": {"value": round(fid_score.item(), 3), "definition": "FID score between valid CFs and training set"}
    }

    # HYPERVOLUME
    hv = hypervolume_indicator(cf_data['pareto_front'])
    evaluation_metrics["Hypervolume"] = {
        "hypervolume": {"value": round(hv.item(), 3), "definition": "Hypervolume indicator of the estimated Pareto front wrt the point (1, 1, 1)"}
    }

    return evaluation_metrics, cf_data['experiment_metadata']


def save_metrics_to_excel(evaluation_metrics, experiment_metadata, output_path):
    """Save evaluation metrics to Excel file"""
    metrics_df = pd.DataFrame([], columns=['Metric_Class', 'Metric_SubClass', 'Metric_Name', 'Value', 'Definition'])

    for key in evaluation_metrics.keys():
        for sub_key in evaluation_metrics[key]:
            if 'value' in evaluation_metrics[key][sub_key]:
                value = evaluation_metrics[key][sub_key]["value"]
                definition = evaluation_metrics[key][sub_key]["definition"]
                name = sub_key
                metric_class = key
                metric_subclass = None
                row = pd.DataFrame([[metric_class, metric_subclass, name, value, definition]],
                                   columns=['Metric_Class', 'Metric_SubClass', 'Metric_Name', 'Value', 'Definition'])
                metrics_df = pd.concat((metrics_df, row))
            else:
                for subkey2 in evaluation_metrics[key][sub_key]:
                    if 'value' in evaluation_metrics[key][sub_key][subkey2]:
                        value = evaluation_metrics[key][sub_key][subkey2]["value"]
                        definition = evaluation_metrics[key][sub_key][subkey2]["definition"]
                        name = subkey2
                        metric_class = key
                        metric_subclass = sub_key
                        row = pd.DataFrame([[metric_class, metric_subclass, name, value, definition]],
                                           columns=['Metric_Class', 'Metric_SubClass', 'Metric_Name', 'Value', 'Definition'])
                        metrics_df = pd.concat((metrics_df, row))

    for key in experiment_metadata:
        metric_class = 'Metadata'
        metric_subclass = 'Hyperparameter'
        name = key
        value = experiment_metadata[key]
        definition = None
        row = pd.DataFrame([[metric_class, metric_subclass, name, value, definition]],
                           columns=['Metric_Class', 'Metric_SubClass', 'Metric_Name', 'Value', 'Definition'])
        metrics_df = pd.concat((metrics_df, row))

    metrics_df.to_excel(output_path, index=False)


def process_single_file(pkl_path, attgan, gan_args, training_set_images, training_set_attributes, overwrite=False, verbose=False):
    """Process a single pickle file and generate evaluation metrics"""

    # Extract identifier from filename (e.g., Front_1088_0.pkl -> 1088)
    filename = Path(pkl_path).stem  # Gets 'Front_1088_0'
    parts = filename.split('_')
    identifier = parts[1]  # Gets '1088'
    front_number = parts[2]  # Gets '0'

    # Define output Excel path
    output_path = Path(pkl_path).parent / f"{identifier}_{front_number}.xlsx"

    # Check if output already exists
    if output_path.exists() and not overwrite:
        print(f"  Skipping {filename} (output already exists)")
        return {'status': 'skipped', 'file': filename, 'reason': 'output exists'}

    try:
        # Load counterfactuals
        cf_data = load_counterfactuals(pkl_path)

        # Compute metrics
        result = compute_evaluation_metrics(cf_data, attgan, gan_args, training_set_images, training_set_attributes)

        if result is None:
            return {'status': 'failed', 'file': filename, 'reason': 'no valid counterfactuals'}

        evaluation_metrics, experiment_metadata = result

        # Save to Excel
        save_metrics_to_excel(evaluation_metrics, experiment_metadata, output_path)

        return {'status': 'success', 'file': filename, 'output': str(output_path)}

    except Exception as e:
        error_msg = str(e)
        if verbose:
            error_msg = traceback.format_exc()
            print(f"\n  Error details for {filename}:")
            print(error_msg)
        return {'status': 'error', 'file': filename, 'error': error_msg}


def main():
    parser = argparse.ArgumentParser(description='Batch process counterfactual pickle files')
    parser.add_argument('--cf-dir', type=str, default='./Counterfactuals',
                        help='Directory containing counterfactual pickle files')
    parser.add_argument('--gan-checkpoint', type=str, default='./384_shortcut1_inject1_none_hq',
                        help='Path to GAN checkpoint directory')
    parser.add_argument('--training-size', type=int, default=1000,
                        help='Training set size for evaluation')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing Excel files')
    parser.add_argument('--pattern', type=str, default='Front_*_0.pkl',
                        help='Pattern to match pickle files')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed error messages')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process (for testing)')

    args = parser.parse_args()

    # Find all pickle files
    cf_dir = Path(args.cf_dir)
    pkl_files = sorted(cf_dir.glob(args.pattern))

    if not pkl_files:
        print(f"No pickle files found matching pattern '{args.pattern}' in {cf_dir}")
        return

    # Limit files if specified
    if args.limit:
        pkl_files = pkl_files[:args.limit]
        print(f"Found {len(pkl_files)} pickle files (limited to {args.limit})")
    else:
        print(f"Found {len(pkl_files)} pickle files to process")

    # Load models and data (once for all files)
    attgan, gan_args, training_set_images, training_set_attributes = load_models_and_data(
        args.gan_checkpoint, args.training_size
    )

    # Process each file
    results = []
    for pkl_path in tqdm(pkl_files, desc="Processing files"):
        print(f"\nProcessing: {pkl_path.name}")
        result = process_single_file(
            pkl_path, attgan, gan_args, training_set_images, training_set_attributes,
            args.overwrite, args.verbose
        )
        results.append(result)
        print(f"  Status: {result['status']}")
        if result['status'] == 'error' and not args.verbose:
            print(f"  Error: {result['error'][:100]}...")  # Show first 100 chars

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    errors = [r for r in results if r['status'] == 'error']
    skipped = [r for r in results if r['status'] == 'skipped']

    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Errors: {len(errors)}")
    print(f"Skipped: {len(skipped)}")

    if failed:
        print("\nFailed files:")
        for r in failed:
            print(f"  - {r['file']}: {r['reason']}")

    if errors:
        print("\nFiles with errors:")
        for r in errors:
            print(f"  - {r['file']}: {r['error']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
