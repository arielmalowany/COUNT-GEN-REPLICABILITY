"""
Batch script: Generate Gender CFs for samples from existing counterfactual pkl files.
Replicates the Generate_Gender_CFs notebook logic in a loop.

Usage:
  python batch_generate_cfs.py                    # run all 21 existing samples
  python batch_generate_cfs.py --batch-size 5     # run 5 random samples from existing
  python batch_generate_cfs.py --samples 1088 9731  # run specific samples
"""

import argparse
import glob
import random
import os
import torch
import json
import torch.utils.data as data
import pickle as pkl
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import MultiComparator, DominanceWithConstraintsComparator
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from my_attgan import AttGAN
from my_mivolo_inference import mivolo_inference
from cf_utils import *
from generate_gender_cfs import AttGanPlausibleCounterfactualProblem

# ============================================================================
# ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description="Batch generate gender counterfactuals")
parser.add_argument("--batch-size", type=int, default=None, help="Number of random samples to pick from existing (default: all)")
parser.add_argument("--samples", nargs='+', type=str, default=None, help="Specific sample folder names to process (e.g. 1088 9731)")
args = parser.parse_args()

# ============================================================================
# DISCOVER EXISTING SAMPLES
# ============================================================================

cf_dir = './Counterfactuals'
all_pkl_files = sorted(glob.glob(os.path.join(cf_dir, 'Front_*_0.pkl')))
all_folders = [os.path.basename(f).split('_')[1] for f in all_pkl_files]

if args.samples:
    selected_folders = [s for s in args.samples if s in all_folders]
    missing = [s for s in args.samples if s not in all_folders]
    if missing:
        print(f"Warning: samples not found: {missing}")
elif args.batch_size:
    selected_folders = random.sample(all_folders, min(args.batch_size, len(all_folders)))
else:
    selected_folders = all_folders

print(f"Found {len(all_folders)} existing samples. Processing {len(selected_folders)}: {selected_folders}")

# ============================================================================
# SETUP
# ============================================================================

print("Loading settings and AttGAN...")

with open('./384_shortcut1_inject1_none_hq/setting.txt', 'r') as f:
    gan_args = json.load(f)

base_attrs = gan_args.get('attrs')

attgan = AttGAN(gan_args)
attgan.load('./384_shortcut1_inject1_none_hq/weights.149.pth')
attgan.eval()

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

pop_size = 100
max_evals = 1000
use_lpips = True

# ============================================================================
# BATCH GENERATION
# ============================================================================

total = len(selected_folders)

for sample_idx, folder in enumerate(selected_folders):

    pkl_path = os.path.join(cf_dir, f'Front_{folder}_0.pkl')

    # Load factual image and attributes from existing pkl
    with open(pkl_path, 'rb') as f:
        _ = pkl.load(f)  # pareto_front (skip)
        factual_img = pkl.load(f)
        factual_atts = pkl.load(f)

    img_file_name = f'{folder}.jpg'
    print(f"\n{'='*60}")
    print(f"[{sample_idx+1}/{total}] Processing: {img_file_name} (from {pkl_path})")
    print(f"{'='*60}")

    # Reconstruction for LPIPS baseline
    factual_img_recons = attgan.G(factual_img, factual_atts)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none')
    lpips_recons = lpips_metric(factual_img, factual_img_recons).item()

    # Gender prediction
    prediction_orig, _ = mivolo_inference(factual_img, True)

    if prediction_orig >= 0.5:
        desired_pred = 0
        gender_str = "Female"
    else:
        desired_pred = 1
        gender_str = "Male"

    print(f"  Original prediction: {prediction_orig:.3f} ({gender_str}) -> Desired: {desired_pred}")

    if use_lpips:
        original_discriminator_score = lpips_recons

    # Define problem
    problem = AttGanPlausibleCounterfactualProblem(
        image=factual_img,
        code=factual_atts,
        decoder=attgan.G,
        discriminator=attgan.D,
        classifier=mivolo_inference,
        original_pred=prediction_orig,
        original_discriminator_score=original_discriminator_score,
        desired_pred=desired_pred,
        use_lpips=use_lpips,
        non_actionable_features=None
    )

    # Define algorithm
    algorithm = NSGAII(
        problem=problem,
        population_size=pop_size,
        offspring_population_size=pop_size,
        selection=BinaryTournamentSelection(
            MultiComparator([DominanceWithConstraintsComparator(), CrowdingDistance.get_comparator()])
        ),
        mutation=PolynomialMutation(
            probability=1/problem._number_of_variables,
            distribution_index=20),
        crossover=SBXCrossover(probability=0.9, distribution_index=20),
        termination_criterion=StoppingByHV(expected_value=0.010, max_steps=3, max_evaluations=max_evals),
        dominance_comparator=DominanceWithConstraintsComparator()
    )

    progress_bar = ProgressBarObserver(max=max_evals)
    algorithm.observable.register(progress_bar)
    hipervolume_indicator = HyperVolumeObserver(frequency=100, img_file_name=img_file_name)
    algorithm.observable.register(hipervolume_indicator)

    # Run
    algorithm.run()
    pareto_front = algorithm.result()
    runtime_in_seconds = round(algorithm.total_computing_time, 3)
    evals = algorithm.evaluations

    # Save results
    write_pkl_file(img_file_name, algorithm, problem, factual_img, factual_atts,
                   pareto_front, runtime_in_seconds, prediction_orig, desired_pred,
                   pop_size, max_evals, evals)

    valid_count = len([s for s in pareto_front if s.objectives[1] < 0.5])
    print(f"  Done! Runtime: {runtime_in_seconds}s | Evals: {evals} | Valid CFs: {valid_count}/{len(pareto_front)}")

print(f"\n{'='*60}")
print("Batch complete! All results saved to ./Counterfactuals/")
print(f"{'='*60}")
