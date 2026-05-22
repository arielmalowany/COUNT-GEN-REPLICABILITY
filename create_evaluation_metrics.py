# %% [markdown]
# ### Import libraries and custom scripts

# %%
import pickle as pkl
import torch
import numpy as np
import json
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
import os
import cv2
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
import matplotlib.pyplot as plt
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.core.quality_indicator import HyperVolume
import openpyxl

# custom scripts
from my_attgan import AttGAN
from my_mivolo_inference import mivolo_inference
from my_mivolo_inference import predictor
from cf_utils import *
from generate_gender_cfs import AttGanPlausibleCounterfactualProblem
from data import Custom
from data import CelebA_HQ_custom

# %%
# Load settings

image_size = 384
with open('./384_shortcut1_inject1_none_hq/setting.txt', 'r') as f:
    gan_args = json.load(f)
    
training_set_size = 1000
training_set_image_size = 384

# %%
# Load AttGAN

attgan = AttGAN(gan_args)
attgan.load('./384_shortcut1_inject1_none_hq/weights.149.pth')
attgan.eval()

# %%
# CelebaHQ N samples

celeba_path = './celeba_hq_dataset/CelebA-HQ-img'
atts_path = './celeba_hq_dataset/CelebAMask-HQ-attribute-anno.txt'
base_attrs = gan_args.get('attrs')

sample_celeba_data = CelebA_HQ_custom(
                       data_path = celeba_path,
                       attr_path = atts_path,
                       selected_attrs = base_attrs,
                       image_size = training_set_image_size,
                       mode = 'train'
                     )

sample_celeba_dataloader = data.DataLoader(
                             sample_celeba_data, batch_size=training_set_size, num_workers=gan_args.get('num_workers'),
                             shuffle=True, drop_last=False
                          )

# %%
# Load valid CFs

folder = '1088'
front = 0
path = os.path.join('./Counterfactuals', folder, 'Front' + f'_{folder}_' + str(front) + '.pkl')

with open(path, 'rb') as f:
  pareto_front = pkl.load(f)
  factual_img = pkl.load(f)
  factual_atts = pkl.load(f)
  runtime_in_seconds = pkl.load(f)
  experiment_metadata = pkl.load(f)
  
raw_x_data, raw_y_data, raw_z_data, new_preds, new_attributes, generated_cfs, dominance_ranking, crowding_distances = unpack_front(pareto_front)

# %%
# Valid CFs

valid_cfs_idx = [i for i, y in enumerate(raw_y_data) if y < 0.5]
valid_cf_images = torch.stack([generated_cfs[i] for i in valid_cfs_idx]).squeeze(1)

count_valid_cfs = len([y for y in raw_y_data if y < 0.5])
valid_cf_atts = torch.stack([new_attributes[i] for i in valid_cfs_idx])

raw_x_data = [raw_x_data[i] for i in valid_cfs_idx]
raw_y_data = [raw_y_data[i] for i in valid_cfs_idx]
raw_z_data = [raw_z_data[i] for i in valid_cfs_idx]

# %%
# Define factual image 

factual_image = factual_img
factual_gender_prob, factual_age = mivolo_inference(factual_image, True)
if factual_gender_prob > 0.5:
  factual_gender_binary = 1
else:
  factual_gender_binary = 0

# %%
# Load batch of N random images

data_iterator = iter(sample_celeba_dataloader)
training_set_images, training_set_attributes, training_set_names = next(data_iterator)

# %%
# Create Metrics Dictionary

evaluation_metrics = {}

# %%
# Metrics

pop_size = len(generated_cfs)

###  SIZE #### 

size = count_valid_cfs / pop_size

evaluation_metrics["Size"] = {
  "size_pct":{"value":size, "definition":"Available CFs wrt required CFs"}, 
  "size_count":{"value":count_valid_cfs, "definition":"Available CFs"}, 
  "required_cfs":{"value":pop_size, "definition":"Required CFs"}
  }

### PROXIMITY ###

# Proximity - features

avg_distance = np.mean(raw_x_data) # code difference

# Proximity - images

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none') # LPIPS needs the images to be in the [-1, 1] range.
lpips_score = 0
for cf in valid_cf_images:
  lpips_score += lpips(factual_image, cf.unsqueeze(0))
lpips_score /= count_valid_cfs

# Minimality - average number of features changed

features_changed = torch.abs(valid_cf_atts - factual_atts)/factual_atts > 0.10
avg_features_changed = torch.sum(torch.sum(features_changed, axis = 1)) / (features_changed.shape[0] * features_changed.shape[1])

evaluation_metrics["Dissimilarity"] = {
  "proximity":{
    "avg_distance_features":{"value":round(avg_distance.item(), 3), "definition":"Average distance between the factual image features and valid CFs features (distance used for the code similarity objective)"}, 
    "avg_distance_images":{"value":round(lpips_score.item(), 3), "definition":"Average LPIPS score between the factual image and valid CFs"}
   },
  "minimality":{"avg_features_changed":{"value":round(avg_features_changed.item(), 3), "definition":"Average number of features changed for valid CFs"}}
  }

### DIVERSITY ###

# Diversity - average distance between attributes

distances = torch.sqrt(((valid_cf_atts - factual_atts) ** 2).sum(-1))
mean_distance = torch.mean(distances)

# Diversity - standard deviation for each attribute

attribute_deviations = torch.std(valid_cf_atts, dim = 0)
attribute_means = torch.mean(valid_cf_atts, dim = 0)
mean_attribute_value = torch.mean(attribute_means)
std_attribute_value = torch.mean(attribute_deviations)

# Diversity - LPIPS distance between images

average_lpips_distance = 0
for cf in valid_cf_images: # n
  average_lpips_distance += torch.sum(distances_between_image_sets(cf, 128, valid_cf_images, 'lpips')[0][1:]) # (n - 1 sumas)
average_lpips_distance /= (count_valid_cfs * (count_valid_cfs - 1))
  
# Diversity - Euclidean distance between images  
  
mean_cf_pixelwise_distance = 0
for cf in valid_cf_images: # n
  mean_cf_pixelwise_distance += torch.sum(distances_between_image_sets(cf, 128, valid_cf_images, 'pixel_wise')[0][1:]) # (n - 1 sumas)
mean_cf_pixelwise_distance /= (count_valid_cfs * (count_valid_cfs - 1))

evaluation_metrics["Diversity"] = {
    "avg_distance_features_L2":{"value":round(mean_distance.item(), 3), "definition":"Average distance between the factual image features and valid CFs features (L2 distance)"},
    "mean_cf_pixelwise_distance":{"value":round(mean_cf_pixelwise_distance.item(), 3), "definition":"Average pixel-wise distance of the CF set"},
    "mean_cf_lpips_distance":{"value":round(average_lpips_distance.item(), 3), "definition":"Average LPIPS distance of the CF set"},
    "mean_std":{"value":round(std_attribute_value.item(), 3), "definition":"Std deviation across all CF's attributes"},
    "attribute_means":{"value":attribute_means, "definition":"Average value of CF's attributes"},
    "attribute_changes":{"value":torch.sum(features_changed, axis = 0)/count_valid_cfs, "definition":"Number of CFs that had changed the attribute's value"},
    "attribute_deviations":{"value":attribute_deviations, "definition":"Standard deviation of CF's attributes"},
   }

# %%
# Discriminative Power - Distinguish between two different classes only using the CFs in C

# Test set - Gender Prediction

def gender_training_set(training_set = training_set_images, threshold = 0.5):
  
  training_set_gender = []

# Recorrer todo el training_set para encontrar las muestras más cercanas

  for sample in training_set: 
    gender_prob, factual_age = mivolo_inference(sample, True)
    if gender_prob >= 0.5:
      pred_gender = 1
    else:
      pred_gender = 0
    training_set_gender.append(pred_gender)
      
  return training_set_gender

def gender_test_set(training_images, training_labels, coverage = 0.30):
  
  # Crear las matrices de distancias
  
  factual_to_training_distances, factual_to_training_indices = distances_between_image_sets(factual_image, img_size = 128, training_set = training_images, distance_metric = 'pixel_wise')
  
  # Seleccionar k ejemplos de la clase positiva y de la negativa
  
  n = len(training_labels)
  n_female = np.sum(training_labels)
  n_male = n - n_female
  
  m_male = round(n_male * coverage)
  m_female = round(n_female * coverage)
  
  female_indices = torch.tensor([i for i, g in zip(factual_to_training_indices, training_labels) if training_labels[i] == 1][0:m_female])
  male_indices = torch.tensor([i for i, g in zip(factual_to_training_indices, training_labels) if training_labels[i] == 0][0:m_male])
    
  male_samples = torch.index_select(training_images, 0, torch.tensor(male_indices))
  female_samples = torch.index_select(training_images, 0, torch.tensor(female_indices))
  
  test_set_samples = torch.cat((female_samples, male_samples), dim = 0)
  test_set_gender = torch.cat((torch.ones(m_female, 1), torch.zeros(m_male, 1)))
  
  closest_data_idx_male = male_indices[0]
  closest_data_idx_female = female_indices[0]
  return test_set_samples, test_set_gender, closest_data_idx_male, closest_data_idx_female

# %%
# Create sets

training_set_gender = gender_training_set()
test_set_samples, test_set_gender, closest_data_idx_male, closest_data_idx_female = gender_test_set(training_set_images, training_set_gender)

# %%
# Training set for 1-NN Classifier

training_set_knn = torch.concat([valid_cf_images, factual_image])
training_set_gender_knn = gender_training_set(valid_cf_images)
training_set_gender_knn.append(factual_gender_prob)

# %%
def predict_gender(image, training_set_knn):
  distances, indices = distances_between_image_sets(image, img_size = 128, training_set = training_set_knn, distance_metric = 'pixel_wise')
  nearest_gender = training_set_gender_knn[indices[0]]
  if nearest_gender >= 0.5:
    p = 1
  else:
    p = 0
  return p

# %%
# Predict Gender on test set

test_set_predictions = []

for sample in test_set_samples:
  p = predict_gender(sample, training_set_knn)
  test_set_predictions.append(p)
  
test_set_predictions = torch.tensor(test_set_predictions).unsqueeze(1)

accuracy = torch.sum(test_set_predictions == test_set_gender) / test_set_gender.shape[0]
evaluation_metrics["Discriminative Power"] = {"1nn_accuracy":{"value":round(accuracy.item(), 3), "definition":"Accuracy of a 1-NN classifier trained on CFs + 1F, evaluated over k members of F and CF classes"}}

# %%
# Runtime

evaluation_metrics["Runtime"] = {"execution_time":{"value":round(runtime_in_seconds, 3), "definition":"Algorithm's Runtime"}}

# %%
# Implausibility - Average distance of the CF from the closest instance in the known set X

min_distances = torch.tensor(0)
for cf in valid_cf_images:
  distances, indices = distances_between_image_sets(cf, img_size = 128, training_set = training_set_images, distance_metric = 'lpips')
  min_d = distances[0]
  min_distances = min_distances + min_d
average_min_distance = min_distances / valid_cf_images.shape[0]

evaluation_metrics["Implausibility"] = {
  "average_min_distance_training":{"value":round(average_min_distance.item(), 3), "definition":"Average distance of the CFs from their closest instance in the training set"}
  }

# %%
# Define image for the instability metric

if factual_gender_binary == 1:
  sample_idx = closest_data_idx_female
else:
  sample_idx = closest_data_idx_male

instability_image = training_set_images[sample_idx].unsqueeze(0)
instability_image_atts = training_set_attributes[sample_idx]
instability_gender_prob, instability_age = mivolo_inference(instability_image, True)
if instability_gender_prob > 0.5:
  instability_gender_binary = 1
else:
  instability_gender_binary = 0

# %%
# Instability

# Generate CFs for the closest sample with the same classification as the factual image

desired_pred = 1 - instability_gender_binary
max_evals = 500
pop_size = 100

problem = AttGanPlausibleCounterfactualProblem(
            image = instability_image, 
            code = factual_atts, # use the predicted scores for each attribute
            decoder = attgan.G, 
            discriminator = attgan.D, 
            classifier = mivolo_inference, 
            original_pred = instability_gender_prob,
            original_discriminator_score = None,
            desired_pred = desired_pred,
            use_lpips = True
          )

algorithm = NSGAII(
             problem=problem,
             population_size=pop_size,
             offspring_population_size=pop_size,
             mutation=PolynomialMutation(
                 probability=1/problem._number_of_variables,
                 distribution_index=20),
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
    new_pred = sol.prediction
    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
      if y < 0.5: # CF validos
        instability_attributes_set.append(new_code)
      
instability_attributes_set = torch.stack(instability_attributes_set)

d, _ = distances_between_image_sets(instability_image, img_size = 128, training_set = factual_image, distance_metric = 'pixel_wise')
instability_score = 0
for cf in valid_cf_atts:
  distance_to_cf = torch.sqrt(((cf - instability_attributes_set) ** 2).sum(-1))
  instability_score += distance_to_cf
instability_score = torch.sum(instability_score) / (instability_attributes_set.shape[0] * valid_cf_atts.shape[0]) * 1 / (1 + d)

evaluation_metrics["Instability"] = {"instability":{"value":round(instability_score.item(), 3), "definition":"Obtain CFs for the closest sample with the same classification as X. Measure the distance between these sets"}}

# %%
# Compute FID score between CFs and training samples

from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(normalize = True)
fid.update(normalize_0_1(training_set_images), real=True)
fid.update(normalize_0_1(valid_cf_images), real=False)
fid_score = fid.compute()

evaluation_metrics["FID_score"] = {"fid":{"value":round(fid_score.item(), 3), "definition":"FID score between valid CFs and training set"}}

# %%
# Hypervolume

hv = hypervolume_indicator(pareto_front)

evaluation_metrics["Hypervolume"] = {
  "hypervolume":{"value":round(hv.item(), 3), "definition":"Hypervolume indicator of the estimated Pareto front wrt the point (1, 1, 1)"}    
}

# %%
# Grabar las metricas en excel 

metrics_df = pd.DataFrame([], columns=['Metric_Class', 'Metric_SubClass', 'Metric_Name', 'Value', 'Definition'])
for key in evaluation_metrics.keys():
    for sub_key in evaluation_metrics[key]:
        if 'value' in evaluation_metrics[key][sub_key]:
            value = evaluation_metrics[key][sub_key]["value"]
            definition = evaluation_metrics[key][sub_key]["definition"]
            name = sub_key
            metric_class = key
            metric_subclass = None
            row = pd.DataFrame([[metric_class, metric_subclass, name, value, definition]], columns=['Metric_Class', 'Metric_SubClass', 'Metric_Name', 'Value', 'Definition'])
            metrics_df = pd.concat((metrics_df, row))
        else:
            for subkey2 in evaluation_metrics[key][sub_key]:
                if 'value' in evaluation_metrics[key][sub_key][subkey2]:
                    value = evaluation_metrics[key][sub_key][subkey2]["value"]
                    definition = evaluation_metrics[key][sub_key][subkey2]["definition"]
                    name = subkey2
                    metric_class = key
                    metric_subclass = sub_key
                    row = pd.DataFrame([[metric_class, metric_subclass, name, value, definition]], columns=['Metric_Class', 'Metric_SubClass', 'Metric_Name', 'Value', 'Definition'])
                    metrics_df = pd.concat((metrics_df, row))

for key in experiment_metadata:
    metric_class = 'Metadata'
    metric_subclass = 'Hyperparameter'
    name = key
    value = experiment_metadata[key]
    definition = None
    row = pd.DataFrame([[metric_class, metric_subclass, name, value, definition]], columns=['Metric_Class', 'Metric_SubClass', 'Metric_Name', 'Value', 'Definition'])
    metrics_df = pd.concat((metrics_df, row))
    
metrics_df.to_excel(os.path.join('./Counterfactuals', folder, folder + '_' + str(front) + '.xlsx'), index = False)


