import torch
import cv2 
import numpy as np
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import pandas as pd
from jmetal.core.quality_indicator import HyperVolume
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms as transforms
import os
import inspect
from datetime import datetime
from jmetal.util.termination_criterion import TerminationCriterion

# Helper functions (from COUNTGEN repo, adapted by me) 
# Map images from (-1, 1) to (0, 1)

def normalize_0_1(_img):
  img = (_img - torch.min(_img))/(torch.max(_img) - torch.min(_img))
  return img

def denormalize(img):
  return img + 1

def prepare_img_for_mivolo(img):
  img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
  img = (img * 255).astype(np.uint8)
  cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  return cv2_img

def unpack_front(front):
    raw_x_data = []
    raw_y_data = []
    new_preds = []
    new_attributes = []
    raw_z_data = []
    image_data = []
    dominance_ranking = []
    crowding_distances = []

    for sol in front:
        x = float(sol.objectives[0])
        y = float(sol.objectives[1])
        z = float(sol.objectives[2])
        new_code = torch.tensor(sol.variables)
        new_pred = sol.prediction
        ranking = sol.attributes.get('dominance_ranking')
        crowding_distance = sol.attributes.get('crowding_distance')
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
          raw_x_data.append(x)
          raw_y_data.append(y)
          raw_z_data.append(z)
          dominance_ranking.append(ranking)
          crowding_distances.append(crowding_distance)
          new_preds.append(new_pred)
          new_attributes.append(new_code)
          image_data.append(sol.counterfactualImage)
      
    return raw_x_data, raw_y_data, raw_z_data, new_preds, new_attributes, image_data, dominance_ranking, crowding_distances

def min_max(_x, _min=None, _max=None):
    if _min is None and _max is None:
        return (_x - np.min(_x)) / (np.max(_x) - np.min(_x))
    else:
        return (_x - _min) / (_max - _min)

def plot_front(valid_cf_n, _ax, _c_ax, _x, _y, _z):
    _ax.set_xlabel('Code Sim')
    _ax.set_ylabel('Adv attack')
    _ax.set_zlabel('Plausibility')
    _cs = _ax.scatter3D(_x, _y, _z, c=(_x + _y + _z))
    plt.colorbar(_cs,
                 orientation="horizontal", pad=0.01, ax=_c_ax)
    return _cs.cmap

def plot_data(valid_cf_n, _ax, _l_ax, _x, _y, _z, _cmap):
    _ax.set_ylim((0, np.max(_x + _y + _x) * 1.1))
    _ax.scatter(range(valid_cf_n), _x, s=1, c='red',
                alpha=0.3)
    l1, = _ax.plot(_x, c='red', label='Code Similarity (X)', alpha=0.3)

    _ax.scatter(range(valid_cf_n), _y, s=1, c='green',

                alpha=0.3)
    l2, = _ax.plot(_y, c='green', label='Adv Attack (Y)', alpha=0.3)

    _ax.scatter(range(valid_cf_n), _z, s=1, c='blue',

                alpha=0.3)
    l3, = _ax.plot(_z, c='blue', alpha=0.3, label='Plausibility (Z)')

    _ax.scatter(range(valid_cf_n), _x + _y + _z,
                s=2,
                marker='x',
                c=(_x + _y + _z),
                cmap=_cmap)
    l4, = _ax.plot(_x + _y + _z,
                   c='black',
                   label='X + Y + Z',
                   alpha=0.3)
    _l_ax.legend(handles=[l1, l2, l3, l4],
                 fontsize='xx-small')
    
def plot_parallel_coordinates(axis, x_data, y_data, z_data, cmap, points = None):

  df = pd.DataFrame({
         'Code Similarity': x_data,
         'Adversarial Attack': y_data,
         'Plausibility': z_data
      })

  df['label'] = 'Multi-objectives front'
  
  # Parallel coordinates plot
  parallel_coordinates(df[['Code Similarity','Adversarial Attack','Plausibility','label']], 'label', color=("#dfd9d9"), ax = axis)
  axis.set_xlabel('Objectives')
  axis.set_ylabel('Objective Values')
  
  if points is not None and len(points) > 0:
    objectives = ['Code Similarity', 'Adversarial Attack', 'Plausibility']

    for i, idx in enumerate(points):
      vals = [x_data[idx], y_data[idx], z_data[idx]]
      axis.plot(
        objectives,
        vals,
        color=cmap(i),
        linewidth=3,
        marker='o',
    )

    axis.set_xlabel('Objectives')
    axis.set_ylabel('Objective Values')
    axis.get_legend().remove()   
    
def plot_cf_images(fig, grid, n_row, n_col):
  counterfactuals_ax = []
  for col in range(n_col):
    for row in range(n_row):
      ax = fig.add_subplot(grid[row, col])
      ax.set_xticks([])
      ax.set_yticks([])
      counterfactuals_ax.append(ax)
  return counterfactuals_ax

def get_counterfactuals(valid_cf_n, n_row, n_col):
  count_idx = np.arange(valid_cf_n)
  np.random.shuffle(count_idx)
  n_samples = n_row * n_col
  count_idx = count_idx[:n_samples]
  return count_idx
    
def plot_fig_from_paper(x_data, y_data, z_data, image_data, valid_cf_n, n_row = 2, n_col = 5, figsize = [12.8, 9.6], titles_font_size = 8):

    fig = plt.figure(figsize = figsize, constrained_layout=True)
    fig.suptitle("NSGAII results (paper's repo)", fontsize=16)
    
    spec2 = fig.add_gridspec(5, 17) # 
    cf_subplot_spec = spec2[2:3, 8:17]
    front_ax = fig.add_subplot(spec2[0:5, 0:7], projection='3d')
    data_ax = fig.add_subplot(spec2[4, 0:8])
    parallel_coords = fig.add_subplot(spec2[3:5, 8:17])
    legend_ax = fig.add_subplot(spec2[4, 8])
    legend_ax.axis('off')
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    counter_spec = gridspec.GridSpecFromSubplotSpec(n_row, n_col, subplot_spec = cf_subplot_spec)
    
            
    front_title = front_ax.set_title('Pareto Front', fontsize = titles_font_size)
    data_ax.set_title('Multi-objectives: Code Similarity f1(+), Adversarial Power f2(+), Plausibility f3(+)', fontsize = titles_font_size)
    parallel_coords.set_title('Trade-off analysis', fontsize = titles_font_size)
    
    front_title_pos = front_title.get_position()
    front_title_xy_coord = front_ax.transAxes.transform(front_title_pos)
    cf_bbx_pos = cf_subplot_spec.get_position(fig)
    x0, y0, x1, y1, w, h = cf_bbx_pos.x0, cf_bbx_pos.y0, cf_bbx_pos.x1, cf_bbx_pos.y1, cf_bbx_pos.width, cf_bbx_pos.height

    fig.text(
      x0 + w/2, y1 + h,
      "Sample Counterfactuals",
      ha='center', va='top',
      fontsize= 8
    )
  
    # PLOT front
    custom_cmap = plot_front(valid_cf_n, front_ax, data_ax, x_data, y_data, z_data)

    # Plot data
    plot_data(valid_cf_n, data_ax, legend_ax, x_data, y_data, z_data, custom_cmap)

    # PLOT Counterfactuals - Sample Random idxs
    highlight_cmap = cm.get_cmap('Set3')
    
    count_idx = get_counterfactuals(valid_cf_n, n_col, n_row)
    counterfactuals_ax = plot_cf_images(fig, counter_spec, n_row, n_col)
    
    for i, idx in enumerate(count_idx):
        value = x_data[idx] + y_data[idx] + z_data[idx]
        
        counterfactuals_ax[i].imshow(min_max(image_data[idx].squeeze(0).permute(1, 2, 0).cpu().numpy()))
        counterfactuals_ax[i].set_title(f"""[{round(x_data[idx], 3)}, {round(y_data[idx], 3)}, {round(z_data[idx], 3)}]""", loc = 'center', y = -0.5, fontsize = 8)

        plt.setp(counterfactuals_ax[i].spines.values(),
                color=highlight_cmap(i),
                linewidth=4)
        data_ax.axvline(x=idx, linestyle='-.', alpha=0.1)
        data_ax.scatter(idx, value,
                        s=30,
                        marker='X',
                        c=highlight_cmap(i))
        front_ax.scatter3D(x_data[idx], y_data[idx], z_data[idx],
                          c=highlight_cmap(i),
                          s=valid_cf_n,
                          marker='X')
        
    # Plot Pareto Front
    
    plot_parallel_coordinates(parallel_coords, x_data, y_data, z_data, highlight_cmap, count_idx)
        
    plt.show()
    
  # HyperVolume Indicator 

def hypervolume_indicator(pareto_front):
  objectives_list = []
  for sol in pareto_front:
    if sol.counterfactualAttributes != {}:
      objectives_list.append(sol.objectives)
  objectives_array = np.array(objectives_list)
  objectives_array = (objectives_array - np.min(objectives_array, axis = 0)) / (np.max(objectives_array, axis = 0) - np.min(objectives_array, axis = 0))

  hypervolume = HyperVolume(reference_point = [1, 1, 1]) # area de soluciones menores a esta referencia. por lo cual representa el mejor caso

  h = hypervolume.compute(objectives_array)
  
  return h

# Funcion for computing distances

def distances_between_image_sets(
                              tensor_img, 
                              img_size = None, 
                              training_set = None,
                              distance_metric = 'pixel_wise'
                            ):
  
    batch_size = training_set.shape[0]
  
    if tensor_img.shape[2] != img_size:
      tensor_img = transforms.functional.resize(tensor_img,[img_size, img_size]) # (B, C, H, W)
    tensor_img = normalize_0_1(tensor_img)
    
    if training_set.shape[2] != img_size:
      training_set = transforms.functional.resize(training_set,[img_size, img_size])
    training_set = normalize_0_1(training_set)
    
    if distance_metric == 'pixel_wise':
      d = torch.sqrt(torch.mean((tensor_img - training_set) ** 2, dim=[1,2,3]))
    if distance_metric == 'lpips':
      lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none', normalize = True)
      d = lpips(tensor_img.expand(batch_size, 3, img_size, img_size), training_set)
    if distance_metric == 'ssim':
      ssim = StructuralSimilarityIndexMeasure()
      d = []
      for s in training_set:
        sim = ssim(tensor_img, s.unsqueeze(0))
        d.append(torch.abs(sim))
      d = torch.stack(d)
      
    values, indices = torch.sort(d)  
    return values, indices
  
def generate_text_explanations(base_atts, f_pred, cf_pred, class_name, att_f, att_cf, top_att = 5, decimals = 2):
  sorted, indices = torch.sort(torch.abs(att_f - att_cf), descending=True)
  idxs = indices[:, :top_att].squeeze(0)
  idxs = [i.item() for i in idxs]
  top_n_cf_atts_vals = att_cf[:, indices[:,:top_att]].reshape(-1) 
  top_n_cf_atts_rescaled = torch.round((top_n_cf_atts_vals + 1)/2, decimals = decimals) * 100
  top_n_f_atts_vals = att_f[:, indices[:,:top_att]].reshape(-1)
  top_n_f_atts_rescaled = torch.round((top_n_f_atts_vals + 1)/2, decimals = decimals) * 100
  top_n_atts_names = [base_atts[i] for i in idxs]
  
  explanation = "\n".join(f"You are predicted to have {f_value:.{decimals}f}% of the {f_name} attribute, whereas your Counterfactual has {cf_value:.{decimals}f}%" for f_value, f_name, cf_value in zip(top_n_f_atts_rescaled, top_n_atts_names, top_n_cf_atts_rescaled))
  complete_explanation = f"You were originally predicted to be {f_pred * 100:.5f}% {class_name}, but now you are predicted to be {cf_pred * 100:.5f}% {class_name} due to the following changes: \n" + explanation
  print(complete_explanation)
  
def file_name_creator(img_file_name):
  existing_files = [f.replace('.pkl', '') for f in os.listdir('./Counterfactuals')]
  base_file_name = 'Front_' + img_file_name.replace('.jpg', '')
  i = 0
  found_file = True
  while found_file:
    file_name = base_file_name + '_' + str(i)
    found_file = file_name in existing_files
    i += 1
  return file_name

def write_pkl_file(img_file_name, algorithm, problem, factual_image, factual_atts, pareto_front, runtime_in_seconds, prediction_orig, desired_pred, pop_size, max_evals, evals):
  file_name = file_name_creator(img_file_name)
  experiment_metadata = {
    "desired_pred":desired_pred, 
    "original_pred":prediction_orig,
    "pop_size":pop_size, 
    "max_evals":max_evals,
    "elapsed_evals":evals,
    "algorithm_mutator":str(algorithm.mutation_operator),
    "algorithm_crossover":str(algorithm.crossover_operator),
    "algorithm_termination":str(algorithm.termination_criterion),
    "problem_recipe":inspect.getsource(problem.evaluate),
    "experiment_timestamp": datetime.now()
    }
  with open('./Counterfactuals/' + file_name + '.pkl', 'wb') as f:
    pkl.dump(pareto_front, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(factual_image, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(factual_atts, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(runtime_in_seconds, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(experiment_metadata, f, pkl.HIGHEST_PROTOCOL)
    
    
from typing import List, TypeVar
import logging
from jmetal.core.observer import Observer

S = TypeVar("S")

LOGGER = logging.getLogger("jmetal")
      
class HyperVolumeObserver(Observer):
    def __init__(self, frequency: int = 1, img_file_name:str = None) -> None:
        """Show the number of evaluations, the best HIV and the computing time.
        :param frequency: Display frequency."""

        self.display_frequency = frequency
        self.hv_history = []
        self.eval_history = []
        self.plot_path = './Counterfactuals/' + file_name_creator(img_file_name) + '.png'
        
    def update(self, *args, **kwargs):
        computing_time = kwargs["COMPUTING_TIME"]
        evaluations = kwargs["EVALUATIONS"] # counter
        solutions = kwargs["SOLUTIONS"] 
        
        if (evaluations % self.display_frequency) == 0 and solutions:
            
            hv = hypervolume_indicator(solutions)
            
            LOGGER.info(
                "Evaluations: {} \n HyperVolume: {} \n Computing time: {}".format(evaluations, hv, computing_time)
            )
            
            self.hv_history.append(hv)
            self.eval_history.append(evaluations)
            self._save_plot()
            
    def _save_plot(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.eval_history, self.hv_history, marker="o")
        plt.xlabel("Evaluations")
        plt.ylabel("HyperVolume")
        plt.title("HV Curve Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

class StoppingByHV(TerminationCriterion):
    def __init__(self, expected_value: float, max_steps: int, max_evaluations: int):
        super(StoppingByHV, self).__init__()
        self.expected_value = expected_value
        self.max_evaluations = max_evaluations
        self.max_steps = max_steps
        self.hv_history = [1]
        self.steps = 0
        
    def update(self, *args, **kwargs):
        solutions = kwargs["SOLUTIONS"]
        self.evaluations = kwargs["EVALUATIONS"]

        if solutions:
            hv = hypervolume_indicator(solutions)
            self.hv_history.append(hv)

    @property
    def is_met(self):
       abs_epsilon = np.abs(self.hv_history[-1]/self.hv_history[-2] - 1)
       if abs_epsilon < self.expected_value:
         self.steps += 1
       else:
         self.steps = 0
       if self.evaluations > self.max_evaluations or self.steps > self.max_steps:
         met = True
         return met