# Generator Class

import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torchsummary import summary
import json
import torch.utils.data as data
import h5py
import pickle as pkl
import torchvision.transforms as transforms
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
import pandas as pd
import numpy as np
import cv2
from my_mivolo_inference import mivolo_inference
from cf_utils import *
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class AttGanPlausibleCounterfactualProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model -- Pytorch
    """

    def __init__(self, 
                 number_of_variables: int = 13, 
                 base_atts = None,
                 image=None,
                 code=None,  
                 decoder=None,
                 discriminator=None,
                 classifier=None,
                 original_pred=None,
                 desired_pred = None,
                 original_discriminator_score = None,
                 threshold_int = 0.5,
                 use_lpips = False,
                 non_actionable_features = None
                 ):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 3
        self._number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self._number_of_objectives)]

        self.lower_bound = number_of_variables * [-1.0]
        self.upper_bound = number_of_variables * [1.0]
        
        if non_actionable_features is not None:
          for att in non_actionable_features:
            idx = base_atts.index(att)
            self.lower_bound[idx] = 0
            self.upper_bound[idx] = 0
        
        """
          Load generator, discriminator, classifier
        """
        
        self.decoder = decoder
        self.discriminator = discriminator
        self.classifier = classifier
        self.use_lpips = use_lpips
        
        """
          Load input data
        """

        self.factual_image = image.detach()
        self.factual_code = code.detach()
        self.factual_disc_score = original_discriminator_score.detach()
        self.factual_pred = original_pred
        self.desired_pred = torch.tensor(desired_pred)
        
        
    def name(self):
      return "AttGanPlausibleCounterfactualProblem"

    def number_of_objectives(self):
      return self._number_of_objectives
      
    def number_of_constraints(self):
      return self._number_of_constraints
    
    def number_of_variables(self):
      return self._number_of_variables

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
      with torch.no_grad():
        real_image = self.factual_image
        factual_code = self.factual_code
        new_code = torch.tensor(solution.variables).unsqueeze(0)
        d_real = self.factual_disc_score
        use_lpips = self.use_lpips
        factual_image = self.factual_image
        
        # Real Image, new code
        
        new_image = self.decoder(real_image, new_code)
        new_pred = self.classifier(new_image)
        
        solution.prediction = new_pred
        
        if not use_lpips:
          # Pass the fake image through the discriminator
          d_fake, _ = self.discriminator(new_image)
          d_fake = d_fake[0].float().item() # Discriminator score
        
        else:
          lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none')
          d_fake = lpips(factual_image, new_image).item()
          
        # Begin Objectives
 
        code_difference = torch.sqrt(torch.sum((new_code - factual_code) ** 2))
        pred_difference = torch.abs(self.desired_pred - new_pred)
        disc_difference = (d_real - d_fake) # > 0 better

        solution.objectives[0] = code_difference.float().item() #torch.sum(torch.abs(new_code - code)).float().item()
        solution.objectives[1] = pred_difference.float().item() # minimize the new score for the base class
        solution.objectives[2] = np.abs(disc_difference).item() # maximize the gan score for the fake image
        
        # End Objectives

        solution.counterfactualImage = new_image # Tensor directo del decoder
        
        return solution