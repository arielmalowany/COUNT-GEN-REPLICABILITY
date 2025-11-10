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
from cf_utils import normalize_0_1

class AttGanPlausibleCounterfactualProblem(FloatProblem):
    """
    Problem of finding counterfactuals for the AttGAN model -- Pytorch
    """

    def __init__(self, number_of_variables: int = 13, image=None,
                 code=None,  
                 decoder=None,
                 discriminator=None,
                 classifier=None,
                 original_pred=None,
                 desired_pred = None,
                 use_mivolo = True,
                 threshold_int = 0.5):
        """ :param number_of_variables: number of decision variables of the problem.
        """
        self._number_of_variables = number_of_variables
        self._number_of_objectives = 3
        self._number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self._number_of_objectives
        self.obj_labels = ['$ f_{} $'.format(i) for i in range(self._number_of_objectives)]

        self.lower_bound = number_of_variables * [-1.0]
        self.upper_bound = number_of_variables * [1.0]

        self.image = image.detach()
        self.code = code.detach()
        self.decoder = decoder
        self.original_pred = original_pred
        self.desired_pred = torch.tensor(desired_pred)
        original_pred_decision = 0
        if original_pred >= 0.5:
            original_pred_decision = 1
        self.original_pred_decision = original_pred_decision
        self.discriminator = discriminator
        self.classifier = classifier
        self.counterfactuals = pd.DataFrame(data=[], columns=[
            'CounterfactualImage', 'ClassifierProbability',
            'DiscriminatorDistance', 'OriginalCode', 'CounterfactualCode'])
        self.use_mivolo = use_mivolo
        
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
        df = self.counterfactuals
        real_image = self.image
        code = self.code
        new_code = torch.tensor(solution.variables).unsqueeze(0)
        
        # Real Image, new code
        new_image = self.decoder(real_image, new_code)
        new_image_n = normalize_0_1(new_image)
        img = new_image_n.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        if self.use_mivolo:
          img = (img * 255).astype(np.uint8)
          cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          new_pred, _ = mivolo_inference(cv2_img, True)
          if new_pred is None:
             new_pred = np.inf
        else:
          new_pred = self.classifier(new_image_n).item()
        
        solution.prediction = new_pred
        
        # Pass the real image through the discriminator
        d_real, _ = self.discriminator(real_image)
        d_real = d_real[0].float().item() # Discriminator score
        
        # Pass the fake image through the discriminator
        d_fake, _ = self.discriminator(new_image)
        d_fake = d_fake[0].float().item() # Discriminator score

        pred_decision = 0
        if new_pred >= 0.5:
          pred_decision = 1

        code_difference = torch.sum((new_code - code) ** 2).float().item()
        #pred_difference = (self.original_pred - new_pred).squeeze().float().item() # Maximize
        disc_difference = (d_real - d_fake) # > 0 better

        solution.objectives[0] = code_difference #torch.sum(torch.abs(new_code - code)).float().item()
        solution.objectives[1] = torch.abs(self.desired_pred - new_pred) # minimize the new score for the base class
        solution.objectives[2] = np.abs(disc_difference) # maximize the gan score for the fake image

        solution.counterfactualImage = new_image # Tensor directo del decoder

        #if (pred_decision != self.original_pred_decision) and (disc_difference < 0):
        #    f, ax = plt.subplots(1, 2)
        #    real_image_to_plot = normalize(real_image).squeeze(0).permute(1, 2, 0).numpy()
        #     new_image_to_plot = normalize(new_image).squeeze(0).permute(1, 2, 0).numpy()
        #     ax[0].imshow(real_image_to_plot)
        #     ax[1].imshow(new_image_to_plot)
            #files = os.listdir('./Counterfactuals/images/')
            #plt.savefig('./Counterfactuals/images/Count' + str(len(files)) + '.png')
        #   plt.close(f)
        return solution