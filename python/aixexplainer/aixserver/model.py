# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict
from enum import Enum
import asyncio
import logging
import kserve
import numpy as np

from aix360.algorithms.lime import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from aix360.algorithms.lime import LimeTextExplainer
from aixserver.lime_image import LimeImage


class AIXModel(kserve.Model):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, predictor_host: str, segm_alg: str, num_samples: str,
                 top_labels: str, min_weight: str, positive_only: str, explainer_type: str):
        super().__init__(name)
        # Kserve model args
        self.name = name
        self.ready = False
        self.predictor_host = predictor_host
        self.positive_only = (positive_only.lower() == "true") | (positive_only.lower() == "t")
        # ML model and explainer args 
        self.top_labels = int(top_labels)
        self.num_samples = int(num_samples)
        self.segmentation_alg = segm_alg
        self.min_weight = float(min_weight)
        self.explainer_type = str.lower(explainer_type)
        #put above script before determining if type is valid and make it to lower case 
        self.ml_args = {"top_labels" : self.top_labels,
                        "segmentation_alg" : self.segmentation_alg,
                        "num_samples" : self.num_samples,
                        "positive_only" : self.positive_only,
                        "min_weight" : self.min_weight}
        
        #verifing input type
        if self.explainer_type == "limeimages":
            self.explainer = LimeImage(self._predict)
        #elif self.explainer_type == "limetexts":
           #self.explainer = LimeText()
        else:
            raise Exception("Invalid explainer type: %s" % explainer_type)
        
    def load(self) -> bool:
        self.ready = True
        return self.ready

    def _predict(self, input_im):
        scoring_data = {'instances': input_im.tolist()if type(
            input_im) != list else input_im}

        loop = asyncio.get_running_loop()
        resp = loop.run_until_complete(self.predict(scoring_data))
        return np.array(resp["predictions"])

    def explain(self, request: Dict) -> Dict:
        try:
            explaination = self.explainer.explain(request,self.ml_args)
            return explaination
        except Exception as err:
            raise Exception("Failed to explain %s" % err)
