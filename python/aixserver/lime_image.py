from aix360.algorithms.lime import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from aixserver.explainer import Explainer
from typing import Callable, Dict
import numpy as np
import logging
class LimeImage(Explainer):
    def __init__(self,_predict_fn:Callable):
        self._predict = _predict_fn
    def explain(self, request:Dict, args:Dict,):
        instances = request["instances"]
        try:
            top_labels = (int(request["top_labels"])
                          if "top_labels" in request else
                          args["top_labels"])
            segmentation_alg = (request["segmentation_alg"]
                                if "segmentation_alg" in request else
                                args["segmentation_alg"])
            num_samples = (int(request["num_samples"])
                           if "num_samples" in request else
                           args["num_samples"])
            positive_only = ((request["positive_only"].lower() == "true") | (request["positive_only"].lower() == "t")
                             if "positive_only" in request else
                             args["positive_only"])
            min_weight = (float(request['min_weight'])
                          if "min_weight" in request else
                          args['min_weight'])
        except Exception as err:
            raise Exception("Failed to specify parameters: %s", (err,))
        
        inputs = np.array(instances[0])
        logging.info("Calling explain on image of shape %s", (inputs.shape,))
        try:
            explainer = LimeImageExplainer(verbose=False)
            segmenter = SegmentationAlgorithm(segmentation_alg, kernel_size=1,
                                                  max_dist=200, ratio=0.2)
            explanation = explainer.explain_instance(inputs,
                                                         classifier_fn=self._predict,
                                                         top_labels=top_labels,
                                                         hide_color=0,
                                                         num_samples=num_samples,
                                                         segmentation_fn=segmenter)
            temp = []
            masks = []
            for i in range(0, top_labels):
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[i],
                                                            positive_only=positive_only,
                                                            num_features=10,
                                                            hide_rest=False,
                                                            min_weight=min_weight)
                masks.append(mask.tolist())

            return {"explanations": {
                "temp": temp.tolist(),
                "masks": masks,
                "top_labels": np.array(explanation.top_labels).astype(np.int32).tolist()
            }}
        except Exception as err:
            raise Exception("Failed to explain %s" % err)