## Matching

import cv2
import numpy as np
import iris
from iris.callbacks.pipeline_trace import Optional
from iris.io.validators import List
from iris.nodes.iris_response.probe_schemas.regular_probe_schema import confloat

# 1. Create IRISPipeline object
iris_pipeline = iris.IRISPipeline()

matcher = iris.HammingDistanceMatcher()

def __init__(
        self,
        rotation_shift: int = 15,
        nm_dist: Optional[confloat(ge=0, le=1, strict=True)] = None,
        weights: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Assign parameters.

        Args:
            rotation_shift (int): rotations allowed in matching, converted to shifts in columns. Defaults to 15.
            nm_dist (Optional[confloat(ge=0, le = 1, strict=True)]): nonmatch distance used for normalized HD. Optional paremeter for normalized HD. Defaults to None.
            weights (Optional[List[np.ndarray]]): list of weights table. Optional paremeter for weighted HD. Defaults to None.
        """


def match(probe, gallery):

    if probe == True and gallery == True:
        probe_imgpixel = cv2.imread(probe, cv2.IMREAD_GRAYSCALE)
        gallery_imgpixel = cv2.imread(gallery, cv2.IMREAD_GRAYSCALE)

        probe_output = iris_pipeline(probe_imgpixel, eye_side="left")
        probe_code = probe_output['iris_template']

        gallery_output = iris_pipeline(gallery_imgpixel, eye_side="left")
        gallery_code = gallery_output['iris_template']

        same_subjects_distance = matcher.run(probe_code, gallery_code)

        #print(same_subjects_distance)

    return same_subjects_distance