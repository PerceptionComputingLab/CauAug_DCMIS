# ------------------------------------------------------------------------------
# Prostate segmentation task from the Medical Segmentation Decathlon
# (http://medicaldecathlon.com/)
# ------------------------------------------------------------------------------

import os
import numpy as np
import SimpleITK as sitk
from mp.utils.load_restore import join_path
from mp.data.datasets.dataset_segmentation import (
    SegmentationDataset,
    SegmentationInstance,
)
from mp.paths import storage_data_path
import mp.data.datasets.dataset_utils as du


class Optic(SegmentationDataset):
    r"""Class for the optic segmentation."""

    def __init__(self, subset="", hold_out_ixs=None, target=-1):
        if hold_out_ixs is None:
            hold_out_ixs = []

        global_name = subset
        dataset_path = os.path.join(storage_data_path, global_name)

        # Fetch all patient/study names
        study_names = set(
            file_name.split(".png")[0].split("_gt")[0]
            for file_name in os.listdir(dataset_path)
        )

        # Build instances
        instances = []
        for study_name in study_names:
            instances.append(
                SegmentationInstance(
                    x_path=os.path.join(dataset_path, study_name + ".png"),
                    y_path=os.path.join(dataset_path, study_name + "_gt.png"),
                    name=study_name,
                    group_id=None,
                    target_class=target,
                )
            )

        label_names = ["background", "optic"]
        super().__init__(
            instances,
            name=global_name,
            label_names=label_names,
            modality="MR",
            nr_channels=1,
            hold_out_ixs=[],
        )
