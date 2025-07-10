import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import mp.data.datasets.dataset_utils as du
from mp.utils.load_restore import join_path

import re


def dryad_extract_images(source_path, target_path, merge_labels, subset):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    def bbox_3D(img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    # Create directories
    if not os.path.exists(target_path):
        os.makedirs(os.path.join(target_path))

    # Patient folders s01, s02, ...
    for patient_folder in filter(
        lambda s: re.match(r"^s[0-9]+.*", s), os.listdir(source_path)
    ):

        # Loading the image
        image_path = os.path.join(
            source_path,
            patient_folder,
            f"{patient_folder}_{subset['Modality'].lower()}_"
            f"{subset['Resolution'].lower()}_defaced_MNI.nii.gz",
        )
        x = sitk.ReadImage(image_path)
        x = sitk.GetArrayFromImage(x)

        # For each MRI, there are 2 segmentation (left and right hippocampus)
        for side in ["L", "R"]:
            # Loading the label
            label_path = os.path.join(
                source_path,
                patient_folder,
                f"{patient_folder}_hippolabels_"
                f"{'hres' if subset['Resolution'] == 'Hires' else 't1w_standard'}"
                f"_{side}_MNI.nii.gz",
            )

            y = sitk.ReadImage(label_path)
            y = sitk.GetArrayFromImage(y)

            # We need to recover the study name of the image name to construct the name of the segmentation files
            study_name = f"{patient_folder}_{side}"

            # Average label shape (T1w, standard): (37.0, 36.3, 26.7)
            # Average label shape (T1w, hires): (94.1, 92.1, 68.5)
            # Average label shape (T2w, hires): (94.1, 92.1, 68.5)
            assert x.shape == y.shape

            # Disclaimer: next part is ugly and not many checks are made

            # So we first compute the bounding box
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(y)

            # Compute the start idx for each dim
            dr = (rmax - rmin) // 4
            dc = (cmax - cmin) // 4
            dz = (zmax - zmin) // 4

            # Reshaping
            y = y[rmin - dr : rmax + dr, cmin - dc : cmax + dc, zmin - dz : zmax + dz]

            if merge_labels:
                y[y > 1] = 1

            x_cropped = x[
                rmin - dr : rmax + dr, cmin - dc : cmax + dc, zmin - dz : zmax + dz
            ]

            x_cropped = (x_cropped - np.min(x_cropped)) / (
                np.max(x_cropped) - np.min(x_cropped)
            )
            x_cropped = np.clip(x_cropped, 0.001, 0.99)

            # Save new images so they can be loaded directly
            sitk.WriteImage(
                sitk.GetImageFromArray(y),
                join_path([target_path, study_name + "_gt.nii.gz"]),
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(x_cropped),
                join_path([target_path, study_name + ".nii.gz"]),
            )


def decathlon_extract_images(source_path, target_path, merge_labels):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    images_path = os.path.join(source_path, "imagesTr")
    labels_path = os.path.join(source_path, "labelsTr")

    # Filenames have the form 'hippocampus_XX.nii.gz'
    filenames = [x for x in os.listdir(images_path) if x[:5] == "hippo"]

    # Create directories
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for filename in filenames:

        # Extract only T2-weighted
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = sitk.GetArrayFromImage(x)
        y = sitk.ReadImage(os.path.join(labels_path, filename))
        y = sitk.GetArrayFromImage(y)

        # Shape expected: (35, 51, 35)
        # Average label shape: (24.5, 37.8, 21.0)
        assert x.shape == y.shape

        # No longer distinguish between hippocampus proper and subiculum
        if merge_labels:
            y[y == 2] = 1

        # Save new images so they can be loaded directly
        study_name = filename.replace("_", "").split(".nii")[0]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = np.clip(x, 0.001, 0.99)
        sitk.WriteImage(
            sitk.GetImageFromArray(x), join_path([target_path, study_name + ".nii.gz"])
        )
        sitk.WriteImage(
            sitk.GetImageFromArray(y),
            join_path([target_path, study_name + "_gt.nii.gz"]),
        )


def harp_extract_images(source_path, target_path, subset):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    def bbox_3D(img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    # Folder 100 is for training (100 subjects), 35 subjects are left over for validation
    affine = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    images_path = os.path.join(source_path, subset)
    labels_path = os.path.join(source_path, f"Labels_{subset}_NIFTI")

    # Create directories
    if not os.path.exists(target_path):
        os.makedirs(os.path.join(target_path))

    # For each MRI, there are 2 segmentation (left and right hippocampus)
    for filename in os.listdir(images_path):
        # Loading the .mnc file and converting it to a .nii.gz file
        minc = nib.load(os.path.join(images_path, filename))
        x = nib.Nifti1Image(minc.get_data(), affine=affine)

        # We need to recover the study name of the image name to construct the name of the segmentation files
        match = re.match(r"ADNI_[0-9]+_S_[0-9]+_[0-9]+", filename)
        if match is None:
            raise Exception(
                f"A file ({filename}) does not match the expected file naming format"
            )

        # For each side of the brain
        for side in ["_L", "_R"]:
            study_name = match[0] + side

            y = sitk.ReadImage(os.path.join(labels_path, study_name + ".nii"))
            y = sitk.GetArrayFromImage(y)

            # Shape expected: (189, 233, 197)
            # Average label shape (Training): (27.1, 36.7, 22.0)
            # Average label shape (Validation): (27.7, 35.2, 21.8)
            assert x.shape == y.shape
            # Disclaimer: next part is ugly and not many checks are made
            # BUGFIX: Some segmentation have some weird values eg {26896.988, 26897.988} instead of {0, 1}
            y = (y - np.min(y.flat)).astype(np.uint32)

            # So we first compute the bounding box
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(y)

            # Compute the start idx for each dim
            dr = (rmax - rmin) // 4
            dc = (cmax - cmin) // 4
            dz = (zmax - zmin) // 4

            # Reshaping
            y = y[rmin - dr : rmax + dr, cmin - dc : cmax + dc, zmin - dz : zmax + dz]

            x_cropped = x.get_data()[
                rmin - dr : rmax + dr, cmin - dc : cmax + dc, zmin - dz : zmax + dz
            ]

            x_cropped = (x_cropped - np.min(x_cropped)) / (
                np.max(x_cropped) - np.min(x_cropped)
            )
            x_cropped = np.clip(x_cropped, 0.001, 0.99)

            # Save new images so they can be loaded directly
            sitk.WriteImage(
                sitk.GetImageFromArray(y),
                join_path([target_path, study_name + "_gt.nii.gz"]),
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(x_cropped),
                join_path([target_path, study_name + ".nii.gz"]),
            )


if __name__ == "__main__":
    storage_data_path = "/Share8/zhuzhanshi/CauAug/storage/data"

    """   -------------------------  """
    global_name = "HarP"
    print(global_name)
    default = {"Part": "All"}
    subset = default
    name = du.get_dataset_name(global_name, subset)
    dataset_path = os.path.join(storage_data_path, global_name)
    original_data_path = (
        "/Share8/zhuzhanshi/download/CL_origin/continual hippocampus/HarP"
    )
    folders = []
    if subset["Part"] in ["Training", "All"]:
        folders.append(("100", "Training"))
    if subset["Part"] in ["Validation", "All"]:
        folders.append(("35", "Validation"))

    for orig_folder, dst_folder in folders:
        # Paths with the sub-folder for the current subset
        dst_folder_path = os.path.join(dataset_path, dst_folder)

        harp_extract_images(original_data_path, dst_folder_path, orig_folder)

    """   -------------------------  """
    global_name = "DecathlonHippocampus"
    print(global_name)
    dataset_path = os.path.join(
        storage_data_path, global_name, "Merged Labels" if True else "Original"
    )
    original_data_path = (
        "/Share8/zhuzhanshi/download/CL_origin/continual hippocampus/DecathonHippocampus"
    )
    decathlon_extract_images(original_data_path, dataset_path, merge_labels=True)

    """   -------------------------  """
    global_name = "DryadHippocampus"
    print(global_name)
    default = {"Modality": "T1w", "Resolution": "Standard"}
    subset = default
    name = du.get_dataset_name(global_name, subset)
    dataset_path = os.path.join(
        storage_data_path,
        global_name,
        "Merged Labels" if True else "Original",
        "".join([f"{key}[{subset[key]}]" for key in ["Modality", "Resolution"]]),
    )
    original_data_path = (
        "/Share8/zhuzhanshi/download/CL_origin/continual hippocampus/DryadHippocampus"
    )
    dryad_extract_images(original_data_path, dataset_path, True, subset)
