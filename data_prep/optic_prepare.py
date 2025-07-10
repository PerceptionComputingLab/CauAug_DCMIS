import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil
import cv2


def rescaling(img):
    image_array = sitk.GetArrayFromImage(img)
    image_array = (image_array - np.min(image_array)) / (
        np.max(image_array) - np.min(image_array)
    )
    image_array = np.clip(image_array, 0.001, 0.99)
    image_new = sitk.GetImageFromArray(image_array)
    image_new.SetOrigin(img.GetOrigin())
    image_new.SetSpacing(img.GetSpacing())
    image_new.SetDirection(img.GetDirection())
    return image_new


H_max = 0
W_max = 0
if __name__ == "__main__":
    groups = {}
    datapath = "/Share8/zhuzhanshi/download/Fundus"
    data_out = "/Share8/zhuzhanshi/CauAug/storage/data"

    hw = 192
    # resize image to hw*hw
    for group in os.listdir(datapath):
        # if not a dictionary, continue
        if not os.path.isdir(os.path.join(datapath, group)):
            continue
        print(group)
        new_dir = os.path.join(data_out, group)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        # move file to new path
        for patient in os.listdir(
            os.path.join(datapath, group, "train", "ROIs", "mask")
        ):
            # add _gt to the end of the file name
            source = os.path.join(datapath, group, "train", "ROIs", "mask", patient)
            target = os.path.join(new_dir, patient)
            target = target.replace(".png", "_gt.png")
            # shift mask from [255,255,255] to 0; from [0,0,0] to 1, from [128,128,128] to 2
            old_mask = cv2.imread(source)
            mask = np.ones(old_mask.shape[:2], dtype=np.uint8)
            mask[old_mask[:, :, 0] == 255] = 0
            mask[old_mask[:, :, 0] == 0] = 1
            mask[old_mask[:, :, 0] == 128] = 2
            # resize mask to hw*hw
            mask = cv2.resize(mask, (hw, hw), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        for patient in os.listdir(
            os.path.join(datapath, group, "train", "ROIs", "image")
        ):
            source = os.path.join(datapath, group, "train", "ROIs", "image", patient)
            target = os.path.join(new_dir, patient)
            # resize image to hw*hw
            image = cv2.imread(source)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (hw, hw), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # shutil.copy(source, target)

        for patient in os.listdir(
            os.path.join(datapath, group, "test", "ROIs", "mask")
        ):
            # add _gt to the end of the file name
            source = os.path.join(datapath, group, "test", "ROIs", "mask", patient)
            target = os.path.join(new_dir, patient)
            target = target.replace(".png", "_gt.png")
            # shift mask from [256,256,256] to 0; from [0,0,0] to 1, from [128,128,128] to 2
            old_mask = cv2.imread(source)
            mask = np.ones(old_mask.shape[:2], dtype=np.uint8)
            mask[old_mask[:, :, 0] == 255] = 0
            mask[old_mask[:, :, 0] == 0] = 1
            mask[old_mask[:, :, 0] == 128] = 2
            # resize mask to hw*hw
            mask = cv2.resize(mask, (hw, hw), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        for patient in os.listdir(
            os.path.join(datapath, group, "test", "ROIs", "image")
        ):
            source = os.path.join(datapath, group, "test", "ROIs", "image", patient)
            target = os.path.join(new_dir, patient)
            # resize image to hw*hw
            image = cv2.imread(source)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (hw, hw), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
