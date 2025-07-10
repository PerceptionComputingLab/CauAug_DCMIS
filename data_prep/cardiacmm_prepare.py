import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pandas


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def crop_pad_data(img, to_size=(None, 256, 256)):
    shape = to_size
    for idx, size in enumerate(shape):
        if size is not None and size < img.shape[idx]:
            # crop current dimension
            before = (img.shape[idx] - size) // 2
            after = (
                img.shape[idx]
                - (img.shape[idx] - size) // 2
                - ((img.shape[idx] - size) % 2)
            )
            slicing = [slice(None)] * img.ndim
            slicing[idx] = slice(before, after)
            img = img[tuple(slicing)]

        elif size is not None and size > img.shape[idx]:
            # pad current dimension
            before = (size - img.shape[idx]) // 2
            after = (size - img.shape[idx]) // 2 + ((size - img.shape[idx]) % 2)
            pad_width = [(0, 0)] * img.ndim
            pad_width[idx] = (before, after)
            img = np.pad(img, pad_width, mode="constant", constant_values=0)

    return img


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
    datapath = "/Share8/zhuzhanshi/download/CL_origin/mm"
    data_out = "/Share8/zhuzhanshi/CauAug/storage/data"
    file_name = os.path.join(
        datapath, "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    )
    df = pandas.read_csv(file_name)
    # print(df['VendorName'])

    for index, row in df.iterrows():
        if row["VendorName"] not in groups:
            groups[row["VendorName"]] = []
            groups[row["VendorName"]].append(
                os.path.join(datapath, row["External code"])
            )
        else:
            groups[row["VendorName"]].append(
                os.path.join(datapath, row["External code"])
            )
    if not os.path.exists(data_out):
        os.mkdir(data_out)

    for gro in groups:
        print(gro)
        if not os.path.exists(os.path.join(data_out, gro)):
            os.mkdir(os.path.join(data_out, gro))
        for in_path in groups[gro]:
            if not os.path.exists(os.path.join(datapath, in_path)):
                continue
            for name in os.listdir(os.path.join(datapath, in_path)):

                if "frame" in name:
                    in_path_frame = os.path.join(datapath, in_path, name)
                else:
                    continue
                print(name)
                data, affine, header = load_nii(in_path_frame)
                data = data.transpose(2, 0, 1)
                z, h, w = data.shape
                data = crop_pad_data(data)
                if "gt" in name:
                    data = data.astype(np.uint8)
                data = sitk.GetImageFromArray(data)
                if "gt" not in name:
                    data = rescaling(data)
                sitk.WriteImage(
                    data,
                    os.path.join(
                        data_out,
                        gro,
                        in_path_frame.split("/")[-2]
                        + "_"
                        + in_path_frame.split("/")[-1],
                    ),
                )
