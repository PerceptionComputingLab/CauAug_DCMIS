from mp.data.datasets.ds_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_hippocampus_dryad import DryadHippocampus
from mp.data.datasets.ds_hippocampus_harp import HarP


def Hippocampus(subset):
    if subset == "DecathlonHippocampus":
        return DecathlonHippocampus()
    elif subset == "DryadHippocampus":
        return DryadHippocampus()
    elif subset == "HarP":
        return HarP()
