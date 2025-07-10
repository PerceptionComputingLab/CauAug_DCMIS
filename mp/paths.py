# ------------------------------------------------------------------------------
# Module where paths should be defined.
# ------------------------------------------------------------------------------
import os

# Path where intermediate and final results are stored
# abs_path = os.path.abspath('.')
abs_path = "/Share8/zhuzhanshi/CauAug"
storage_path = "storage"
storage_data_path = os.path.join(abs_path, storage_path, "data")

original_data_paths = {
    "DecathlonHippocampus": abs_path + "/storage/data/DecathlonHippocampus",
    "DryadHippocampus": abs_path + "/storage/data/DryadHippocampus",
    "HarP": abs_path + "/storage/data/HarP",
}
