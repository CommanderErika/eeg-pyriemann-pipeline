import os

import mne
import moabb.utils

def set_download_dir(dir: str) -> None:
    mne.utils.set_config(key="MNE_DATA", value=dir)
    #mne.utils.set_config(key="MNE_DATASETS_BNCI_PATH", value=dir+"/datasets")
    #create_folder(folder_name=dir+'/datasets')
    #moabb.utils.set_download_dir(path=dir)

def create_folder(folder_name: str) -> None:
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
