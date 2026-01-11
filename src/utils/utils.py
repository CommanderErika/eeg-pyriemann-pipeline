import os

import mne
import moabb.utils
import numpy as np

def set_download_dir(dir: str) -> None:
    mne.utils.set_config(key="MNE_DATA", value=dir)
    #mne.utils.set_config(key="MNE_DATASETS_BNCI_PATH", value=dir+"/datasets")
    #create_folder(folder_name=dir+'/datasets')
    #moabb.utils.set_download_dir(path=dir)

def create_folder(folder_name: str) -> None:
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def get_epoch_num(text: str) -> int:
    return int(text.replace(":", "").split(" ")[-1])

def get_count_samples(text: str) -> int:
    # Split in line
    lines: list[str] = text.split("\n")
    # Iterate over
    count: int = 0
    for line in lines:
        if "Epoch" in line:
            count += 1
    return count

def get_count_samples_ts(text: str) -> int:
    # Split in line
    lines: list[str] = text.split("\n")
    # Iterate over
    count: int = 0
    for line in lines:
        if "TS" in line:
            count += 1
    return count

def line_to_array(text: str) -> np.ndarray:
    lines = text.split(",")
    if len(lines) <= 1:
        lines = text.split(" ")
    samples = [float(sample) for sample in lines if sample != ""]
    return np.asarray(samples)

def line_to_array_c(text: str, n_channels: int) -> np.ndarray:
    lines = text.split(",")
    if len(lines) <= 1:
        lines = text.split(" ")
    samples = [float(sample) for sample in lines if sample != ""]
    return np.asarray(samples).reshape(n_channels, n_channels)

def read_file_cov_c(file_name: str, n_channels: int) -> np.ndarray:
    # Reading file
    file_samples = open(file=file_name, mode="r")
    text: str = file_samples.read()
    # Text split by "\n"
    lines: list[str] = text.split("\n")
    # Get Cov
    n_samples: int = get_count_samples(text)
    epoch: int = -1
    epochs: np.ndarray = np.zeros(shape=(n_samples, n_channels, n_channels))
    ch_count = 0
    for line in lines:
        if line == "\n":
            continue
        if "Covariance matrix from Epoch" in line:
            epoch = get_epoch_num(line)
            ch_count = 0
        elif epoch != -1:
            epochs[epoch, :, :] = line_to_array_c(line, n_channels)
            ch_count += 1
            if ch_count == 1:
                epoch = -1
    file_samples.close()
    return epochs

def read_file_ts_c(file_name: str, n_channels: int) -> np.ndarray:
    # Reading file
    file_samples = open(file=file_name, mode="r")
    text: str = file_samples.read()
    # Text split by "\n"
    lines: list[str] = text.split("\n")
    # Get Cov
    n_samples: int = get_count_samples(text)
    epoch: int = -1
    N: int = int((n_channels * (n_channels + 1)) / 2)
    epochs: np.ndarray = np.zeros(shape=(n_samples, 253))
    ch_count = 0
    for line in lines:
        if line == "\n":
            continue
        if "Tangent Space for Epoch" in line:
            epoch = get_epoch_num(line)
            ch_count = 0
        elif epoch != -1:
            epochs[epoch, :] = line_to_array(line)
            ch_count += 1
            if ch_count == 1:
                epoch = -1
    file_samples.close()
    return epochs

def cal_mse_ts(array_p: np.ndarray, array_c: np.ndarray):
    error_array = (array_p - array_c) ** 2
    return error_array.mean(axis=0)

def cal_mse_cov(array_p: np.ndarray, array_c: np.ndarray):
    error_array = (array_p - array_c) ** 2
    return error_array.mean(axis=0).flatten()

def calculate_mse(matrix_py: np.ndarray, matrix_cpp: np.ndarray):
    return np.mean((matrix_py - matrix_cpp) ** 2, axis=0)

def calculate_mae(matrix_py: np.ndarray, matrix_cpp: np.ndarray):
    return np.mean(np.abs(matrix_py - matrix_cpp), axis=0)

def calculate_rms(matrix_py: np.ndarray, matrix_cpp: np.ndarray):
    return np.sqrt(calculate_mse(matrix_py, matrix_cpp))