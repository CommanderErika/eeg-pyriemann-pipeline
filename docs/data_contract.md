# 💾 HDF5 Data Interface & Schema

This document defines the **Input/Output Data Contract** for the project.

The system uses a **Strategy Pattern** via the `HDF5Manager` class. It automatically selects the correct loading strategy based on a specific metadata attribute (`data_type`) found in the file header.

Any HDF5 file generated for this pipeline must strictly adhere to one of the schemas defined below.

---

## 1. Global Metadata (The Contract Header)

Every HDF5 file **must** contain the following attributes in the root group (`/`). These attributes determine how the file is parsed.

| Attribute Key | Data Type | Required? | Description | Valid Values |
| :--- | :--- | :--- | :--- | :--- |
| **`data_type`** | String | **YES** | The "Discriminator" that selects the loading strategy. | `'moabb'`, `'covariances'`, `'tangent'`, `'riemann_tangent_space'` |
| **`version`** | String | **YES** | Schema version compatibility. | `'1.0'` |
| **`channel_names`** | Array (String/Bytes) | No | List of channel labels (e.g., `['C3', 'Cz', 'C4']`). | Any string list |
| **`freqr`** | Float/Int | No | Sampling frequency in Hz. | Positive number |

---

## 2. Data Schemas (Strategies)

### 2.1. Tangent Space Data
* **Attribute Value:** `data_type = 'tangent'` (or `'riemann_tangent_space'`)
* **Use Case:** Processed features (Tangent Vectors) ready for Machine Learning classifiers (SVM/LDA).
* **Python Dataclass:** `TangentSpaceData`

| Dataset Path | Shape | Type | Description |
| :--- | :--- | :--- | :--- |
| **/tangent** | $[N, F]$ | `Float32` | **Features Matrix**. <br>$N$: Trials<br>$F$: Features ($C(C+1)/2$) |
| **/labels** | $[N]$ | `Int` or `String` | **Target Vector**. Ground truth labels for classification. |
| **/subjects** | $[N]$ | `Int` | **Group Vector**. Subject IDs used for LOSO cross-validation. |

*> **Note on Loading:** The loader is flexible. It will look for dataset keys `['x', 'tangent']` for features and `['y', 'labels']` for labels.*

### 2.2. Covariance Data
* **Attribute Value:** `data_type = 'covariances'`
* **Use Case:** Intermediate spatial filtering results (SPD Matrices) before projection to Tangent Space.
* **Python Dataclass:** `CovarianceData`

| Dataset Path | Shape | Type | Description |
| :--- | :--- | :--- | :--- |
| **/covariances** | $[N, C, C]$ | `Float32` | **Covariance Matrices**. <br>$N$: Trials<br>$C$: Channels |
| **/labels** | $[N]$ | `Int` or `String` | **Target Vector**. |
| **/subjects** | $[N]$ | `Int` | **Group Vector**. |

*> **Note on Loading:** The loader looks for dataset keys `['covariances', 'x', 'convariances']`.*

### 2.3. Raw MOABB Data
* **Attribute Value:** `data_type = 'moabb'`
* **Use Case:** Raw EEG trials downloaded from MOABB or acquired from APIs.
* **Python Dataclass:** `MOABBData`

| Dataset Path | Shape | Type | Description |
| :--- | :--- | :--- | :--- |
| **/eeg_data** | $[N, C, T]$ | `Float32` | **Raw EEG**. <br>$N$: Trials<br>$C$: Channels<br>$T$: Timepoints |
| **/labels** | $[N]$ | `Int` or `String` | **Target Vector**. |
| **/subjects** | $[N]$ | `Int` | **Group Vector**. |

---

## 3. Encoding & Technical Details

### 3.1. String Handling (Labels & Metadata)
HDF5 has complex string handling. The `HDF5Manager` enforces specific rules to ensure compatibility between Python (UTF-8) and C/C++ (Bytes/Fixed-width).

* **Saving:** Strings are automatically encoded as fixed-width ASCII/Bytes (`dtype='S64'`) or Variable-Length UTF-8 depending on the numpy version.
* **Loading:** The manager includes an auto-decoder:
    1.  Checks if data is `bytes` (`S` or `a` dtype).
    2.  If yes, decodes using `utf-8`.
    3.  Strips null-padding (`\x00`) common in C++ fixed-width arrays.

### 3.2. Data Types
* **Features:** Stored as `Float32` (Single Precision) to save space, but loaded into memory compatible with Scikit-Learn (which may cast to Float64 internally).
* **Subjects:** Must be Integers.

---

## 4. Example Layout (Visual)

A valid **Tangent Space** file (`BNCI2014_001.h5`) structure looks like this:

```text
BNCI2014_001.h5
├── / (Root Group)
│   ├── Attributes:
│   │   ├── 'data_type': 'tangent'
│   │   ├── 'version': '1.0'
│   │   └── 'channel_names': ['C3', 'Cz', 'C4']
│   │
│   ├── /tangent      (Dataset: 5184 x 253, float32)
│   ├── /labels       (Dataset: 5184, string)
│   └── /subjects     (Dataset: 5184, int)