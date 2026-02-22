# Wi-Fi Indoor Positioning

![Universitat Jaume I](images/dom_jaime.png)

![R](https://img.shields.io/badge/Language-R-276DC3?logo=r&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UJIIndoorLoc-orange)
![ML](https://img.shields.io/badge/Task-Classification%20%26%20Regression-blueviolet)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-lightgrey)

> Predicting a user's indoor location (building, floor, latitude, longitude) from Wi-Fi signal strength fingerprints using KNN, SVM, and C5.0 models in R.

---

## Objective

The goal of this project is to explore the [UJIIndoorLoc](https://archive.ics.uci.edu/ml/datasets/ujiindoorloc) dataset provided by the **UC Irvine Machine Learning Repository** and apply machine learning techniques to predict the location of a user in a university campus based on the intensity of the signals received by different wireless access points (WAPs) across campus from the user's Wi-Fi device.

---

## Background

Many real-world applications need to know the localization of a user to provide their services. Outdoor localization is solved accurately via GPS, but **indoor localization** remains an open problem due to GPS signal loss inside buildings. This project uses **WLAN fingerprint-based positioning** — measuring the signal strengths from many access points at different known locations, then learning a model that maps signal patterns to physical positions.

---

## Dataset

The **UJIIndoorLoc** database covers three buildings of Universitat Jaume I (Spain) with 4–5 floors each, collected in 2013 using 20+ users and 25 Android devices.

| File | Records | Description |
|---|---|---|
| `trainingData.csv` | 19,937 | Training / reference observations |
| `validationData.csv` | 1,111 | Held-out test observations |

Each record has **529 attributes**:

| Attributes | Description |
|---|---|
| WAP001–WAP520 | Signal strength (dBm). Range: −104 to 0 (detected) or 100 (not detected) |
| LONGITUDE | Longitude coordinate |
| LATITUDE | Latitude coordinate |
| FLOOR | Floor number (0–4) |
| BUILDINGID | Building identifier (0–2) |
| SPACEID | Room/corridor ID |
| RELATIVEPOSITION | Inside (1) or outside door (2) |
| USERID | User identifier |
| PHONEID | Android device identifier |
| TIMESTAMP | UNIX timestamp of capture |

---

## How to Run

### 1. Download the Dataset

The CSVs are not included in this repository (size ~110 MB). Download them from the UCI repository:

👉 [**Download UJIIndoorLoc Dataset**](https://archive.ics.uci.edu/ml/machine-learning-databases/00310/)

Place the two CSV files inside a `data/` folder in the project root:

```
wifi-positioning/
└── data/
    ├── trainingData.csv
    └── validationData.csv
```

### 2. Install R Dependencies

Open R (or RStudio) and install the required packages:

```r
install.packages(c("here", "caret", "readr", "dplyr", "tidyr",
                   "data.table", "ggplot2", "RWeka", "ggthemes", "e1071"))
```

> **Note:** The `RWeka` package requires Java. Install a JDK and run `options(java.parameters = "-Xmx4g")` before loading it if you encounter memory errors.

### 3. Run the Analysis

Open the project in RStudio by double-clicking `wifi-positioning.Rproj`, then open and run:

```
wifi_positioning_analysis.R
```

The script is fully self-contained and runs end-to-end: data loading → wrangling → normalisation → model training → validation.

---

## Data Preparation

Before modelling, the following cleaning steps were applied:

| Step | Rationale |
|---|---|
| Converted undetected signals (100) and signals ≤ −90 to −100 | Brings consistency to very weak signals |
| Removed zero-variance WAP columns | WAPs with no variation across observations carry no information |
| Removed duplicate rows | Avoids redundant training signal |
| Removed User 6 | 430 out of 976 observations had anomalously strong signals — likely a faulty device |
| Capped signals above −30 dBm to −100 | Perfect signals are unrealistic and treated as measurement errors |
| Normalised by row | Chosen over column-wise and global normalisation after comparing model performance |
| Sampled 3,000 training records | SVM is computationally expensive on 520-dimensional data; sampling preserves representativeness |

---

## Results

### Building & Floor Classification

Models were trained on the full campus first (building prediction), then per-building (floor prediction). SVM achieved the best overall accuracy:

|  | KNN | SVM | SVM3 | C5.0 |
|---|---|---|---|---|
| **Building** | 0.984 | 1.000 | 1.000 | 0.999 |
| **Floor – Building 0** | 0.948 | 0.976 | 0.948 | 0.970 |
| **Floor – Building 1** | 0.959 | 1.000 | 0.984 | 0.970 |
| **Floor – Building 2** | 0.950 | 0.982 | 0.979 | 0.970 |

SVM achieved **100% building accuracy** and the best floor prediction rates, so it was selected for the final validation phase.

![Floor prediction results on the test set](images/floor_results2.PNG)

### Latitude & Longitude Regression

The same hierarchical approach (whole campus → per building) was applied for coordinate regression. KNN significantly outperformed SVM on this task:

| RMSE | KNN | SVM |
|---|---|---|
| **Latitude** | 7.07 | 18.87 |
| **Longitude** | 6.65 | 32.20 |
| **Latitude – Building 0** | 6.10 | 10.18 |
| **Longitude – Building 0** | 5.49 | 12.04 |
| **Latitude – Building 1** | 7.07 | 12.61 |
| **Longitude – Building 1** | 7.77 | 14.15 |
| **Latitude – Building 2** | 7.48 | 14.18 |
| **Longitude – Building 2** | 7.24 | 14.67 |

After collecting predicted coordinates, the **Euclidean distance** between actual and predicted positions was calculated. Comparing the scatter plots below confirms a good degree of positional accuracy — the predicted layout closely mirrors the actual campus shape.

**Actual coordinates:**

![Actual coordinates on campus](images/actual.jpg)

**Predicted coordinates:**

![Predicted coordinates on campus](images/predicted.jpg)

---

## Project Structure

```
wifi-positioning/
├── images/                         # Result plots (floor accuracy, coordinate maps)
├── All Code.R                      # Consolidated end-to-end analysis
├── Data Wrangling.r                # Data loading & preprocessing steps
├── Building.r                      # Building classification models
├── Floor.R                         # Floor classification (whole campus)
├── Floor - Building 0.R            # Floor classification – Building 0
├── Floor - Building 1.R            # Floor classification – Building 1
├── Floor - Building 2.R            # Floor classification – Building 2
├── Latitude.R                      # Latitude regression (whole campus)
├── Latitude - Building 0.R         # Latitude regression – Building 0
├── Latitude - Building 1.R         # Latitude regression – Building 1
├── Latitude - Building 2.R         # Latitude regression – Building 2
├── Longitude.R                     # Longitude regression (whole campus)
├── Longitude - Building 0.R        # Longitude regression – Building 0
├── Longitude - Building 1.R        # Longitude regression – Building 1
├── Longitude - Building 2.R        # Longitude regression – Building 2
├── Lat_Long - Error.R              # Euclidean error analysis
├── wifi_positioning_analysis.R     # Cleaned-up consolidated script
├── dataset_link.md                 # Dataset download instructions & citation
├── wifi-positioning.Rproj          # RStudio project file
└── README.md
```

---

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Joaquín Torres-Sospedra et al., 2014.
