## Project Introduction

This project is used to build machine learning models based on raster environmental factor data and sample observation data for spatial modeling, prediction, and result interpretation of soil organic matter.  
The core scripts are `model.py` (data reading, feature selection, model training/evaluation, spatial prediction and visualization menu) and `shap_analysis.py` (provides the `SHAPAnalyzer` class, called by `model.py` to complete SHAP and PDP analyses).

## Directory Structure

- **`model.py`**: Main program that completes data reading, feature selection (based on correlation with target `O` and significance testing), multi-model grid search and evaluation, optimal model selection, SHAP analyzer initialization, interactive menu, and raster full-image prediction.
- **`shap_analysis.py`**: Defines the `SHAPAnalyzer` class, imported and used by `model.py` for calculating and plotting SHAP importance, summary plot, PDP/ICE curves, and batch PDP summary plots.
- **`2010.csv`**: Sample data file (2010 sampling points and their attributes, target variable and environmental factor information).
- **`2010preprocessed/`**: Preprocessed raster data directory for 2010 (environmental factor rasters).
  - `Elevation.tif`: Elevation
  - `MAP.tif`: Mean Annual Precipitation
  - `MRRTF.tif`: Multi-Resolution Ridge Top Flatness
  - `MRVBF.tif`: Multi-Resolution Valley Bottom Flatness
  - `N.tif`: Soil nitrogen content
  - `P.tif`: Soil phosphorus content
  - `PH.tif`: Soil pH
- **`2024preprocessed/`**: Preprocessed raster data directory for 2024, used for extrapolation prediction based on trained models.
  - File structure is similar to `2010preprocessed/`.
- **`selected_features_corr.csv`**: Feature list or feature correlation matrix obtained through correlation filtering, used for feature selection in subsequent modeling.


## Runtime Environment and Dependencies

It is recommended to use **Python 3.10+** and run this project in an isolated virtual environment (this project has been tested on **Python 3.12.4**, **Python 3.12.4** is recommended).

Core dependencies (directly used in `model.py` / `shap_analysis.py`):

- **Basic Scientific Computing**
  - `numpy (==1.26.4)`
  - `pandas (==2.2.0)`
  - `scipy (==1.14.1)`
- **Machine Learning / Modeling**
  - `scikit-learn (==1.6.0)`
  - `lightgbm (==4.6.0)`
- **Visualization and Analysis**
  - `matplotlib (==3.9.4)`
  - `seaborn (==0.13.2)`
  - `shap (==0.48.0)`
- **Geospatial / Raster Processing**
  - `rasterio (==1.4.3)`
- **Image Processing / Dependency Support**
  - `Pillow (==11.0.0)` (usually as a dependency of libraries like `rasterio`, managed uniformly in `requirements.txt`)

> For complete and reproducible dependency versions, please refer to `requirements.txt` in the project root directory.

The project provides `requirements.txt`, recommended installation:

```bash
pip install -r requirements.txt
```

If installation fails, you can manually install according to the above list, for example:

```bash
pip install -U numpy==1.26.4 pandas==2.2.0 scipy==1.14.1 scikit-learn==1.6.0 lightgbm==4.6.0 rasterio==1.4.3 shap==0.48.0 matplotlib==3.9.4 seaborn==0.13.2 Pillow==11.0.0
```

## Quick Start

### 1. Prepare Python Environment

Execute in the project root directory (directory containing `model.py`):

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell

pip install -r requirements.txt  
```

### 2. Data Preparation Instructions

- Ensure the following files/folders are in the project root directory:
  - `2010.csv`
  - `2010preprocessed/`
  - `2024preprocessed/`
  - `selected_features_corr.csv`
- Raster data for different years should have:
  - Consistent spatial reference system;
  - Consistent resolution;
  - Aligned spatial extent.

- `2010.csv` contains:
  - Spatial location fields (latitude and longitude);
  - Target variable field (the indicator to be modeled, i.e., soil organic matter);
  - Environmental factor fields corresponding to each raster file.

### 3. Run Main Model Script

Execute in the project root directory:

```bash
python model.py
```

After the script starts, it will automatically:

- Read `2010.csv`, use column `O` as the target variable, and the remaining columns as candidate features;
- Perform correlation and significance testing on training set features and target `O`, automatically filter features according to `method='pearson', threshold=0.2, alpha=0.01`, and save as `selected_features_corr.csv`;
- Use the filtered features to perform 5-fold `GridSearchCV` grid search on Random Forest, Gradient Boosting, ExtraTrees, and LightGBM;
- Select the model with the highest test set R² as the optimal model, and initialize `SHAPAnalyzer`;
- Enter the **interactive command-line menu**, execute subsequent analysis and raster prediction based on your input.

> Note: `model.py` works through command-line interaction (`input()`), please run directly in the terminal and enter options according to prompts. It does not support non-interactive batch processing parameter mode.

## Interactive Menu Function Overview

After running `python model.py`, a menu will be displayed in a loop (excerpt):

- **1–4: View Model Accuracy and Optimal Hyperparameters**
  - **1**: View current optimal model (name, method, test set R²/RMSE/MAE).
  - **2**: View optimal hyperparameters of the optimal model.
  - **3 / 4**: Output test set / training set accuracy metrics respectively.
- **5–9,10: Scatter Plots and Multi-Model Comparison**
  - **5 / 6**: "Actual vs Predicted" scatter plot of the optimal model on test set / training set.
  - **7**: Read any CSV prediction data, predict using the optimal model and save as `predict_result.csv`.
  - **8**: Print R² / RMSE / MAE comparison of all models on training set and test set.
  - **9**: Select any model to output training set and test set scatter plots.
  - **10**: Output optimal hyperparameters of all models.
- **11–13,19,21: SHAP and PDP Analysis**
  - **11**: SHAP overall importance bar chart.
  - **12**: SHAP summary plot.
  - **13**: Single feature PDP.
  - **19**: SHAP importance ranking plot with contribution rate pie chart.
  - **21**: Batch generate PDP plots for all features and summary mosaic.
- **14: Correlation Heatmap**
  - Generate correlation matrix heatmap using all original data (including target `O`).
- **15–18,20: Raster Full-Image Prediction and Raster SHAP**
  - **15**: Use the "current optimal model" to perform full-image prediction on rasters in the specified directory, output prediction `GeoTIFF` (optional preprocessing, supports automatic calculation of Lat/Lon).
  - **16**: Independent raster preprocessing function to unify multiple rasters to the same size, resolution, and projection.
  - **17**: Select any one from all trained models as the raster prediction model.
  - **18**: Calculate SHAP value rasters for all environmental factors.
  - **20**: Calculate SHAP value raster for a **single** environmental factor and output detailed statistics.
- **0: Exit Program**

The specific prompt text, output file names, and save directories for each menu item are printed in English in `model.py` and can be adjusted as needed.

## Notes

- Please ensure all input file paths are set correctly in the script to avoid "file not found" errors.
- Raster data is large and has high resolution, so runtime and memory usage may be significant. It is recommended to run on a machine with good performance.
- When errors occur, you can focus on checking:
  - Whether Python version and dependency package versions are compatible;
  - Whether CSV field names match the column names read in the code;
  - Whether the coordinate system and resolution of raster data are unified.

---
