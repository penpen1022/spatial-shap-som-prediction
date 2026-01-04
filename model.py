import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
from shap_analysis import SHAPAnalyzer
import os
import glob
from scipy.stats import pearsonr, spearmanr
from scipy import stats
try:
    import rasterio
    from rasterio.transform import Affine
except Exception:
    rasterio = None
warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'font.family': ['sans-serif'],
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif'],
    'axes.unicode_minus': False,  
    'axes.labelsize': 12,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True,
})
# Read dataset
df = pd.read_csv('2010.csv')
# Define features and target variable
y = df.O
x = df.drop(['O'], axis=1)
# ========== Feature selection based on correlation with target column ==========
def select_features_by_correlation(X, y, method='pearson', top_k=None, threshold=None, alpha=0.01):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y_series = pd.Series(y)
    if method not in ('pearson', 'spearman'):
        method = 'pearson'
    X_numeric = X.apply(pd.to_numeric, errors='coerce')
    # Align indices, drop samples containing NaN
    data = pd.concat([X_numeric, y_series.rename('__target__')], axis=1)
    data = data.dropna(axis=0, how='any')
    if data.empty:
        # If no data after cleaning, fall back to original columns
        return X, X.columns
    
    # Calculate correlation
    corr = data.corr(method=method)['__target__'].drop('__target__')
    corr_abs = corr.abs().sort_values(ascending=False)
    
    # Calculate significance test p-values for each feature
    y_clean = data['__target__'].values
    significant_features = []
    p_values = {}
    
    for feat in corr_abs.index:
        x_feat = data[feat].values
        # Select corresponding test function based on method
        if method == 'pearson':
            _, p_val = pearsonr(x_feat, y_clean)
        else:  # spearman
            _, p_val = spearmanr(x_feat, y_clean)
        
        p_values[feat] = p_val
        # Only keep features that pass significance test (p < alpha)
        if p_val < alpha:
            significant_features.append(feat)
    
    if len(significant_features) == 0:
        print(f"Warning: No features passed significance level α={alpha}, will keep the feature with highest correlation")
        significant_features = [corr_abs.index[0]]
    
    # Further filter based on top_k or threshold
    if top_k is not None and top_k > 0:
        significant_corr_abs = corr_abs[significant_features].sort_values(ascending=False)
        selected = significant_corr_abs.head(min(top_k, len(significant_corr_abs))).index
    elif threshold is not None:
        # Select features that meet threshold from those passing significance test
        significant_corr_abs = corr_abs[significant_features]
        selected = significant_corr_abs[significant_corr_abs >= float(threshold)].index
        if len(selected) == 0:
            # Prevent all from being filtered, fallback to select the top 1 feature that passed significance test and has highest correlation
            if len(significant_features) > 0:
                selected = pd.Index([significant_corr_abs.idxmax()])
            else:
                selected = corr_abs.head(1).index
    else:
        # Default: keep all features that passed significance test
        selected = pd.Index(significant_features)
    
    # Output significance test information
    print(f"\nSignificance test results (α={alpha}):")
    print(f"Number of features passing significance test: {len(significant_features)}/{len(corr_abs)}")
    
    return X.loc[:, selected], selected
grid_params = {
      'RandomForest': {
        'model': RandomForestRegressor(random_state=24),
        'params': {
            'n_estimators': [100, 150, 200],  
            'max_depth': [7, 8, 9],  
            'min_samples_split': [10, 15, 20],  
            'min_samples_leaf': [5, 8, 10]  
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=24),
        'params': {
            'n_estimators': [20, 30, 40],  
            'max_depth': [2, 3, 4],  
            'learning_rate': [0.02, 0.05, 0.1],  
            'subsample': [0.65, 0.75, 0.85],  
            'min_samples_split': [15, 20, 25], 
            'min_samples_leaf': [8, 10, 12],  
            'max_features': [0.6, 0.8, 1.0]  
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesRegressor(random_state=24),
        'params': {
            'n_estimators': [30, 50, 80],  
            'max_depth': [6, 8, 10],  
            'min_samples_split': [3, 5, 8],  
            'min_samples_leaf': [1, 2, 3],  
            'max_features': [0.8, 0.9, 1.0]  
        }
    },
    'LightGBM': {
         'model': lgb.LGBMRegressor(random_state=24, verbose=-1, force_col_wise=True),
        'params': {
            'n_estimators': [120, 160, 200],  
            'max_depth': [3, 4],  
            'learning_rate': [0.03, 0.05, 0.08],  
            'subsample': [0.75, 0.85],  
            'colsample_bytree': [0.75, 0.85],  
            'min_child_samples': [10, 12, 15],  
            'reg_alpha': [0, 0.05, 0.1],  
            'reg_lambda': [0.5, 1.0, 1.5],  
            'num_leaves': [20, 31]  
        }
    }
}
# Split training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=24)

X_train_corr, selected_features = select_features_by_correlation(
    x_train, y_train, method='pearson', top_k=None, threshold=0.2
)
X_test_corr = x_test.loc[:, selected_features]

# —— Output and save features retained after correlation filtering ——
print(f"\nNumber of features retained after correlation filtering: {len(selected_features)}")
print("Retained feature list:")
for i, feat in enumerate(selected_features, 1):
    print(f"{i:2d}. {feat}")

# Check if all features with |corr| >= 0.2 that passed significance test are included, if not, manually add them
if len(selected_features) < 9:  # According to previous analysis, there should be 9 features
    print("\nDetected that feature filtering may have missed relevant features, supplementing...")
    # Recalculate correlation and significance on full dataset
    data = pd.concat([x, y.rename('__target__')], axis=1)
    data = data.dropna(axis=0, how='any')
    corr = data.corr(method='pearson')['__target__'].drop('__target__')
    corr_abs = pd.Series(corr.abs())
    
    # Calculate significance test (α=0.01)
    y_clean = data['__target__'].values
    alpha = 0.01
    all_selected = []
    for feat in corr_abs.index:
        if corr_abs[feat] >= 0.2:  # Meet correlation threshold
            x_feat = data[feat].values
            _, p_val = pearsonr(x_feat, y_clean)
            if p_val < alpha:  # Pass significance test
                all_selected.append(feat)
    
    # Find missing features
    missing_features = [feat for feat in all_selected if feat not in selected_features]
    
    if missing_features:
        print(f"Supplementing missing features (passed significance test α={alpha}): {missing_features}")
        # Add missing features to selected_features
        selected_features = list(selected_features) + missing_features
        # Reselect features
        X_train_corr = x_train.loc[:, selected_features]
        X_test_corr = x_test.loc[:, selected_features]
        print(f"Updated number of features: {len(selected_features)}")

print(f"\nFinal number of retained features: {len(selected_features)}")
print("Final retained feature list:")
for i, feat in enumerate(selected_features, 1):
    print(f"{i:2d}. {feat}")

try:
    pd.Series(selected_features, name='selected_feature').to_csv('selected_features_corr.csv', index=False, encoding='utf-8-sig')
    print("Saved filtered features to: selected_features_corr.csv")
except Exception as e:
    print(f"Failed to save filtered features: {e}")

# Evaluation metrics
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# Correlation heatmap function (including significance stars)
def correlation_heatmap_with_significance(data, method='pearson', title='correlation heatmap', 
                                        figsize=(12, 10), save_path=None, star_position='below',
                                        font_size=8, star_font_size=6):
   
    # Calculate correlation matrix
    corr_matrix = data.corr(method=method)
    
    # Calculate significance test
    n = len(data)
    significance_matrix = np.zeros_like(corr_matrix, dtype=object)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j:
                # Calculate correlation coefficient and p-value
                corr_val, p_val = pearsonr(data.iloc[:, i], data.iloc[:, j])
                
                # Add stars based on p-value
                if p_val < 0.001:
                    significance_matrix[i, j] = '***'
                elif p_val < 0.01:
                    significance_matrix[i, j] = '**'
                elif p_val < 0.05:
                    significance_matrix[i, j] = '*'
                else:
                    significance_matrix[i, j] = ''
            else:
                significance_matrix[i, j] = ''
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Only show lower triangle
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8},
                linewidths=0.5,
                annot_kws={'fontsize': font_size},  # Control number font size
                ax=ax)
    
    # Add significance stars (adjust display position based on position parameter)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i > j:  # Only add stars in lower triangle
                if significance_matrix[i, j]:
                    # Adjust star position based on star_position parameter
                    if star_position == 'below':
                        # Stars displayed below numbers
                        ax.text(j + 0.5, i + 0.5 - 0.2, significance_matrix[i, j], 
                                ha='center', va='center', fontsize=star_font_size, 
                                fontweight='bold', color='black')
                    elif star_position == 'above':
                        # Stars displayed above numbers
                        ax.text(j + 0.5, i + 0.5 + 0.2, significance_matrix[i, j], 
                                ha='center', va='center', fontsize=star_font_size, 
                                fontweight='bold', color='black')
                    elif star_position == 'right':
                        # Stars displayed to the right of numbers
                        ax.text(j + 0.5 + 0.2, i + 0.5, significance_matrix[i, j], 
                                ha='center', va='center', fontsize=star_font_size, 
                                fontweight='bold', color='black')
                    elif star_position == 'left':
                        # Stars displayed to the left of numbers
                        ax.text(j + 0.5 - 0.2, i + 0.5, significance_matrix[i, j], 
                                ha='center', va='center', fontsize=star_font_size, 
                                fontweight='bold', color='black')
                    else:
                        # Default: display below
                        ax.text(j + 0.5, i + 0.5 - 0.2, significance_matrix[i, j], 
                                ha='center', va='center', fontsize=star_font_size, 
                                fontweight='bold', color='black')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {save_path}")
    
    # Try to display figure, if failed then only save
    try:
        plt.show()
        print("Heatmap displayed")
    except Exception as e:
        print(f"Unable to display heatmap window: {e}")
        print("Heatmap saved to file, please check the saved image")
    
    # Close figure to release memory
    plt.close(fig)
    
    # Print significance explanation
    print("\nSignificance level explanation:")
    print("*** p < 0.001 (highly significant)")
    print("**  p < 0.01  (very significant)")
    print("*   p < 0.05  (significant)")
    print("No star: p >= 0.05 (not significant)")
    
    return corr_matrix, significance_matrix

# Multi-model comparison - using original features
results = {}
best_test_r2 = -np.inf
best_model = None
best_model_name = None
best_params = None
best_train_pred = None
best_test_pred = None
best_method_suffix = 'Original'

# ========== Train models using only correlation-filtered features ==========
print(f"\n=== Using correlation-filtered features (method=pearson, threshold=0.2, alpha=0.01) ===")
for name, item in grid_params.items():
    print(f"Training model: {name} (number of features: {X_train_corr.shape[1]})")
    try:
        if item['params']:
            grid = GridSearchCV(item['model'], item['params'], scoring='r2', cv=5, n_jobs=-1)
            grid.fit(X_train_corr, y_train)
            model = grid.best_estimator_
            params = grid.best_params_
        else:
            model = item['model']
            model.fit(X_train_corr, y_train)
            params = None
        y_train_pred = model.predict(X_train_corr)
        y_test_pred = model.predict(X_test_corr)
        r2_train, rmse_train, mae_train = get_metrics(y_train, y_train_pred)
        r2_test, rmse_test, mae_test = get_metrics(y_test, y_test_pred)

        model_key = f"{name}_Corr"
        results[model_key] = {
            'model': model,
            'params': params,
            'train': {'r2': r2_train, 'rmse': rmse_train, 'mae': mae_train},
            'test': {'r2': r2_test, 'rmse': rmse_test, 'mae': mae_test},
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'method_suffix': 'Corr',
            'selected_features': selected_features,
            'x_train_selected': X_train_corr,
            'x_test_selected': X_test_corr
        }
        if r2_test > best_test_r2:
            best_test_r2 = r2_test
            best_model = model
            best_model_name = name
            best_params = params
            best_train_pred = y_train_pred
            best_test_pred = y_test_pred
            best_method_suffix = 'Corr'
        print(f"  {name} completed - Test set R2: {r2_test:.4f}")
    except Exception as e:
        print(f"  {name} training failed: {e}")
        continue

print(f"\nBest model: {best_model_name} (method: {best_method_suffix})")
print(f"Best test set R2: {best_test_r2:.4f}")

# Initialize SHAP analyzer
print("\nInitializing SHAP analyzer...")
try:
    best_model_key = f"{best_model_name}_{best_method_suffix}"
    best_feats = list(results[best_model_key]['selected_features'])
    X_train_sel = results[best_model_key]['x_train_selected']
    X_test_sel = results[best_model_key]['x_test_selected']
    shap_analyzer = SHAPAnalyzer(
        model=best_model,
        X_train=X_train_sel,
        X_test=X_test_sel,
        feature_names=list(best_feats),
        y_train=y_train,
        y_test=y_test
    )
    print("SHAP analyzer initialized successfully!")
except Exception as e:
    print(f"SHAP analyzer initialization failed: {e}")
    shap_analyzer = None

# Scatter plot
def scatter_plot(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    
    # Calculate evaluation metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ax.scatter(y_true, y_pred, alpha=0.8, color='#2E86AB', s=50, edgecolor='white', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='#E63946', lw=2, linestyle='-')
    
    ax.set_xlabel('Actual Value', fontsize=14)
    ax.set_ylabel('Predicted Value', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    textstr = f'R² = {r2:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', 
            fontweight='bold')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    
    plt.show()


# Output prediction results
def save_predictions(model, df, filename):
    """Predict and save using features from the best model."""
    best_key = f"{best_model_name}_{best_method_suffix}"
    used_features = list(results[best_key]['selected_features'])
    # Only use feature columns from training, raise error if missing
    missing = [c for c in used_features if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Prediction data missing required features for training: {missing}")
    pre_df_need = df[used_features]
    y_pred = model.predict(pre_df_need)
    predict_value = pd.DataFrame({"Predicted_Value": y_pred})
    output_df = pd.concat([df.drop(used_features, axis=1, errors='ignore'), predict_value], axis=1)
    output_df.insert(0, 'id', np.arange(1, 1 + len(output_df)))
    output_df.to_csv(filename, index=False)
    print(f"Output results saved to {filename}")

# ===================== Raster full-image prediction (GeoTIFF) =====================
def _safe_feature_filename(name):
    return ''.join([c if c.isalnum() or c in ('_', '-', '.') else '_' for c in str(name)])

def preprocess_rasters(feature_names, raster_dir, output_dir=None, pattern="{name}.tif", reference_feature=None):

    if rasterio is None:
        raise RuntimeError("rasterio is not installed, cannot perform raster preprocessing. Please install rasterio first")
    
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    
    if output_dir is None:
        output_dir = os.path.join(raster_dir, 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate features that need to be read from files (exclude Lat/Lon)
    file_features = [f for f in feature_names if f not in ['Lat', 'Lon']]
    
    if not file_features:
        print("Warning: No file features need preprocessing (only Lat/Lon included)")
        return output_dir
    
    # Determine reference raster
    if reference_feature is None:
        reference_feature = file_features[0]
    elif reference_feature not in file_features:
        print(f"Warning: Specified reference feature {reference_feature} is not in feature list, using first feature {file_features[0]}")
        reference_feature = file_features[0]
    
    # Read reference raster metadata
    ref_path = os.path.join(raster_dir, pattern.format(name=reference_feature))
    if not os.path.isfile(ref_path):
        alt = os.path.join(raster_dir, _safe_feature_filename(reference_feature))
        if os.path.isfile(alt):
            ref_path = alt
        else:
            raise FileNotFoundError(f"Reference raster file not found: {reference_feature} -> {ref_path}")
    
    print(f"\n=== Raster Preprocessing ===")
    print(f"Reference raster: {reference_feature}")
    print(f"Input directory: {raster_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of features to process: {len(file_features)}")
    
    with rasterio.open(ref_path) as ref_src:
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_width = ref_src.width
        ref_height = ref_src.height
        ref_bounds = ref_src.bounds
        ref_nodata = ref_src.nodata if ref_src.nodata is not None else -9999.0
        
        print(f"\nReference raster information:")
        print(f"  Size: {ref_width} × {ref_height}")
        print(f"  CRS: {ref_crs}")
        print(f"  Resolution: {ref_transform.a:.6f} × {-ref_transform.e:.6f}")
        print(f"  Bounds: {ref_bounds}")
    
    # Process each feature
    processed_count = 0
    for feat in file_features:
        input_path = os.path.join(raster_dir, pattern.format(name=feat))
        if not os.path.isfile(input_path):
            alt = os.path.join(raster_dir, _safe_feature_filename(feat))
            if os.path.isfile(alt):
                input_path = alt
            else:
                print(f"Warning: Raster file for feature {feat} not found, skipping")
                continue
        
        output_path = os.path.join(output_dir, pattern.format(name=feat))
        
        # Check if resampling is needed
        with rasterio.open(input_path) as src:
            needs_resampling = (
                src.width != ref_width or 
                src.height != ref_height or 
                src.transform != ref_transform or 
                src.crs != ref_crs
            )
            
            if not needs_resampling:
                # If already consistent, directly copy
                print(f"✓ {feat}: Already aligned, copying directly")
                import shutil
                shutil.copy2(input_path, output_path)
                processed_count += 1
                continue
            
            print(f"→ {feat}: Resampling...")
            print(f"    Original size: {src.width} × {src.height}, CRS: {src.crs}")
            
            # Read source data
            src_array = src.read(1)
            src_nodata = src.nodata if src.nodata is not None else ref_nodata
            
            # Create target array
            dst_array = np.empty((ref_height, ref_width), dtype=src.dtypes[0])
            
            # Perform resampling
            reproject(
                source=src_array,
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=ref_nodata,
                resampling=Resampling.bilinear  # Use bilinear interpolation
            )
            
            # Write resampled raster
            profile = {
                'driver': 'GTiff',
                'height': ref_height,
                'width': ref_width,
                'count': 1,
                'dtype': src.dtypes[0],
                'crs': ref_crs,
                'transform': ref_transform,
                'nodata': ref_nodata,
                'compress': 'lzw'  # Use LZW compression to reduce file size
            }
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(dst_array, 1)
            
            print(f"  ✓ Completed, output to: {output_path}")
            processed_count += 1
    
    print(f"\nPreprocessing completed! Successfully processed {processed_count}/{len(file_features)} raster files")
    print(f"Preprocessed rasters saved in: {output_dir}")
    return output_dir

def build_raster_stack(feature_names, raster_dir, pattern="{name}.tif"):
    if rasterio is None:
        raise RuntimeError("rasterio is not installed, cannot perform raster read/write. Please install rasterio first")

    if not os.path.isdir(raster_dir):
        raise FileNotFoundError(f"Raster directory does not exist: {raster_dir}")

    # Separate features that need to be read from files and features that need to calculate lat/lon
    spatial_features = ['Lat', 'Lon']
    file_features = []
    spatial_indices = {}  # Record Lat/Lon positions in feature list
    
    for idx, feat in enumerate(feature_names):
        if feat in spatial_features:
            spatial_indices[feat] = idx
        else:
            file_features.append((idx, feat))
    
    # If there are no file features, need to find a reference raster to get metadata
    if not file_features:
        # Try to find any tif file as reference
        tif_files = glob.glob(os.path.join(raster_dir, "*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No .tif files found in raster directory: {raster_dir}")
        reference_file = tif_files[0]
        print(f"Using reference raster file: {reference_file}")
    else:
        # Use first feature file as reference
        reference_feature = file_features[0][1]
        reference_file = os.path.join(raster_dir, pattern.format(name=reference_feature))
        if not os.path.isfile(reference_file):
            alt = os.path.join(raster_dir, _safe_feature_filename(reference_feature))
            if os.path.isfile(alt):
                reference_file = alt
            else:
                raise FileNotFoundError(f"Missing raster file for feature: {reference_feature} -> {reference_file}")

    # Get metadata from reference file
    with rasterio.open(reference_file) as src0:
        height, width = src0.height, src0.width
        transform = src0.transform
        crs = src0.crs
        nodata0 = src0.nodata

    # Initialize band list and mask list
    bands_dict = {} 
    masks = []

    # Read file features
    for idx, feat in file_features:
        file_path = os.path.join(raster_dir, pattern.format(name=feat))
        if not os.path.isfile(file_path):
            alt = os.path.join(raster_dir, _safe_feature_filename(feat))
            if os.path.isfile(alt):
                file_path = alt
            else:
                raise FileNotFoundError(f"Missing raster file for feature: {feat} -> {file_path}")
        
        with rasterio.open(file_path) as src:
            if src.count != 1:
                raise ValueError(f"File should be single band: {file_path}")
            if src.height != height or src.width != width:
                raise ValueError(f"Size mismatch: {file_path}")
            if src.transform != transform:
                raise ValueError(f"Affine transform mismatch (need to resample uniformly before use): {file_path}")
            if src.crs != crs:
                raise ValueError(f"CRS mismatch (need to reproject uniformly before use): {file_path}")
            arr = src.read(1).astype(np.float32)
            nd = src.nodata if src.nodata is not None else nodata0
            if nd is not None:
                mask = (arr != nd)
            else:
                mask = ~np.isnan(arr)
            bands_dict[idx] = arr
            masks.append(mask)

    # Calculate lat/lon features (if needed)
    if 'Lat' in spatial_indices or 'Lon' in spatial_indices:
        print("Detected Lat/Lon features, calculating lat/lon from raster geotransform information...")
        
        # Create pixel center coordinate grid
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Use affine transform to convert row/col coordinates to geographic coordinates
        # transform * (col, row) -> (x, y)
        xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
        xs = np.array(xs).reshape(height, width)
        ys = np.array(ys).reshape(height, width)
        
        # If CRS is not geographic coordinate system, need to convert to lat/lon
        if crs and not crs.is_geographic:
            try:
                from rasterio.warp import transform as warp_transform
                print(f"Raster CRS is projected coordinate system ({crs}), converting to lat/lon (EPSG:4326)...")
                lons, lats = warp_transform(crs, 'EPSG:4326', xs.flatten(), ys.flatten())
                lons = np.array(lons).reshape(height, width).astype(np.float32)
                lats = np.array(lats).reshape(height, width).astype(np.float32)
            except Exception as e:
                print(f"Coordinate conversion failed: {e}")
                print("Assuming raster is already in geographic coordinate system...")
                lons = xs.astype(np.float32)
                lats = ys.astype(np.float32)
        else:
            # Already in geographic coordinate system
            lons = xs.astype(np.float32)
            lats = ys.astype(np.float32)
        
        if 'Lon' in spatial_indices:
            bands_dict[spatial_indices['Lon']] = lons
            print(f"Lon range: {lons.min():.6f} ~ {lons.max():.6f}")
        
        if 'Lat' in spatial_indices:
            bands_dict[spatial_indices['Lat']] = lats
            print(f"Lat range: {lats.min():.6f} ~ {lats.max():.6f}")

    bands = []
    for i in range(len(feature_names)):
        if i in bands_dict:
            bands.append(bands_dict[i])
        else:
            raise ValueError(f"Feature index {i} ({feature_names[i]}) data not found")

    stack = np.stack(bands, axis=-1)  # (H, W, C)
    
    if masks:
        valid_mask = np.logical_and.reduce(masks)
    else:
        valid_mask = ~np.isnan(stack).any(axis=-1)
    
    meta = {
        'height': height,
        'width': width,
        'transform': transform,
        'crs': crs,
        'dtype': 'float32',
        'nodata': -9999.0
    }
    return stack, meta, valid_mask

def predict_raster_and_save(model, feature_names, raster_dir, out_tif_path, pattern="{name}.tif", batch_pixels=2_000_000):
   
    stack, meta, valid_mask = build_raster_stack(feature_names, raster_dir, pattern)
    h, w, c = stack.shape
    flat = stack.reshape(-1, c)
    valid = valid_mask.reshape(-1)

    y_pred = np.full((flat.shape[0],), meta['nodata'], dtype=np.float32)
    idxs = np.where(valid)[0]
    if idxs.size == 0:
        raise ValueError("Number of valid pixels is 0, please check NoData and mask conditions")

    # Batch prediction
    start = 0
    while start < idxs.size:
        end = min(start + batch_pixels, idxs.size)
        batch_idx = idxs[start:end]
        Xb = flat[batch_idx]
        yb = model.predict(Xb)
        y_pred[batch_idx] = yb.astype(np.float32)
        start = end

    out_arr = y_pred.reshape(h, w)

    if rasterio is None:
        raise RuntimeError("rasterio is not installed, cannot write GeoTIFF")

    os.makedirs(os.path.dirname(out_tif_path) or '.', exist_ok=True)
    profile = {
        'driver': 'GTiff',
        'height': meta['height'],
        'width': meta['width'],
        'count': 1,
        'dtype': meta['dtype'],
        'crs': meta['crs'],
        'transform': meta['transform'],
        'nodata': meta['nodata']
    }
    with rasterio.open(out_tif_path, 'w', **profile) as dst:
        dst.write(out_arr, 1)
    print(f"GeoTIFF saved: {out_tif_path}")


def calculate_single_feature_shap_raster(model, feature_names, target_feature, raster_dir, output_path, 
                                         pattern="{name}.tif", batch_pixels=100_000, sample_ratio=0.1, random_state=42):
   
    print(f"\n=== Calculate SHAP value spatial distribution map for single environmental factor '{target_feature}' ===")
    
    # Check if target feature exists
    if target_feature not in feature_names:
        raise ValueError(f"Target feature '{target_feature}' is not in feature list: {feature_names}")
    
    target_idx = feature_names.index(target_feature)
    
    # 1. Read raster data
    print("Step 1: Reading raster image data...")
    stack, meta, valid_mask = build_raster_stack(feature_names, raster_dir, pattern)
    h, w, c = stack.shape
    flat = stack.reshape(-1, c)
    valid = valid_mask.reshape(-1)
    
    idxs = np.where(valid)[0]
    if idxs.size == 0:
        raise ValueError("Number of valid pixels is 0, please check NoData and mask conditions")
    
    print(f"✓ Raster size: {h} x {w}, number of features: {c}, valid pixels: {idxs.size}")
    
    # 2. Initialize SHAP explainer (use sampled data to speed up)
    print(f"\nStep 2: Initializing SHAP explainer (sampling ratio: {sample_ratio*100}%)...")
    sample_size = max(100, int(idxs.size * sample_ratio))
    sample_size = min(sample_size, 10000)  # Use at most 10000 samples for initialization
    
    np.random.seed(random_state)
    sample_indices = np.random.choice(idxs, size=sample_size, replace=False)
    X_sample = flat[sample_indices]
    
    # Create SHAP explainer
    try:
        model_type = str(type(model)).lower()
        print(f"Model type: {type(model).__name__}")
        
        if 'lgb' in model_type or 'randomforest' in model_type or \
           'extratrees' in model_type or 'gradientboosting' in model_type:
            explainer = shap.Explainer(model)
            print("✓ Using TreeExplainer")
        else:
            # For other models, use KernelExplainer (slower but universal)
            explainer = shap.KernelExplainer(model.predict, X_sample[:100])
            print("✓ Using KernelExplainer")
    except Exception as e:
        print(f"SHAP explainer initialization failed: {e}")
        raise
    
    # 3. Initialize SHAP value array (only for target feature)
    print(f"\nStep 3: Initializing SHAP value array (feature: {target_feature})...")
    shap_array = np.full((flat.shape[0],), meta['nodata'], dtype=np.float32)
    
    # 4. Calculate SHAP values in batches
    print(f"\nStep 4: Calculating SHAP values in batches (batch size: {batch_pixels})...")
    total_batches = int(np.ceil(idxs.size / batch_pixels))
    
    start = 0
    batch_count = 0
    while start < idxs.size:
        end = min(start + batch_pixels, idxs.size)
        batch_idx = idxs[start:end]
        Xb = flat[batch_idx]
        
        batch_count += 1
        print(f"  Processing batch {batch_count}/{total_batches} ({end}/{idxs.size} pixels, {end/idxs.size*100:.1f}%)...", end='\r')
        
        try:
            # Calculate SHAP values for this batch
            shap_values = explainer(Xb)
            
            # Extract SHAP value array
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values
            else:
                shap_vals = shap_values
            
            # Only save SHAP values for target feature
            shap_array[batch_idx] = shap_vals[:, target_idx].astype(np.float32)
                
        except Exception as e:
            print(f"\nWarning: Batch {batch_count} SHAP value calculation failed: {e}")
            # Failed batches remain as nodata
            continue
        
        start = end
    
    print(f"\n✓ SHAP value calculation completed!")
    
    # 5. Save SHAP value raster
    print(f"\nStep 5: Saving SHAP value raster...")
    if rasterio is None:
        raise RuntimeError("rasterio is not installed, cannot write GeoTIFF")
    
    # Create output directory (if needed)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Reshape to 2D array
    shap_raster = shap_array.reshape(h, w)
    
    # Write GeoTIFF
    profile = {
        'driver': 'GTiff',
        'height': meta['height'],
        'width': meta['width'],
        'count': 1,
        'dtype': 'float32',
        'crs': meta['crs'],
        'transform': meta['transform'],
        'nodata': meta['nodata'],
        'compress': 'lzw'  # Add compression to reduce file size
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(shap_raster, 1)
    
    print(f"  ✓ {target_feature}: {output_path}")
    
    # 6. Generate statistical summary
    print("\nStep 6: Generating SHAP value statistical summary...")
    print("\n" + "="*70)
    print(f"Feature '{target_feature}' SHAP value spatial distribution statistics")
    print("="*70)
    
    # Calculate statistics
    valid_shap = shap_array[idxs]
    mean_abs_shap = np.abs(valid_shap).mean()
    mean_shap = valid_shap.mean()
    std_shap = valid_shap.std()
    min_shap = valid_shap.min()
    max_shap = valid_shap.max()
    
    print(f"\n{'Statistic':<20} {'Value':<15}")
    print("-"*35)
    print(f"{'Mean |SHAP|':<20} {mean_abs_shap:<15.4f}")
    print(f"{'Mean SHAP':<20} {mean_shap:<15.4f}")
    print(f"{'Std Dev':<20} {std_shap:<15.4f}")
    print(f"{'Min':<20} {min_shap:<15.4f}")
    print(f"{'Max':<20} {max_shap:<15.4f}")
    print(f"{'Valid Pixels':<20} {len(valid_shap):<15}")
    
    # Calculate positive/negative impact ratios
    positive_ratio = (valid_shap > 0).sum() / len(valid_shap) * 100
    negative_ratio = (valid_shap < 0).sum() / len(valid_shap) * 100
    neutral_ratio = (valid_shap == 0).sum() / len(valid_shap) * 100
    
    print(f"\nImpact direction analysis:")
    print("-"*35)
    print(f"Positive impact pixel ratio: {positive_ratio:.2f}%")
    print(f"Negative impact pixel ratio: {negative_ratio:.2f}%")
    print(f"Neutral pixel ratio: {neutral_ratio:.2f}%")
    
    print("\n" + "="*70)
    print(f"✓ SHAP value raster saved to: {output_path}")
    print("="*70)
    
    return output_path


def calculate_shap_raster_and_save(model, feature_names, raster_dir, output_dir, pattern="{name}.tif", 
                                    batch_pixels=100_000, sample_ratio=0.1, random_state=42):
   
    print("\n=== Starting calculation of SHAP value spatial distribution map ===")
    
    # 1. Read raster data
    print("Step 1: Reading raster image data...")
    stack, meta, valid_mask = build_raster_stack(feature_names, raster_dir, pattern)
    h, w, c = stack.shape
    flat = stack.reshape(-1, c)
    valid = valid_mask.reshape(-1)
    
    idxs = np.where(valid)[0]
    if idxs.size == 0:
        raise ValueError("Number of valid pixels is 0, please check NoData and mask conditions")
    
    print(f"✓ Raster size: {h} x {w}, number of features: {c}, valid pixels: {idxs.size}")
    
    # 2. Initialize SHAP explainer (use sampled data to speed up)
    print(f"\nStep 2: Initializing SHAP explainer (sampling ratio: {sample_ratio*100}%)...")
    sample_size = max(100, int(idxs.size * sample_ratio))
    sample_size = min(sample_size, 10000)  # Use at most 10000 samples for initialization
    
    np.random.seed(random_state)
    sample_indices = np.random.choice(idxs, size=sample_size, replace=False)
    X_sample = flat[sample_indices]
    
    # Create SHAP explainer
    try:
        model_type = str(type(model)).lower()
        print(f"Model type: {type(model).__name__}")
        
        if 'lgb' in model_type or 'randomforest' in model_type or \
           'extratrees' in model_type or 'gradientboosting' in model_type:
            explainer = shap.Explainer(model)
            print("✓ Using TreeExplainer")
        else:
            # For other models, use KernelExplainer (slower but universal)
            explainer = shap.KernelExplainer(model.predict, X_sample[:100])
            print("✓ Using KernelExplainer")
    except Exception as e:
        print(f"SHAP explainer initialization failed: {e}")
        raise
    
    # 3. Initialize SHAP value arrays
    print("\nStep 3: Initializing SHAP value arrays...")
    num_features = len(feature_names)
    shap_arrays = {}
    for i, feat in enumerate(feature_names):
        shap_arrays[feat] = np.full((flat.shape[0],), meta['nodata'], dtype=np.float32)
    
    # 4. Calculate SHAP values in batches
    print(f"\nStep 4: Calculating SHAP values in batches (batch size: {batch_pixels})...")
    total_batches = int(np.ceil(idxs.size / batch_pixels))
    
    start = 0
    batch_count = 0
    while start < idxs.size:
        end = min(start + batch_pixels, idxs.size)
        batch_idx = idxs[start:end]
        Xb = flat[batch_idx]
        
        batch_count += 1
        print(f"  Processing batch {batch_count}/{total_batches} ({end}/{idxs.size} pixels, {end/idxs.size*100:.1f}%)...", end='\r')
        
        try:
            # Calculate SHAP values for this batch
            shap_values = explainer(Xb)
            
            # Extract SHAP value array
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values
            else:
                shap_vals = shap_values
            
            # Store SHAP values to corresponding arrays
            for i, feat in enumerate(feature_names):
                shap_arrays[feat][batch_idx] = shap_vals[:, i].astype(np.float32)
                
        except Exception as e:
            print(f"\nWarning: Batch {batch_count} SHAP value calculation failed: {e}")
            # Failed batches remain as nodata
            continue
        
        start = end
    
    print(f"\n✓ SHAP value calculation completed!")
    
    # 5. Save SHAP value rasters
    print(f"\nStep 5: Saving SHAP value rasters to {output_dir}...")
    if rasterio is None:
        raise RuntimeError("rasterio is not installed, cannot write GeoTIFF")
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = {}
    for feat in feature_names:
        shap_raster = shap_arrays[feat].reshape(h, w)
        output_path = os.path.join(output_dir, f"SHAP_{feat}.tif")
        
        # Write GeoTIFF
        profile = {
            'driver': 'GTiff',
            'height': meta['height'],
            'width': meta['width'],
            'count': 1,
            'dtype': 'float32',
            'crs': meta['crs'],
            'transform': meta['transform'],
            'nodata': meta['nodata'],
            'compress': 'lzw'  
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(shap_raster, 1)
        
        output_files[feat] = output_path
        print(f"  ✓ {feat}: {output_path}")
    
    print("\nStep 6: Generating SHAP value statistical summary...")
    print("\n" + "="*70)
    print("SHAP Value Spatial Distribution Statistical Summary")
    print("="*70)
    
    # Calculate SHAP value statistics for each feature
    shap_stats = []
    for feat in feature_names:
        valid_shap = shap_arrays[feat][idxs]
        mean_abs_shap = np.abs(valid_shap).mean()
        mean_shap = valid_shap.mean()
        std_shap = valid_shap.std()
        min_shap = valid_shap.min()
        max_shap = valid_shap.max()
        
        shap_stats.append({
            'Feature': feat,
            'Mean_Abs_SHAP': mean_abs_shap,
            'Mean_SHAP': mean_shap,
            'Std_SHAP': std_shap,
            'Min_SHAP': min_shap,
            'Max_SHAP': max_shap
        })
    
    # Sort by mean absolute SHAP value
    shap_stats_df = pd.DataFrame(shap_stats).sort_values('Mean_Abs_SHAP', ascending=False)
    
    print(f"\n{'Feature':<15} {'Mean |SHAP|':<12} {'Mean SHAP':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-"*70)
    for _, row in shap_stats_df.iterrows():
        print(f"{row['Feature']:<15} {row['Mean_Abs_SHAP']:<12.4f} {row['Mean_SHAP']:<12.4f} "
              f"{row['Std_SHAP']:<12.4f} {row['Min_SHAP']:<12.4f} {row['Max_SHAP']:<12.4f}")
    
    # Save statistical summary to CSV
    stats_path = os.path.join(output_dir, "SHAP_statistics.csv")
    shap_stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Statistical summary saved: {stats_path}")
    
    print("\n" + "="*70)
    print(f"✓ All SHAP value rasters saved to: {output_dir}")
    print(f"✓ Generated {len(output_files)} SHAP value raster files")
    print("="*70)
    
    return output_files

def get_data_mode_choice():
   
    print("\nPlease select dataset:")
    print("1. Test Set")
    print("2. Train Set")
    print("3. All Data")
    
    choice = input("Please select (1/2/3, default 1-Test Set): ").strip()
    
    if choice == '2':
        data_mode = 'train'
        data_name_cn = "Train Set"
        data_name_en = "Train Set"
    elif choice == '3':
        data_mode = 'all'
        data_name_cn = "All Data"
        data_name_en = "All Data"
    else:
        data_mode = 'test'
        data_name_cn = "Test Set"
        data_name_en = "Test Set"
    
    return data_mode, data_name_cn, data_name_en

# Menu interaction
while True:
    print("\n-----------------Menu-----------------")
    print("1->Best model accuracy  2->Best hyperparameters  3->Test set accuracy  4->Train set accuracy  5->Test set actual-predicted scatter plot  6->Train set actual-predicted scatter plot  7->Output best model prediction results  8->All models comparison  9->Select model output scatter plot  10->Output optimal hyperparameters for each model")
    print("11->SHAP overall importance ranking plot  12->SHAP Summary Plot  13->PDP partial dependence curve  14->Correlation heatmap for all original data (with significance stars)  15->Full-image prediction on raster images using best model and output GeoTIFF")
    print("16->Raster image preprocessing (unify size and projection)  17->Select model for full-image prediction on raster images  18->Calculate SHAP value spatial distribution map for environmental factors (raster output)  19->SHAP feature importance ranking plot (with contribution rate pie chart)  20->Calculate SHAP value spatial distribution map for single environmental factor  21->Batch output all PDP plots  0->Exit program")
    choose = input("Please enter option: ").strip()
    if choose == '1':
        best_model_key = f"{best_model_name}_{best_method_suffix}"
        print(f"Best model: {best_model_name} (method: {best_method_suffix})")
        print(f"Test set: R2: {results[best_model_key]['test']['r2']:.4f}, RMSE: {results[best_model_key]['test']['rmse']:.4f}, MAE: {results[best_model_key]['test']['mae']:.4f}")
        print("Train pred range:", float(best_train_pred.min()), float(best_train_pred.max()))
        print("Test  pred range:", float(best_test_pred.min()), float(best_test_pred.max()))
        print("Target O range:", float(y.min()), float(y.max()))

    elif choose == '2':
        print(f"Best hyperparameters: {best_params}")
    elif choose == '3':
        best_model_key = f"{best_model_name}_{best_method_suffix}"
        print(f"Test set accuracy: R2: {results[best_model_key]['test']['r2']:.4f}, RMSE: {results[best_model_key]['test']['rmse']:.4f}, MAE: {results[best_model_key]['test']['mae']:.4f}")
    elif choose == '4':
        best_model_key = f"{best_model_name}_{best_method_suffix}"
        print(f"Train set accuracy: R2: {results[best_model_key]['train']['r2']:.4f}, RMSE: {results[best_model_key]['train']['rmse']:.4f}, MAE: {results[best_model_key]['train']['mae']:.4f}")
    elif choose == '5':
        # Ask whether to save image
        save_choice = input("Save image? (y/n, default n): ").strip().lower()
        save_path = None
        if save_choice == 'y':
            save_path = f"figures/scatter_plot_{best_model_name}_test.png"
            # Ensure figures directory exists
            os.makedirs('figures', exist_ok=True)
        scatter_plot(y_test, best_test_pred, title=f"{best_model_name}-Test", save_path=save_path)
    elif choose == '6':
        # Ask whether to save image
        save_choice = input("Save image? (y/n, default n): ").strip().lower()
        save_path = None
        if save_choice == 'y':
            save_path = f"figures/scatter_plot_{best_model_name}_train.png"
            # Ensure figures directory exists
            os.makedirs('figures', exist_ok=True)
        scatter_plot(y_train, best_train_pred, title=f"{best_model_name}-Train", save_path=save_path)
    elif choose == '7':
        pre_path = input("Please enter CSV file path for prediction data (pre_df): ").strip()
        try:
            pre_df = pd.read_csv(pre_path)
            save_predictions(best_model, pre_df, "predict_result.csv")
        except Exception as e:
            print(f"Error reading or predicting: {e}")
    elif choose == '8':
        print("All models comparison:")
        for name, res in results.items():
            print(f"Model: {name}")
            print(f"  Test set: R2: {res['test']['r2']:.4f}, RMSE: {res['test']['rmse']:.4f}, MAE: {res['test']['mae']:.4f}")
            print(f"  Train set: R2: {res['train']['r2']:.4f}, RMSE: {res['train']['rmse']:.4f}, MAE: {res['train']['mae']:.4f}")
    elif choose == '9':
        print("Available models:")
        for name in results.keys():
            print(name)
        model_name = input("Please enter model name to output scatter plot: ").strip()
        if model_name in results:
            # Ask whether to save image
            save_choice = input("Save image? (y/n, default n): ").strip().lower()
            save_path_test = None
            save_path_train = None
            if save_choice == 'y':
                save_path_test = f"figures/scatter_plot_{model_name.split('_')[0]}_test.png"
                save_path_train = f"figures/scatter_plot_{model_name.split('_')[0]}_train.png"
                # Ensure figures directory exists
                os.makedirs('figures', exist_ok=True)
            print("Test set actual-predicted scatter plot:")
            scatter_plot(y_test, results[model_name]['test_pred'], title=f"{model_name.split('_')[0]}-Test", save_path=save_path_test)
            print("Train set actual-predicted scatter plot:")
            scatter_plot(y_train, results[model_name]['train_pred'], title=f"{model_name.split('_')[0]}-Train", save_path=save_path_train)
        else:
            print("Model name does not exist!")
    elif choose == '10':
        print("Optimal hyperparameters for each model:")
        for name, res in results.items():
            print(f"{name}: {res['params'] if res['params'] is not None else 'No hyperparameters/Grid search not performed'}")
    
    elif choose == '11':
        if shap_analyzer is None:
            print("SHAP analyzer not initialized, cannot perform SHAP analysis")
        else:
            print("Preparing to generate overall importance ranking plot...")
            try:
                sample_size = input("Please enter sample size to speed up calculation (press Enter to use all data): ").strip()
                sample_size = int(sample_size) if sample_size else None
                
                # Select dataset
                data_mode, data_name_cn, data_name_en = get_data_mode_choice()
                print(f"Using {data_name_cn} data to calculate SHAP values...")
                
                shap_values = shap_analyzer.calculate_shap_values(sample_size=sample_size, data_mode=data_mode)
                if shap_values is not None:
                    top_n = int(input("Please enter number of top N important features to display (default 20): ").strip() or '20')
                    # Ask whether to save image
                    save_choice = input("Save image? (y/n, default n): ").strip().lower()
                    save_path = None
                    if save_choice == 'y':
                        save_path = f"figures/shap_importance_{data_name_en.replace(' ', '_')}_top{top_n}.png"
                        # Ensure figures directory exists
                        os.makedirs('figures', exist_ok=True)
                    shap_analyzer.plot_shap_importance(shap_values, top_n=top_n, 
                                                     title=f"Soil Organic Matter Prediction - SHAP Feature Importance Ranking ({data_name_en})", save_path=save_path)
                else:
                    print("SHAP value calculation failed")
            except Exception as e:
                print(f"SHAP overall importance ranking plot generation failed: {e}")
    
    elif choose == '12':
        if shap_analyzer is None:
            print("SHAP analyzer not initialized, cannot perform SHAP analysis")
        else:
            print("Generating SHAP Summary Plot...")
            try:
                sample_size = input("Please enter sample size to speed up calculation (press Enter to use all data): ").strip()
                sample_size = int(sample_size) if sample_size else None
                
                # Select dataset
                data_mode, data_name_cn, data_name_en = get_data_mode_choice()
                print(f"Using {data_name_cn} data to calculate SHAP values...")
                
                shap_values = shap_analyzer.calculate_shap_values(sample_size=sample_size, data_mode=data_mode)
                if shap_values is not None:
                    max_display = int(input("Please enter maximum number of features to display (default 20): ").strip() or '20')
                    # Ask whether to save image
                    save_choice = input("Save image? (y/n, default n): ").strip().lower()
                    save_path = None
                    if save_choice == 'y':
                        save_path = f"figures/shap_summary_max{max_display}.png"
                        # Ensure figures directory exists
                        os.makedirs('figures', exist_ok=True)
                    shap_analyzer.plot_shap_summary(shap_values, max_display=max_display,
                                                   title="Soil Organic Matter Prediction - SHAP Summary Plot", save_path=save_path)
                else:
                    print("SHAP value calculation failed")
            except Exception as e:
                print(f"SHAP Summary Plot generation failed: {e}")
    
    elif choose == '13':
        if shap_analyzer is None:
            print("SHAP analyzer not initialized, cannot perform PDP plotting")
        else:
            try:
                print("Available features:")
                for i, feature in enumerate(shap_analyzer.feature_names, 1):
                    print(f"{i:2d}. {feature}")
                feature_name = input("Please enter feature name to plot PDP: ").strip()
                if feature_name not in shap_analyzer.feature_names:
                    print("Feature name does not exist")
                else:
                    grid_res = int(input("Please enter grid resolution (default 50): ").strip() or '50')
                    
                    # Dataset selection
                    data_mode, data_name_cn, data_name_en = get_data_mode_choice()
                    
                    n_ice = int(input("Number of ICE curves to overlay (default 0): ").strip() or '0')
                    # Ask whether to save image
                    save_choice = input("Save image? (y/n, default n): ").strip().lower()
                    save_path = None
                    if save_choice == 'y':
                        save_path = f"figures/pdp_{feature_name}_{data_name_en.replace(' ', '_')}_ice{n_ice}.png"
                        # Ensure figures directory exists
                        os.makedirs('figures', exist_ok=True)
                    
                    shap_analyzer.plot_partial_dependence(
                        feature_name,
                        grid_resolution=grid_res,
                        n_ice_curves=n_ice,
                        title=f"Partial Dependence Plot - {feature_name} ({data_name_en})",
                        save_path=save_path,
                        data_mode=data_mode
                    )
            except Exception as e:
                print(f"PDP plotting failed: {e}")

    elif choose == '14':
        print("Generating correlation heatmap for all original data (with significance stars)...")
        try:
            # Use all original data (including target variable O)
            all_data = df
            
            # Ask user to select correlation calculation method
            method_choice = input("Please select correlation calculation method (1: Pearson, 2: Spearman, default 1): ").strip()
            method = 'pearson' if method_choice != '2' else 'spearman'
            
            # Ask star position
            print("Please select significance star position:")
            print("1: Below numbers (default)")
            print("2: Above numbers")
            print("3: Right of numbers")
            print("4: Left of numbers")
            position_choice = input("Please enter choice (1-4, default 1): ").strip()
            position_map = {'1': 'below', '2': 'above', '3': 'right', '4': 'left'}
            star_position = position_map.get(position_choice, 'below')
            
            # Ask font size settings
            print("Please select font size settings:")
            print("1: Small font (number 8, star 6) - suitable for large matrices")
            print("2: Medium font (number 10, star 8) - default")
            print("3: Large font (number 12, star 10) - suitable for small matrices")
            font_choice = input("Please enter choice (1-3, default 2): ").strip()
            font_settings = {
                '1': (8, 6),   # Small font
                '2': (10, 8),  # Medium font
                '3': (12, 10)  # Large font
            }
            font_size, star_font_size = font_settings.get(font_choice, (10, 8))
            
            # Ask whether to save image
            save_choice = input("Save image? (y/n, default y): ").strip().lower()
            save_path = None
            if save_choice != 'n':
                save_path = f"figures/corr_heatmap_all_features_with_significance_{method}_{star_position}.png"
                # Ensure figures directory exists
                os.makedirs('figures', exist_ok=True)
            
            # Generate correlation heatmap
            corr_matrix, significance_matrix = correlation_heatmap_with_significance(
                all_data, 
                method=method,
                title=f'Correlation Heatmap of All Features ({method.capitalize()})',
                figsize=(14, 12),
                save_path=save_path,
                star_position=star_position,
                font_size=font_size,
                star_font_size=star_font_size
            )
            
            print(f"\nCorrelation matrix calculation completed, total {len(all_data.columns)} variables")
            print(f"Significance star position: {star_position}")
            print(f"Font settings: number {font_size}, star {star_font_size}")
            print("Heatmap generated and displayed")
            
        except Exception as e:
            print(f"Correlation heatmap generation failed: {e}")

    elif choose == '15':
        try:
            if rasterio is None:
                print("rasterio is not installed, cannot perform raster prediction. Please install: pip install rasterio")
            else:
                best_key = f"{best_model_name}_{best_method_suffix}"
                used_features = list(results[best_key]['selected_features'])
                print("\n=== Raster Full-Image Prediction ===")
                print("Raster files will be read in the same order as features during training.")
                print("Assume each feature has one single-band tif file, filename pattern can use {name} placeholder, e.g.: {name}.tif")
                print("\nSpecial notes:")
                print("- If features include Lat (latitude) or Lon (longitude), the system will automatically calculate from raster geotransform information")
                print("- You don't need to prepare Lat.tif or Lon.tif files")
                print("- If raster is in projected coordinate system, the system will automatically convert to geographic coordinate system (WGS84)")
                print(f"\nFeatures used for this prediction: {', '.join(used_features)}")
                
                # Check if Lat/Lon is included
                has_spatial = any(feat in ['Lat', 'Lon'] for feat in used_features)
                if has_spatial:
                    spatial_feats = [f for f in used_features if f in ['Lat', 'Lon']]
                    print(f"✓ Detected spatial features: {', '.join(spatial_feats)} - will be automatically calculated")
                
                raster_dir = input("\nPlease enter raster directory (e.g. D:/rasters): ").strip()
                pattern = input("Please enter filename pattern (default {name}.tif): ").strip() or "{name}.tif"
                
                # Ask whether preprocessing is needed
                preprocess_choice = input("\nDo you need to preprocess raster images to unify size and projection? (y/n, default y): ").strip().lower()
                if preprocess_choice != 'n':
                    print("\n--- Raster Preprocessing Options ---")
                    print("Preprocessing will unify all rasters' size, resolution, projection and extent")
                    
                    # Select reference raster
                    file_features = [f for f in used_features if f not in ['Lat', 'Lon']]
                    if file_features:
                        print(f"\nAvailable reference raster features:")
                        for i, feat in enumerate(file_features, 1):
                            print(f"{i}. {feat}")
                        ref_choice = input(f"Please select reference raster (1-{len(file_features)}, default 1): ").strip()
                        if ref_choice.isdigit() and 1 <= int(ref_choice) <= len(file_features):
                            reference_feature = file_features[int(ref_choice) - 1]
                        else:
                            reference_feature = file_features[0]
                        print(f"Selected reference raster: {reference_feature}")
                    else:
                        reference_feature = None
                    
                    # Execute preprocessing
                    try:
                        preprocessed_dir = preprocess_rasters(
                            used_features, 
                            raster_dir, 
                            output_dir=None,  # Default: create preprocessed subdirectory
                            pattern=pattern,
                            reference_feature=reference_feature
                        )
                        # Use preprocessed directory for prediction
                        raster_dir = preprocessed_dir
                        print(f"\nWill use preprocessed rasters for prediction: {raster_dir}")
                    except Exception as e:
                        print(f"Preprocessing failed: {e}")
                        retry_choice = input("Continue using original rasters for prediction? (y/n, default n): ").strip().lower()
                        if retry_choice != 'y':
                            raise
                
                out_tif = input("\nPlease enter output GeoTIFF path (e.g. result.tif): ").strip() or "predict_map.tif"
                
                # Validate output path
                # If user only entered directory, automatically add default filename
                if os.path.isdir(out_tif):
                    out_tif = os.path.join(out_tif, "predict_map.tif")
                    print(f"Detected input is a directory, automatically set output file to: {out_tif}")
                # If path doesn't contain extension, add .tif
                elif not out_tif.lower().endswith(('.tif', '.tiff')):
                    out_tif = out_tif + ".tif"
                    print(f"Automatically added .tif extension: {out_tif}")
                
                # Ensure output directory exists
                out_dir = os.path.dirname(out_tif)
                if out_dir and not os.path.exists(out_dir):
                    try:
                        os.makedirs(out_dir, exist_ok=True)
                        print(f"Created output directory: {out_dir}")
                    except Exception as e:
                        print(f"Warning: Unable to create output directory {out_dir}: {e}")
                
                batch = input("Batch pixel count (default 2000000): ").strip()
                batch = int(batch) if batch else 2_000_000
                
                print(f"\nStarting prediction...")
                print(f"Output file: {out_tif}")
                predict_raster_and_save(best_model, used_features, raster_dir, out_tif, pattern=pattern, batch_pixels=batch)
        except Exception as e:
            print(f"Full-image prediction failed: {e}")
            import traceback
            traceback.print_exc()

    elif choose == '16':
        try:
            if rasterio is None:
                print("rasterio is not installed, cannot perform raster preprocessing. Please install: pip install rasterio")
            else:
                print("\n=== Raster Image Preprocessing ===")
                print("This function will unify all rasters' size, resolution, projection and extent")
                print("Use resampling technology to align all rasters to reference raster")
                
                # Ask whether to use model features or custom feature list
                use_model_features = input("\nUse best model's features? (y/n, default y): ").strip().lower()
                if use_model_features != 'n':
                    best_key = f"{best_model_name}_{best_method_suffix}"
                    feature_list = list(results[best_key]['selected_features'])
                    print(f"Will use best model's features ({len(feature_list)}): {', '.join(feature_list)}")
                else:
                    # Custom feature list
                    features_input = input("Please enter feature name list (comma-separated): ").strip()
                    feature_list = [f.strip() for f in features_input.split(',') if f.strip()]
                    if not feature_list:
                        print("Feature list is empty, operation cancelled")
                        continue
                    print(f"Will process the following features ({len(feature_list)}): {', '.join(feature_list)}")
                
                raster_dir = input("\nPlease enter input raster directory: ").strip()
                if not raster_dir or not os.path.isdir(raster_dir):
                    print("Directory does not exist or is invalid")
                    continue
                
                pattern = input("Please enter filename pattern (default {name}.tif): ").strip() or "{name}.tif"
                
                output_dir = input("Please enter output directory (default: create preprocessed subdirectory in input directory): ").strip()
                if not output_dir:
                    output_dir = None  # Use default value
                
                # Select reference raster
                file_features = [f for f in feature_list if f not in ['Lat', 'Lon']]
                reference_feature = None
                if file_features:
                    print(f"\nAvailable reference raster features:")
                    for i, feat in enumerate(file_features, 1):
                        print(f"{i}. {feat}")
                    ref_choice = input(f"Please select reference raster (1-{len(file_features)}, default 1): ").strip()
                    if ref_choice.isdigit() and 1 <= int(ref_choice) <= len(file_features):
                        reference_feature = file_features[int(ref_choice) - 1]
                    else:
                        reference_feature = file_features[0]
                    print(f"Selected reference raster: {reference_feature}")
                
                # Execute preprocessing
                print("\nStarting preprocessing...")
                preprocessed_dir = preprocess_rasters(
                    feature_list,
                    raster_dir,
                    output_dir=output_dir,
                    pattern=pattern,
                    reference_feature=reference_feature
                )
                print(f"\n✓ Preprocessing completed!")
                print(f"Preprocessed rasters saved in: {preprocessed_dir}")
                
        except Exception as e:
            print(f"Raster preprocessing failed: {e}")
            import traceback
            traceback.print_exc()

    elif choose == '17':
        try:
            if rasterio is None:
                print("rasterio is not installed, cannot perform raster prediction. Please install: pip install rasterio")
            else:
                print("\n=== Select Model for Raster Prediction ===")
                print("Available model list:")
                model_list = list(results.keys())
                for i, model_name in enumerate(model_list, 1):
                    model_info = results[model_name]
                    print(f"{i}. {model_name} - R²: {model_info['test']['r2']:.4f}, RMSE: {model_info['test']['rmse']:.4f}, MAE: {model_info['test']['mae']:.4f}")
                
                model_choice = input(f"\nPlease select model (1-{len(model_list)}): ").strip()
                if not model_choice.isdigit() or not (1 <= int(model_choice) <= len(model_list)):
                    print("Invalid choice")
                    continue
                
                selected_model_key = model_list[int(model_choice) - 1]
                selected_model = results[selected_model_key]['model']
                used_features = list(results[selected_model_key]['selected_features'])
                
                print(f"\nSelected model: {selected_model_key}")
                print(f"Model accuracy - R²: {results[selected_model_key]['test']['r2']:.4f}, RMSE: {results[selected_model_key]['test']['rmse']:.4f}, MAE: {results[selected_model_key]['test']['mae']:.4f}")
                print(f"Features used ({len(used_features)}): {', '.join(used_features)}")
                
                # Check if Lat/Lon is included
                has_spatial = any(feat in ['Lat', 'Lon'] for feat in used_features)
                if has_spatial:
                    spatial_feats = [f for f in used_features if f in ['Lat', 'Lon']]
                    print(f"✓ Detected spatial features: {', '.join(spatial_feats)} - will be automatically calculated")
                
                raster_dir = input("\nPlease enter raster directory: ").strip()
                pattern = input("Please enter filename pattern (default {name}.tif): ").strip() or "{name}.tif"
                
                # Ask whether preprocessing is needed
                preprocess_choice = input("\nDo you need to preprocess raster images to unify size and projection? (y/n, default y): ").strip().lower()
                if preprocess_choice != 'n':
                    print("\n--- Raster Preprocessing Options ---")
                    print("Preprocessing will unify all rasters' size, resolution, projection and extent")
                    
                    # Select reference raster
                    file_features = [f for f in used_features if f not in ['Lat', 'Lon']]
                    if file_features:
                        print(f"\nAvailable reference raster features:")
                        for i, feat in enumerate(file_features, 1):
                            print(f"{i}. {feat}")
                        ref_choice = input(f"Please select reference raster (1-{len(file_features)}, default 1): ").strip()
                        if ref_choice.isdigit() and 1 <= int(ref_choice) <= len(file_features):
                            reference_feature = file_features[int(ref_choice) - 1]
                        else:
                            reference_feature = file_features[0]
                        print(f"Selected reference raster: {reference_feature}")
                    else:
                        reference_feature = None
                    
                    # Execute preprocessing
                    try:
                        preprocessed_dir = preprocess_rasters(
                            used_features, 
                            raster_dir, 
                            output_dir=None,  # Default: create preprocessed subdirectory
                            pattern=pattern,
                            reference_feature=reference_feature
                        )
                        # Use preprocessed directory for prediction
                        raster_dir = preprocessed_dir
                        print(f"\nWill use preprocessed rasters for prediction: {raster_dir}")
                    except Exception as e:
                        print(f"Preprocessing failed: {e}")
                        retry_choice = input("Continue using original rasters for prediction? (y/n, default n): ").strip().lower()
                        if retry_choice != 'y':
                            raise
                
                out_tif = input("\nPlease enter output GeoTIFF path (e.g. result.tif): ").strip() or f"predict_{selected_model_key}.tif"
                
                # Validate output path
                if os.path.isdir(out_tif):
                    out_tif = os.path.join(out_tif, f"predict_{selected_model_key}.tif")
                    print(f"Detected input is a directory, automatically set output file to: {out_tif}")
                elif not out_tif.lower().endswith(('.tif', '.tiff')):
                    out_tif = out_tif + ".tif"
                    print(f"Automatically added .tif extension: {out_tif}")
                
                # Ensure output directory exists
                out_dir = os.path.dirname(out_tif)
                if out_dir and not os.path.exists(out_dir):
                    try:
                        os.makedirs(out_dir, exist_ok=True)
                        print(f"Created output directory: {out_dir}")
                    except Exception as e:
                        print(f"Warning: Unable to create output directory {out_dir}: {e}")
                
                batch = input("Batch pixel count (default 2000000): ").strip()
                batch = int(batch) if batch else 2_000_000
                
                print(f"\nStarting prediction...")
                print(f"Using model: {selected_model_key}")
                print(f"Output file: {out_tif}")
                predict_raster_and_save(selected_model, used_features, raster_dir, out_tif, pattern=pattern, batch_pixels=batch)
                
        except Exception as e:
            print(f"Raster prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif choose == '18':
        try:
            if rasterio is None:
                print("rasterio is not installed, cannot perform raster processing. Please install: pip install rasterio")
            else:
                print("\n=== Calculate SHAP Value Spatial Distribution Map for Environmental Factors ===")
                print("This function will generate independent SHAP value raster maps for each environmental factor")
                print("SHAP values reflect the contribution of each factor to prediction results at each pixel")
                
                # Select model
                print("\nAvailable model list:")
                model_list = list(results.keys())
                for i, model_name in enumerate(model_list, 1):
                    model_info = results[model_name]
                    print(f"{i}. {model_name} - R²: {model_info['test']['r2']:.4f}, RMSE: {model_info['test']['rmse']:.4f}, MAE: {model_info['test']['mae']:.4f}")
                
                model_choice = input(f"\nPlease select model (1-{len(model_list)}, default 1-best model): ").strip()
                if model_choice and model_choice.isdigit() and (1 <= int(model_choice) <= len(model_list)):
                    selected_model_key = model_list[int(model_choice) - 1]
                else:
                    selected_model_key = f"{best_model_name}_{best_method_suffix}"
                    print(f"Using best model: {selected_model_key}")
                
                selected_model = results[selected_model_key]['model']
                used_features = list(results[selected_model_key]['selected_features'])
                
                print(f"\nSelected model: {selected_model_key}")
                print(f"Model accuracy - R²: {results[selected_model_key]['test']['r2']:.4f}, RMSE: {results[selected_model_key]['test']['rmse']:.4f}, MAE: {results[selected_model_key]['test']['mae']:.4f}")
                print(f"Features used ({len(used_features)}): {', '.join(used_features)}")
                
                # Check if Lat/Lon is included
                has_spatial = any(feat in ['Lat', 'Lon'] for feat in used_features)
                if has_spatial:
                    spatial_feats = [f for f in used_features if f in ['Lat', 'Lon']]
                    print(f"✓ Detected spatial features: {', '.join(spatial_feats)} - will be automatically calculated")
                
                # Input raster directory
                raster_dir = input("\nPlease enter raster directory: ").strip()
                if not raster_dir or not os.path.isdir(raster_dir):
                    print("Directory does not exist or is invalid")
                    continue
                
                pattern = input("Please enter filename pattern (default {name}.tif): ").strip() or "{name}.tif"
                
                # Ask whether preprocessing is needed
                preprocess_choice = input("\nDo you need to preprocess raster images to unify size and projection? (y/n, default y): ").strip().lower()
                if preprocess_choice != 'n':
                    print("\n--- Raster Preprocessing Options ---")
                    print("Preprocessing will unify all rasters' size, resolution, projection and extent")
                    
                    # Select reference raster
                    file_features = [f for f in used_features if f not in ['Lat', 'Lon']]
                    if file_features:
                        print(f"\nAvailable reference raster features:")
                        for i, feat in enumerate(file_features, 1):
                            print(f"{i}. {feat}")
                        ref_choice = input(f"Please select reference raster (1-{len(file_features)}, default 1): ").strip()
                        if ref_choice.isdigit() and 1 <= int(ref_choice) <= len(file_features):
                            reference_feature = file_features[int(ref_choice) - 1]
                        else:
                            reference_feature = file_features[0]
                        print(f"Selected reference raster: {reference_feature}")
                    else:
                        reference_feature = None
                    
                    # Execute preprocessing
                    try:
                        preprocessed_dir = preprocess_rasters(
                            used_features, 
                            raster_dir, 
                            output_dir=None,  # Default: create preprocessed subdirectory
                            pattern=pattern,
                            reference_feature=reference_feature
                        )
                        # Use preprocessed directory
                        raster_dir = preprocessed_dir
                        print(f"\nWill use preprocessed rasters: {raster_dir}")
                    except Exception as e:
                        print(f"Preprocessing failed: {e}")
                        retry_choice = input("Continue using original rasters? (y/n, default n): ").strip().lower()
                        if retry_choice != 'y':
                            continue
                
                # Output directory
                output_dir = input("\nPlease enter SHAP value raster output directory (default shap_rasters): ").strip()
                if not output_dir:
                    output_dir = "shap_rasters"
                
                # Advanced parameter settings
                print("\n--- Advanced Parameter Settings ---")
                batch_pixels_input = input("Batch processing pixel count (default 100000, larger is faster but uses more memory): ").strip()
                batch_pixels = int(batch_pixels_input) if batch_pixels_input else 100_000
                
                sample_ratio_input = input("SHAP explainer initialization sampling ratio (0-1, default 0.1): ").strip()
                sample_ratio = float(sample_ratio_input) if sample_ratio_input else 0.1
                sample_ratio = max(0.01, min(1.0, sample_ratio))  # Limit between 0.01-1.0
                
                print(f"\nParameter confirmation:")
                print(f"  - Model: {selected_model_key}")
                print(f"  - Raster directory: {raster_dir}")
                print(f"  - Output directory: {output_dir}")
                print(f"  - Batch size: {batch_pixels}")
                print(f"  - Sampling ratio: {sample_ratio}")
                
                confirm = input("\nStart calculation? (y/n, default y): ").strip().lower()
                if confirm == 'n':
                    print("Cancelled")
                    continue
                
                # Start calculation
                print("\n" + "="*70)
                print("Starting calculation of SHAP value spatial distribution map...")
                print("Note: This process may take a long time, depending on raster size and model complexity")
                print("="*70)
                
                output_files = calculate_shap_raster_and_save(
                    model=selected_model,
                    feature_names=used_features,
                    raster_dir=raster_dir,
                    output_dir=output_dir,
                    pattern=pattern,
                    batch_pixels=batch_pixels,
                    sample_ratio=sample_ratio
                )
                
                print("\n" + "="*70)
                print("✓ SHAP value spatial distribution map calculation completed!")
                print(f"✓ Output directory: {output_dir}")
                print(f"✓ Generated {len(output_files)} SHAP value raster files")
                print("\nYou can use GIS software (such as QGIS, ArcGIS) to open these raster files to view spatial distribution")
                print("="*70)
                
        except Exception as e:
            print(f"SHAP value spatial distribution map calculation failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif choose == '19':
        if shap_analyzer is None:
            print("SHAP analyzer not initialized, cannot perform SHAP analysis")
        else:
            print("Generating SHAP feature importance ranking plot with contribution rate pie chart...")
            try:
                sample_size = input("Please enter sample size to speed up calculation (press Enter to use all data): ").strip()
                sample_size = int(sample_size) if sample_size else None
                
                # Select dataset
                data_mode, data_name_cn, data_name_en = get_data_mode_choice()
                print(f"Using {data_name_cn} data to calculate SHAP values...")
                
                shap_values = shap_analyzer.calculate_shap_values(sample_size=sample_size, data_mode=data_mode)
                if shap_values is not None:
                    top_n = int(input("Please enter number of top N important features to display (default 8): ").strip() or '8')
                    
                    # Ask pie chart position
                    print("Please select pie chart position:")
                    print("1: Embedded (upper right corner)")
                    print("2: Independent (right side)")
                    pie_choice = input("Please select (1-2, default 1): ").strip()
                    pie_position = 'embedded' if pie_choice != '2' else 'right'
                    
                    # Ask whether to save image
                    save_choice = input("Save image? (y/n, default n): ").strip().lower()
                    save_path = None
                    if save_choice == 'y':
                        save_path = f"figures/shap_importance_with_pie_{data_name_en.replace(' ', '_')}_top{top_n}.png"
                        # Ensure figures directory exists
                        os.makedirs('figures', exist_ok=True)
                    
                    shap_analyzer.plot_shap_importance_with_pie(
                        shap_values, 
                        top_n=top_n, 
                        title=f"Soil Organic Matter Prediction - SHAP Feature Importance Ranking ({data_name_cn})", 
                        save_path=save_path,
                        pie_position=pie_position
                    )
                else:
                    print("SHAP value calculation failed")
            except Exception as e:
                print(f"SHAP importance ranking plot with pie chart generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif choose == '20':
        try:
            if rasterio is None:
                print("rasterio is not installed, cannot perform raster processing. Please install: pip install rasterio")
            else:
                print("\n=== Calculate SHAP Value Spatial Distribution Map for Single Environmental Factor ===")
                print("This function will generate SHAP value raster map for the specified single environmental factor")
                print("SHAP values reflect the contribution of this factor to prediction results at each pixel")
                
                # Select model
                print("\nAvailable model list:")
                model_list = list(results.keys())
                for i, model_name in enumerate(model_list, 1):
                    model_info = results[model_name]
                    print(f"{i}. {model_name} - R²: {model_info['test']['r2']:.4f}, RMSE: {model_info['test']['rmse']:.4f}, MAE: {model_info['test']['mae']:.4f}")
                
                model_choice = input(f"\nPlease select model (1-{len(model_list)}, default 1-best model): ").strip()
                if model_choice and model_choice.isdigit() and (1 <= int(model_choice) <= len(model_list)):
                    selected_model_key = model_list[int(model_choice) - 1]
                else:
                    selected_model_key = f"{best_model_name}_{best_method_suffix}"
                    print(f"Using best model: {selected_model_key}")
                
                selected_model = results[selected_model_key]['model']
                used_features = list(results[selected_model_key]['selected_features'])
                
                print(f"\nSelected model: {selected_model_key}")
                print(f"Model accuracy - R²: {results[selected_model_key]['test']['r2']:.4f}, RMSE: {results[selected_model_key]['test']['rmse']:.4f}, MAE: {results[selected_model_key]['test']['mae']:.4f}")
                
                # Select target environmental factor
                print(f"\nAvailable environmental factors ({len(used_features)}):")
                for i, feat in enumerate(used_features, 1):
                    print(f"{i:2d}. {feat}")
                
                feature_choice = input(f"\nPlease select environmental factor to calculate SHAP values (1-{len(used_features)}): ").strip()
                if not feature_choice.isdigit() or not (1 <= int(feature_choice) <= len(used_features)):
                    print("Invalid choice")
                    continue
                
                target_feature = used_features[int(feature_choice) - 1]
                print(f"\nSelected environmental factor: {target_feature}")
                
                # Check if Lat/Lon is included
                has_spatial = any(feat in ['Lat', 'Lon'] for feat in used_features)
                if has_spatial:
                    spatial_feats = [f for f in used_features if f in ['Lat', 'Lon']]
                    print(f"✓ Detected spatial features: {', '.join(spatial_feats)} - will be automatically calculated")
                
                # Input raster directory
                raster_dir = input("\nPlease enter raster directory: ").strip()
                if not raster_dir or not os.path.isdir(raster_dir):
                    print("Directory does not exist or is invalid")
                    continue
                
                pattern = input("Please enter filename pattern (default {name}.tif): ").strip() or "{name}.tif"
                
                # Ask whether preprocessing is needed
                preprocess_choice = input("\nDo you need to preprocess raster images to unify size and projection? (y/n, default y): ").strip().lower()
                if preprocess_choice != 'n':
                    print("\n--- Raster Preprocessing Options ---")
                    print("Preprocessing will unify all rasters' size, resolution, projection and extent")
                    
                    # Select reference raster
                    file_features = [f for f in used_features if f not in ['Lat', 'Lon']]
                    if file_features:
                        print(f"\nAvailable reference raster features:")
                        for i, feat in enumerate(file_features, 1):
                            print(f"{i}. {feat}")
                        ref_choice = input(f"Please select reference raster (1-{len(file_features)}, default 1): ").strip()
                        if ref_choice.isdigit() and 1 <= int(ref_choice) <= len(file_features):
                            reference_feature = file_features[int(ref_choice) - 1]
                        else:
                            reference_feature = file_features[0]
                        print(f"Selected reference raster: {reference_feature}")
                    else:
                        reference_feature = None
                    
                    # Execute preprocessing
                    try:
                        preprocessed_dir = preprocess_rasters(
                            used_features, 
                            raster_dir, 
                            output_dir=None,  # Default: create preprocessed subdirectory
                            pattern=pattern,
                            reference_feature=reference_feature
                        )
                        # Use preprocessed directory
                        raster_dir = preprocessed_dir
                        print(f"\nWill use preprocessed rasters: {raster_dir}")
                    except Exception as e:
                        print(f"Preprocessing failed: {e}")
                        retry_choice = input("Continue using original rasters? (y/n, default n): ").strip().lower()
                        if retry_choice != 'y':
                            continue
                
                # Output path
                default_output = f"SHAP_{target_feature}.tif"
                output_path = input(f"\nPlease enter SHAP value raster output path (default {default_output}): ").strip()
                if not output_path:
                    output_path = default_output
                
                # Ensure output path has .tif extension
                if not output_path.lower().endswith(('.tif', '.tiff')):
                    output_path = output_path + ".tif"
                
                # Advanced parameter settings
                print("\n--- Advanced Parameter Settings ---")
                batch_pixels_input = input("Batch processing pixel count (default 100000, larger is faster but uses more memory): ").strip()
                batch_pixels = int(batch_pixels_input) if batch_pixels_input else 100_000
                
                sample_ratio_input = input("SHAP explainer initialization sampling ratio (0-1, default 0.1): ").strip()
                sample_ratio = float(sample_ratio_input) if sample_ratio_input else 0.1
                sample_ratio = max(0.01, min(1.0, sample_ratio))  # Limit between 0.01-1.0
                
                print(f"\nParameter confirmation:")
                print(f"  - Model: {selected_model_key}")
                print(f"  - Environmental factor: {target_feature}")
                print(f"  - Raster directory: {raster_dir}")
                print(f"  - Output path: {output_path}")
                print(f"  - Batch size: {batch_pixels}")
                print(f"  - Sampling ratio: {sample_ratio}")
                
                confirm = input("\nStart calculation? (y/n, default y): ").strip().lower()
                if confirm == 'n':
                    print("Cancelled")
                    continue
                
                # Start calculation
                print("\n" + "="*70)
                print(f"Starting calculation of SHAP value spatial distribution map for '{target_feature}'...")
                print("Note: This process may take a long time, depending on raster size and model complexity")
                print("="*70)
                
                output_file = calculate_single_feature_shap_raster(
                    model=selected_model,
                    feature_names=used_features,
                    target_feature=target_feature,
                    raster_dir=raster_dir,
                    output_path=output_path,
                    pattern=pattern,
                    batch_pixels=batch_pixels,
                    sample_ratio=sample_ratio
                )
                
                print("\n" + "="*70)
                print("✓ SHAP value spatial distribution map calculation completed!")
                print(f"✓ Output file: {output_file}")
                print("\nYou can use GIS software (such as QGIS, ArcGIS) to open this raster file to view spatial distribution")
                print("="*70)
                
        except Exception as e:
            print(f"Single environmental factor SHAP value spatial distribution map calculation failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif choose == '21':
        if shap_analyzer is None:
            print("SHAP analyzer not initialized, cannot perform PDP plotting")
        else:
            try:
                print("\n=== Batch Generate PDP Plots for All Features ===")
                print("This function will generate Partial Dependence Plots (PDP) for all environmental factors")
                print("PDP plots show the average effect of a single feature change on model predictions")
                
                # Parameter settings
                print("\n--- Parameter Settings ---")
                
                # Grid resolution
                grid_res_input = input("Please enter grid resolution (default 50, larger makes curves smoother but calculation slower): ").strip()
                grid_res = int(grid_res_input) if grid_res_input else 50
                
                # Dataset selection
                data_mode, data_name_cn, data_name_en = get_data_mode_choice()
                
                # ICE curves
                n_ice_input = input("Number of ICE curves to overlay (default 0, recommended 20-50): ").strip()
                n_ice = int(n_ice_input) if n_ice_input else 0
                
                # Save directory
                save_dir_input = input("Please enter save directory (default figures/pdp): ").strip()
                save_dir = save_dir_input if save_dir_input else 'figures/pdp'
                
                # Image size
                figsize_input = input("Please enter single image size (format: width,height, default 8,5): ").strip()
                if figsize_input:
                    try:
                        w, h = map(float, figsize_input.split(','))
                        figsize = (w, h)
                    except:
                        figsize = (8, 5)
                        print("Input format error, using default size (8, 5)")
                else:
                    figsize = (8, 5)
                
                # Summary plot columns
                cols_input = input("Please enter number of columns per row in summary plot (default 3): ").strip()
                cols = int(cols_input) if cols_input else 3
                
                # Confirm parameters
                print(f"\nParameter confirmation:")
                print(f"  - Dataset: {data_name_cn}")
                print(f"  - Number of features: {len(shap_analyzer.feature_names)}")
                print(f"  - Grid resolution: {grid_res}")
                print(f"  - Number of ICE curves: {n_ice}")
                print(f"  - Save directory: {save_dir}")
                print(f"  - Image size: {figsize}")
                print(f"  - Summary plot columns: {cols}")
                
                confirm = input("\nStart generation? (y/n, default y): ").strip().lower()
                if confirm == 'n':
                    print("Cancelled")
                    continue
                
                # Start generation
                print("\n" + "="*70)
                print(f"Starting batch generation of PDP plots (using {data_name_cn} data)...")
                print("Note: This process may take a long time, depending on number of features and grid resolution")
                print("="*70)
                
                shap_analyzer.plot_all_partial_dependence(
                    grid_resolution=grid_res,
                    n_ice_curves=n_ice,
                    save_dir=save_dir,
                    figsize=figsize,
                    cols=cols,
                    data_mode=data_mode
                )
                
            except Exception as e:
                print(f"Batch PDP plot generation failed: {e}")
                import traceback
                traceback.print_exc()
    
    elif choose == '0':
        print("Exiting program")
        break
    else:
        print(f"{choose} is not a valid option, please select again")
