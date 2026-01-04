import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
from matplotlib import rcParams
warnings.filterwarnings('ignore')

# Configure matplotlib font settings to ensure proper display of English text
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['figure.dpi'] = 120
rcParams['savefig.dpi'] = 300
rcParams['figure.autolayout'] = True

class SHAPAnalyzer:
    
    def __init__(self, model, X_train, X_test, feature_names, y_train=None, y_test=None):

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.y_train = y_train
        self.y_test = y_test
        
        # Initialize SHAP explainer
        self.explainer = None
        self.shap_values_train = None
        self.shap_values_test = None
        self.shap_values_all = None  # SHAP values for all datasets
        self._init_explainer()
    
    def _init_explainer(self):
        try:
            model_type = str(type(self.model)).lower()
            
            if 'lgb' in model_type:
                self.explainer = shap.Explainer(self.model)
            elif 'randomforest' in model_type or 'extratrees' in model_type:
                self.explainer = shap.Explainer(self.model)
            elif 'gradientboosting' in model_type:
                self.explainer = shap.Explainer(self.model)
            else:
                # For other models, use KernelExplainer
                self.explainer = shap.KernelExplainer(self.model.predict, self.X_train[:100])
            
            print(f"SHAP explainer initialized successfully: {type(self.explainer).__name__}")
            
        except Exception as e:
            print(f"SHAP explainer initialization failed: {e}")
            self.explainer = None
    
    def calculate_shap_values(self, use_test=True, sample_size=None, data_mode=None):
        
        if self.explainer is None:
            print("SHAP explainer not initialized, cannot calculate SHAP values")
            return None
        
        try:
            # Select dataset
            if data_mode is not None:
                # Use new data_mode parameter
                if data_mode == 'test':
                    X_data = self.X_test
                    data_name = "Test Set"
                elif data_mode == 'train':
                    X_data = self.X_train
                    data_name = "Train Set"
                elif data_mode == 'all':
                    # Merge training and test sets
                    if not isinstance(self.X_train, np.ndarray):
                        X_train_arr = np.array(self.X_train)
                    else:
                        X_train_arr = self.X_train
                    if not isinstance(self.X_test, np.ndarray):
                        X_test_arr = np.array(self.X_test)
                    else:
                        X_test_arr = self.X_test
                    X_data = np.vstack([X_train_arr, X_test_arr])
                    data_name = "All Data"
                else:
                    raise ValueError(f"Invalid data_mode parameter: {data_mode}, should be 'test', 'train' or 'all'")
            else:
                # Backward compatibility: use use_test parameter
                if use_test:
                    X_data = self.X_test
                    data_name = "Test Set"
                else:
                    X_data = self.X_train
                    data_name = "Train Set"
            
            # Sample to speed up calculation
            if sample_size and len(X_data) > sample_size:
                indices = np.random.choice(len(X_data), sample_size, replace=False)
                X_sample = X_data[indices]
                print(f"Using {sample_size} samples from {data_name} for SHAP calculation...")
            else:
                X_sample = X_data
                print(f"Using all {len(X_sample)} samples from {data_name} for SHAP calculation...")
            
            # Calculate SHAP values
            if isinstance(self.explainer, shap.KernelExplainer):
                shap_values = self.explainer.shap_values(X_sample)
            else:
                shap_values = self.explainer(X_sample)
            
            # Store SHAP values (according to data mode)
            if data_mode == 'test' or (data_mode is None and use_test):
                self.shap_values_test = shap_values
            elif data_mode == 'train' or (data_mode is None and not use_test):
                self.shap_values_train = shap_values
            elif data_mode == 'all':
                self.shap_values_all = shap_values
            
            print(f"SHAP values calculation completed, shape: {shap_values.values.shape if hasattr(shap_values, 'values') else shap_values.shape}")
            return shap_values
            
        except Exception as e:
            print(f"SHAP values calculation failed: {e}")
            return None
    
    def plot_shap_importance(self, shap_values=None, top_n=20, title="SHAP Feature Importance Ranking", save_path=None):
       
        if shap_values is None:
            shap_values = self.shap_values_test if self.shap_values_test is not None else self.shap_values_train
        
        if shap_values is None:
            print("Please calculate SHAP values first")
            return
        
        try:
            # Calculate feature importance
            if hasattr(shap_values, 'values'):
                importance_scores = np.abs(shap_values.values).mean(axis=0)
            else:
                importance_scores = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=True)
            
            # Get top top_n features
            top_features = importance_df.tail(top_n)
            
            # Set font parameters to ensure proper display of English feature names
            plt.rcParams.update({
                'font.family': 'DejaVu Sans',
                'font.size': 10,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10
            })
            
            # Plot horizontal bar chart
            plt.figure(figsize=(10, max(6, top_n * 0.4)))
            bars = plt.barh(range(len(top_features)), top_features['importance'], 
                           color='steelblue', alpha=0.7)
            
            # Set labels - ensure feature names use correct font
            feature_labels = [str(name) for name in top_features['feature'].tolist()]
            plt.yticks(range(len(top_features)), feature_labels, fontsize=10, fontfamily='DejaVu Sans')
            plt.xlabel('SHAP Importance (Mean |SHAP Value|)', fontsize=12, fontweight='bold')
            plt.title(title, fontsize=14, fontweight='bold')
            
            # Add values on bars
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{importance:.4f}', va='center', ha='left', fontsize=10, fontfamily='DejaVu Sans')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP importance plot saved to: {save_path}")
            
            plt.show()
            
            # Print importance ranking
            print(f"\n{title}:")
            print("-" * 50)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<20} {row['importance']:.4f}")
            
        except Exception as e:
            print(f"Failed to plot SHAP importance: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_shap_importance_with_pie(self, shap_values=None, top_n=20, title="SHAP Feature Importance Ranking", 
                                       save_path=None, pie_position='right'):
       
        if shap_values is None:
            shap_values = self.shap_values_test if self.shap_values_test is not None else self.shap_values_train
        
        if shap_values is None:
            print("Please calculate SHAP values first")
            return
        
        try:
            # Calculate feature importance
            if hasattr(shap_values, 'values'):
                importance_scores = np.abs(shap_values.values).mean(axis=0)
            else:
                importance_scores = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame and sort (descending, most important on top)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            # Get top top_n features
            top_features = importance_df.head(top_n).copy()
            
            # Calculate contribution rate (percentage)
            total_importance = top_features['importance'].sum()
            top_features['contribution'] = (top_features['importance'] / total_importance) * 100
            
            # Keep font style consistent with summary plot (don't override rcParams globally, only control in this plot)
            
            # Create figure and subplot (consistent size strategy with summary plot)
            fig = plt.figure(figsize=(10, max(6, top_n * 0.3)))
            
            # Left bar chart (occupies main space)
            ax_bar = plt.subplot(1, 1, 1)
            
            # Reverse order so most important is on top
            top_features_reversed = top_features.iloc[::-1]
            
            # Generate color gradient (from dark blue to light blue)
            colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features_reversed)))
            
            # Plot horizontal bar chart
            bars = ax_bar.barh(range(len(top_features_reversed)), 
                              top_features_reversed['importance'].values,
                              color=colors_bar, 
                              alpha=0.8,
                              edgecolor='white',
                              linewidth=0.5)
            
            # Set Y-axis labels (feature names)
            feature_labels = [str(name) for name in top_features_reversed['feature'].tolist()]
            ax_bar.set_yticks(range(len(top_features_reversed)))
            ax_bar.set_yticklabels(feature_labels, fontsize=10)
            
            # Set X-axis label
            ax_bar.set_xlabel('SHAP value', fontsize=10, fontweight='bold')
            ax_bar.set_title(title, fontsize=14, fontweight='bold', pad=15)
            
            # Add values on the right side of bars
            max_width = max(top_features_reversed['importance'])
            for i, (idx, row) in enumerate(top_features_reversed.iterrows()):
                ax_bar.text(row['importance'] + max_width * 0.01, i,
                           f"{row['importance']:.4f}", 
                           va='center', ha='left', 
                           fontsize=10)
            
            # Remove top and right spines
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['left'].set_linewidth(1.2)
            ax_bar.spines['bottom'].set_linewidth(1.2)
            
            # Add grid
            ax_bar.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
            ax_bar.set_axisbelow(True)
            
            # Pie chart position settings
            if pie_position == 'embedded' or pie_position == 'upper_right':
                # Embed pie chart in upper right corner of bar chart (enlarged)
                # Calculate pie chart position (based on axis coordinates)
                ax_pie = fig.add_axes([0.58, 0.45, 0.40, 0.50])  # [left, bottom, width, height]
            else:
                # Independent pie chart on the right (enlarged)
                ax_pie = fig.add_axes([0.70, 0.15, 0.30, 0.70])
            
            # Prepare pie chart data (keep original order, most important first)
            pie_labels = [str(name) for name in top_features['feature'].tolist()]
            pie_sizes = top_features['contribution'].values
            
            # Generate pie chart colors (corresponding to bar chart colors)
            colors_pie = plt.cm.Blues(np.linspace(0.9, 0.4, len(top_features)))
            
            # Plot pie chart (only show percentage, not feature names)
            wedges, texts, autotexts = ax_pie.pie(pie_sizes, 
                                        labels=None,  # Don't use labels parameter
                                        autopct='%1.1f%%',  # Only show percentage
                                        startangle=90,
                                        colors=colors_pie,
                                        textprops={'fontsize': 10, 'weight': 'bold', 'color': 'white', 'family': 'DejaVu Sans'},
                                        pctdistance=0.7)  # Text distance from center
            
            # Rotate percentage text to follow wedge direction (rotate around center)
            for autotext, wedge in zip(autotexts, wedges):
                # Calculate wedge center angle
                angle = (wedge.theta2 + wedge.theta1) / 2
                # Adjust text direction: flip text in left half circle to avoid inversion
                if 90 < angle < 270:
                    angle = angle - 180
                autotext.set_rotation(angle)
                autotext.set_rotation_mode('anchor')
                autotext.set_ha('center')
                autotext.set_va('center')
            
            # Adjust layout
            plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1)
            
            # Save figure
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"SHAP importance plot with pie chart saved to: {save_path}")
            
            plt.show()
            
            # Print statistics
            print(f"\n{title}")
            print("="*70)
            print(f"{'Rank':<6} {'Feature':<20} {'SHAP Value':<12} {'Contribution':<12}")
            print("-"*70)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                print(f"{i:<6} {row['feature']:<20} {row['importance']:<12.4f} {row['contribution']:<12.2f}%")
            print("="*70)
            print(f"Cumulative contribution of top {top_n} features: 100.00%")
            
            # Calculate cumulative contribution rates for top 5 and top 10
            if len(top_features) >= 5:
                top5_contrib = top_features.head(5)['contribution'].sum()
                print(f"Cumulative contribution of top 5 features: {top5_contrib:.2f}%")
            if len(top_features) >= 10:
                top10_contrib = top_features.head(10)['contribution'].sum()
                print(f"Cumulative contribution of top 10 features: {top10_contrib:.2f}%")
            
        except Exception as e:
            print(f"Failed to plot SHAP importance with pie chart: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_shap_summary(self, shap_values=None, max_display=20, title="SHAP Summary Plot", save_path=None):
       
        if shap_values is None:
            shap_values = self.shap_values_test if self.shap_values_test is not None else self.shap_values_train
        
        if shap_values is None:
            print("Please calculate SHAP values first")
            return
        
        try:
            # Get corresponding data
            if shap_values == self.shap_values_test:
                X_data = self.X_test
            else:
                X_data = self.X_train
            
            # Ensure data shapes match
            if hasattr(shap_values, 'values'):
                if len(X_data) != len(shap_values.values):
                    # If lengths don't match, take first N samples
                    min_len = min(len(X_data), len(shap_values.values))
                    X_data = X_data[:min_len]
                    shap_values = shap_values[:min_len]
            
            # Create DataFrame
            X_df = pd.DataFrame(X_data, columns=self.feature_names)
            
            # Plot summary plot
            plt.figure(figsize=(10, max(6, max_display * 0.3)))
            shap.summary_plot(shap_values, X_df, max_display=max_display, show=False, cmap='RdBu_r')
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save figure
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP summary plot saved to: {save_path}")
            
            plt.show()
            
            print(f"{title} generated successfully")
            
        except Exception as e:
            print(f"Failed to plot SHAP summary: {e}")
    
    
    def plot_partial_dependence(self, feature_name, grid_resolution=50, use_test=True,
                                 n_ice_curves=0, title=None, save_path=None, data_mode=None):
   
        try:
            if feature_name not in self.feature_names:
                print(f"Feature '{feature_name}' does not exist")
                return
            
            # Select data
            if data_mode is not None:
                if data_mode == 'test':
                    X_data = self.X_test
                elif data_mode == 'train':
                    X_data = self.X_train
                elif data_mode == 'all':
                    # Merge training and test sets
                    if not isinstance(self.X_train, np.ndarray):
                        X_train_arr = np.array(self.X_train)
                    else:
                        X_train_arr = self.X_train
                    if not isinstance(self.X_test, np.ndarray):
                        X_test_arr = np.array(self.X_test)
                    else:
                        X_test_arr = self.X_test
                    X_data = np.vstack([X_train_arr, X_test_arr])
                else:
                    raise ValueError(f"Invalid data_mode parameter: {data_mode}, should be 'test', 'train' or 'all'")
            else:
                # Backward compatibility: use use_test parameter
                X_data = self.X_test if use_test else self.X_train
            
            if not isinstance(X_data, np.ndarray):
                X_data = np.array(X_data)
            feature_idx = self.feature_names.index(feature_name)

            # Feature value grid (using quantiles is more robust)
            col = X_data[:, feature_idx].astype(float)
            valid = ~np.isnan(col)
            if not np.any(valid):
                print(f"No valid data for PDP of '{feature_name}'")
                return
            col = col[valid]
            quantiles = np.linspace(0.01, 0.99, grid_resolution)
            grid = np.quantile(col, quantiles)

            # Calculate PDP: for each grid value, replace the feature, predict and take mean
            pdp_values = []
            ice_samples = None
            if n_ice_curves and n_ice_curves > 0:
                n_ice = min(n_ice_curves, len(X_data))
                rng = np.random.default_rng(42)
                ice_idx = rng.choice(len(X_data), n_ice, replace=False)
                ice_samples = X_data[ice_idx].copy()
                ice_curves = np.zeros((n_ice, len(grid)))

            for j, v in enumerate(grid):
                X_tmp = X_data.copy()
                X_tmp[:, feature_idx] = v
                preds = self.model.predict(X_tmp)
                pdp_values.append(np.mean(preds))
                if ice_samples is not None:
                    X_ice = ice_samples.copy()
                    X_ice[:, feature_idx] = v
                    ice_curves[:, j] = self.model.predict(X_ice)

            pdp_values = np.array(pdp_values)

            # Plot
            plt.figure(figsize=(9, 6))
            plt.plot(grid, pdp_values, color='black', linewidth=2, label='PDP (average)')
            if ice_samples is not None:
                for i in range(ice_curves.shape[0]):
                    plt.plot(grid, ice_curves[i], color='gray', alpha=0.3, linewidth=1)
                plt.legend(frameon=False, fontsize=10)
            display_name = 'pH' if str(feature_name).upper() == 'PH' else str(feature_name)
            plt.xlabel(f"{display_name}", fontsize=12)
            plt.ylabel("Model Prediction", fontsize=12)
            if title is None:
                title = f"Partial Dependence Plot - {display_name}"
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(False)  # Remove grid background
            plt.tight_layout()
            
            # Save figure
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"PDP saved to: {save_path}")
            
            plt.show()
            print("PDP generated successfully")
        except Exception as e:
            print(f"Failed to plot Partial Dependence for '{feature_name}': {e}")

    def plot_all_partial_dependence(self, grid_resolution=50, use_test=True, n_ice_curves=0, 
                                     save_dir='figures/pdp', figsize=(8, 5), cols=3, data_mode=None):
       
        import os
        import math
        
        try:
            # Create save directory
            os.makedirs(save_dir, exist_ok=True)
            
            n_features = len(self.feature_names)
            
            # Select dataset
            if data_mode is not None:
                # Use new data_mode parameter
                if data_mode == 'test':
                    X_data = self.X_test
                    data_type = "Test Set"
                elif data_mode == 'train':
                    X_data = self.X_train
                    data_type = "Train Set"
                elif data_mode == 'all':
                    # Merge training and test sets
                    if not isinstance(self.X_train, np.ndarray):
                        X_train_arr = np.array(self.X_train)
                    else:
                        X_train_arr = self.X_train
                    if not isinstance(self.X_test, np.ndarray):
                        X_test_arr = np.array(self.X_test)
                    else:
                        X_test_arr = self.X_test
                    X_data = np.vstack([X_train_arr, X_test_arr])
                    data_type = "All Data"
                else:
                    raise ValueError(f"Invalid data_mode parameter: {data_mode}, should be 'test', 'train' or 'all'")
            else:
                # Backward compatibility: use use_test parameter
                X_data = self.X_test if use_test else self.X_train
                data_type = "Test Set" if use_test else "Train Set"
            
            if not isinstance(X_data, np.ndarray):
                X_data = np.array(X_data)
            
            print(f"\nStarting batch generation of PDP plots for {n_features} features...")
            print(f"Dataset: {data_type}")
            print(f"Grid resolution: {grid_resolution}")
            print(f"Number of ICE curves: {n_ice_curves}")
            print(f"Save directory: {save_dir}")
            print("="*70)
            
            success_count = 0
            failed_features = []
            
            # Generate PDP plot for each feature
            for idx, feature_name in enumerate(self.feature_names, 1):
                try:
                    print(f"[{idx}/{n_features}] Generating PDP plot for '{feature_name}'...", end=' ')
                    
                    feature_idx = self.feature_names.index(feature_name)
                    
                    # Feature value grid (using quantiles is more robust)
                    col = X_data[:, feature_idx].astype(float)
                    valid = ~np.isnan(col)
                    if not np.any(valid):
                        print(f"❌ No valid data")
                        failed_features.append(feature_name)
                        continue
                    
                    col = col[valid]
                    quantiles = np.linspace(0.01, 0.99, grid_resolution)
                    grid = np.quantile(col, quantiles)
                    
                    # Calculate PDP: for each grid value, replace the feature, predict and take mean
                    pdp_values = []
                    ice_samples = None
                    if n_ice_curves and n_ice_curves > 0:
                        n_ice = min(n_ice_curves, len(X_data))
                        rng = np.random.default_rng(42)
                        ice_idx = rng.choice(len(X_data), n_ice, replace=False)
                        ice_samples = X_data[ice_idx].copy()
                        ice_curves = np.zeros((n_ice, len(grid)))
                    
                    for j, v in enumerate(grid):
                        X_tmp = X_data.copy()
                        X_tmp[:, feature_idx] = v
                        preds = self.model.predict(X_tmp)
                        pdp_values.append(np.mean(preds))
                        if ice_samples is not None:
                            X_ice = ice_samples.copy()
                            X_ice[:, feature_idx] = v
                            ice_curves[:, j] = self.model.predict(X_ice)
                    
                    pdp_values = np.array(pdp_values)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=figsize)
                    
                    # Plot ICE curves (if any)
                    if ice_samples is not None:
                        for i in range(ice_curves.shape[0]):
                            ax.plot(grid, ice_curves[i], color='gray', alpha=0.2, linewidth=0.8, zorder=1)
                    
                    # Plot main PDP curve
                    ax.plot(grid, pdp_values, color='#2E86AB', linewidth=2.5, label='PDP', zorder=2)
                    
                    # Set labels and title (normalize PH to pH)
                    display_name = 'pH' if str(feature_name).upper() == 'PH' else str(feature_name)
                    ax.set_xlabel(f"{display_name}", fontsize=20, fontweight='bold')
                    ax.set_ylabel("Predicted Value", fontsize=20, fontweight='bold')
                    ax.tick_params(axis='both', labelsize=20)
                                     
                    # Add legend (if ICE curves exist)
                    if ice_samples is not None:
                        ice_line = plt.Line2D([0], [0], color='gray', alpha=0.5, linewidth=1)
                        ax.legend([ice_line, ax.lines[-1]], 
                                [f'ICE (n={n_ice_curves})', 'PDP (average)'],
                                frameon=True, fontsize=10, loc='best', 
                                fancybox=True, shadow=True)
                    
                    # Grid and style optimization
                    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    plt.tight_layout()
                    
                    # Save figure
                    save_path = os.path.join(save_dir, f"pdp_{feature_name}.png")
                    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    
                    print(f"✓ Saved")
                    success_count += 1
                    
                except Exception as e:
                    print(f"❌ Failed: {e}")
                    failed_features.append(feature_name)
                    plt.close('all')
                    continue
            
            # Output summary information
            print("\n" + "="*70)
            print(f"✓ Batch PDP plot generation completed!")
            print(f"✓ Success: {success_count}/{n_features}")
            if failed_features:
                print(f"✗ Failed features ({len(failed_features)}): {', '.join(failed_features)}")
            print(f"✓ Save directory: {save_dir}")
            print("="*70)
            
            # Generate summary plot (optional) - combine all PDP plots into one large figure
            self._create_pdp_summary_grid(save_dir, cols=cols)
            
        except Exception as e:
            print(f"\nBatch PDP plot generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_pdp_summary_grid(self, pdp_dir, cols=3):
       
        import os
        import glob
        from PIL import Image
        import math
        
        try:
            # Get all PDP images
            pdp_files = sorted(glob.glob(os.path.join(pdp_dir, "pdp_*.png")))
            if not pdp_files:
                return
            
            n_plots = len(pdp_files)
            rows = math.ceil(n_plots / cols)
            
            print(f"\nGenerating PDP summary grid plot ({rows}×{cols})...")
            
            # Create subplot grid
            fig, axes = plt.subplots(3, 3, figsize=(18, 13.5))
            cols = 3
            rows = 3
            # Ensure axes is a 2D array
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            for idx, pdp_file in enumerate(pdp_files):
                row = idx // cols
                col = idx % cols
                
                img = Image.open(pdp_file)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
            for idx in range(n_plots, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].axis('off')
            
           
            plt.tight_layout()
            
            # Save summary plot
            summary_path = os.path.join(pdp_dir, "pdp_all_summary.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"✓ PDP summary grid plot saved: {summary_path}")
            
        except Exception as e:
            print(f"Failed to generate PDP summary grid plot: {e}")
            plt.close('all')

