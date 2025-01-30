#!/usr/bin/env python3.9

import sys
import shutil
import os

#%% Import Libraries
import neuroHarmonize as nh
import numpy as np
import pandas as pd
import sklearn.manifold as manifold
import warnings
import dataload
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%%
class T1MRIHarmonizer:
    """A class to process and harmonize T1 MRI data."""

    def __init__(self, 
                 roi_volume: pd.DataFrame, 
                 metadata: pd.DataFrame,
                 harmonized_data: Optional[pd.DataFrame] = None):
        """
        Initialize the T1MRIHarmonizer.
        
        Args:
            cache_size (int): Size of the cache for memoization
        """
        warnings.filterwarnings('ignore', category=FutureWarning)
        self._original_data = roi_volume
        self._harmonized_data = harmonized_data
        self._metadata = metadata

    def harmonize_roi_volume(self) -> pd.DataFrame:
        """
        Harmonize the roi_volume data using neuroHarmonize package.
        
        Args:
            roi_volume: The roi_volume data
            metadata: The metadata containing site, age, control status, and sex
            
        Returns:
            The harmonized roi_volume data
        """
        # Input validation
        required_columns = ['SITE', 'age', 'isControl', 'sexisMale']
        if not all(col in self._metadata.columns for col in required_columns):
            raise ValueError(f"Metadata missing required columns: {required_columns}")

        # Extract features and covariates more efficiently
        data = self._original_data.iloc[:, 5:].values
        covars = self._metadata[required_columns]

        # Perform harmonization
        harmonization_results = nh.harmonizationLearn(
            data,
            covars,
            smooth_terms=['age'],
            ref_batch='PEN',
            seed=27
        )

        # Create harmonized dataframe efficiently
        self._harmonized_data = self._original_data.copy()
        self._harmonized_data.iloc[:, 5:] = harmonization_results[1]

        return self._harmonized_data

    def visualize_harmonization_tsne(self,
                                   not_harmonized: Optional[np.ndarray] = None,
                                   harmonized: Optional[np.ndarray] = None,
                                   batch: Optional[pd.Series] = None,
                                   save_path: Optional[Path] = None) -> None:
        """
        Visualize the harmonization results using t-SNE.
        
        Args:
            not_harmonized: The not harmonized data
            harmonized: The harmonized data
            batch: The batch information
            save_path: Optional path to save the plot as PNG
        """

        # Use provided data or fall back to stored data
        not_harmonized = (self._original_data.iloc[:, 5:].values 
                         if not_harmonized is None else not_harmonized)
        harmonized = (self._harmonized_data.iloc[:, 5:].values 
                     if harmonized is None else harmonized)
        batch = self._metadata['SITE'] if batch is None else batch

        # # Convert data to tuple for caching
        # not_harmonized_tuple = tuple(map(tuple, not_harmonized))
        # harmonized_tuple = tuple(map(tuple, harmonized))

        # Compute t-SNE with caching
        harmonization = self._compute_tsne(harmonized)
        noharmonization = self._compute_tsne(not_harmonized)

        # Create visualization
        fig = self._plot_tsne_results(harmonization, noharmonization, batch, block=False)
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / 'tsne_harmonization.png', bbox_inches='tight', dpi=300)

    def visualize_harmonization_roc(self,
                                  not_harmonized: Optional[np.ndarray] = None,
                                  harmonized: Optional[np.ndarray] = None,
                                  batch: Optional[pd.Series] = None,
                                  save_path: Optional[Path] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                                           Tuple[np.ndarray, np.ndarray]]:
        """
        Visualize ROC curves for harmonized and non-harmonized data.
        
        Args:
            not_harmonized: The not harmonized data
            harmonized: The harmonized data
            batch: The batch information
            save_path: Optional path to save the plots as PNG
            
        Returns:
            Tuple of (non_harmonized_predictions, harmonized_predictions)
            where each prediction is a tuple of (true_values, predicted_probabilities)
        """

        # Use provided data or fall back to stored data
        not_harmonized = (self._original_data.iloc[:, 5:].values 
                         if not_harmonized is None else not_harmonized)
        harmonized = (self._harmonized_data.iloc[:, 5:].values 
                     if harmonized is None else harmonized)
        batch = self._metadata['SITE'] if batch is None else batch

        # Compute predictions
        print("\nComputing ROC curves for non-harmonized data:")
        nh_true, nh_prob, nh_pred = self._predict_sites(not_harmonized, batch)
        print("\nComputing ROC curves for harmonized data:")
        h_true, h_prob, h_pred = self._predict_sites(harmonized, batch)

        # Plot ROC curves
        print("\nPlotting ROC Curves for Non-harmonized Data:")
        fig1 = self.plot_roc_curves(nh_true, nh_prob, batch, block=False)
        
        print("\nPlotting ROC Curves for Harmonized Data:")
        fig2 = self.plot_roc_curves(h_true, h_prob, batch, block=False)

        # Plot confusion matrices
        print("\nPlotting Confusion Matrix for Non-harmonized Data:")
        cm_fig1 = self.plot_confusion_matrix(
            batch, 
            nh_pred, 
            batch, 
            title="Confusion Matrix - Non-harmonized Data",
            block=False
        )
        
        print("\nPlotting Confusion Matrix for Harmonized Data:")
        cm_fig2 = self.plot_confusion_matrix(
            batch, 
            h_pred, 
            batch, 
            title="Confusion Matrix - Harmonized Data",
            block=False
        )

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig1.savefig(save_path / 'roc_non_harmonized.png', bbox_inches='tight', dpi=300)
            fig2.savefig(save_path / 'roc_harmonized.png', bbox_inches='tight', dpi=300)
            cm_fig1.savefig(save_path / 'confusion_matrix_non_harmonized.png', bbox_inches='tight', dpi=300)
            cm_fig2.savefig(save_path / 'confusion_matrix_harmonized.png', bbox_inches='tight', dpi=300)

        return (nh_true, nh_pred), (h_true, h_pred)

    def plot_roc_curves(self, 
                        y_true_bin: np.ndarray, 
                        y_pred_prob: np.ndarray, 
                        batch_labels: pd.Series,
                        block: bool = False) -> plt.Figure:
        """
        Plot ROC curves for each class and macro-average.
        
        Returns:
            matplotlib Figure object
        """
        # Get color palette
        colors = sns.color_palette("husl", len(batch_labels.unique()))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Initialize arrays for macro-average calculation
        all_fpr = np.linspace(0, 1, 100)  # Common FPR thresholds
        mean_tpr = np.zeros_like(all_fpr)
        
        # Plot individual ROC curves and collect TPRs for macro-average
        for class_id in range(len(batch_labels.unique())):
            # Calculate ROC curve for each class
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_id], y_pred_prob[:, class_id])
            roc_auc = auc(fpr, tpr)
            
            # Plot individual curve
            plt.plot(
                fpr, 
                tpr,
                color=colors[class_id],
                label=f"{batch_labels.unique()[class_id]} (AUC = {roc_auc:.2f})",
                alpha=0.5
            )
            
            # Interpolate TPR values for macro-average calculation
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        # Calculate and plot macro-average ROC curve
        mean_tpr /= len(batch_labels.unique())
        macro_roc_auc = auc(all_fpr, mean_tpr)
        
        plt.plot(
            all_fpr,
            mean_tpr,
            color='black',
            label=f'Macro-average ROC (AUC = {macro_roc_auc:.2f})',
            linewidth=2
        )
        
        # Plot chance level
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        self._format_roc_plot(ax)
        plt.show(block=block)
        
        return fig

    def plot_confusion_matrix(self,
                            y_true: pd.Series,
                            y_pred: np.ndarray,
                            batch_labels: pd.Series,
                            title: str = "Confusion Matrix",
                            block: bool = False) -> plt.Figure:
        """
        Plot confusion matrix for site predictions.
        
        Args:
            y_true: True site labels
            y_pred: Predicted site labels
            batch_labels: Series containing site information
            title: Title for the confusion matrix plot
            block: Whether to block execution until plot window is closed
            
        Returns:
            matplotlib Figure object
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=batch_labels.unique()
        )
        
        # Plot with customizations
        disp.plot(
            ax=ax,
            cmap=plt.cm.Blues,
            values_format='d',  # Show counts as integers
            xticks_rotation=45,
            colorbar=True
        )
        
        # Customize plot
        plt.title(title)
        plt.tight_layout()
        plt.show(block=block)
        
        return fig

    # Private helper methods (internal use)
    def _compute_tsne(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute t-SNE transformation.
        
        Args:
            data: Input data
            
        Returns:
            t-SNE transformed data
            
        """
        tsne = manifold.TSNE(
            metric='mahalanobis', 
            random_state=0,
            n_jobs=-1  # Use all available cores
        ).fit_transform(data)

        return tsne
    

    def _predict_sites(self,
                    data: np.ndarray,
                    batch: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform site prediction using Leave-One-Out cross-validation.
        
        Args:
            data: Input data array
            batch: Series containing site information
            
        Returns:
            Tuple containing (binary truth values, prediction probabilities, predictions)
        """
        # Binarize the output labels for multi-class ROC
        classes = np.unique(batch)
        y_true_bin = label_binarize(batch, classes=classes)

        # Get classifier pipeline
        clf = self._create_classifier()
        
        # Perform Leave-One-Out Cross-Validation
        y_pred_prob, y_pred = self._perform_cross_validation(clf, data, batch)

        return y_true_bin, y_pred_prob, y_pred

    def _create_classifier(self) -> Pipeline:
        """
        Create and return the classifier pipeline.
        
        Returns:
            Sklearn Pipeline with StandardScaler and OneVsRestClassifier
        """
        weak_learner = DecisionTreeClassifier(max_leaf_nodes=5)
        clf = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                AdaBoostClassifier(
                    estimator=weak_learner,
                    n_estimators=200,
                    algorithm="SAMME",
                    random_state=27
                )
            )
        )
        return clf

    def _perform_cross_validation(self, 
                                clf: Pipeline, 
                                data: np.ndarray, 
                                batch: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Leave-One-Out cross-validation.
        
        Args:
            clf: Classifier pipeline
            data: Input data array
            batch: Series containing site information
            
        Returns:
            Tuple of (prediction_probabilities, predictions)
        """
        loo = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Get probabilities
        probabilities = cross_val_predict(
            clf, 
            data, 
            batch, 
            cv=loo, 
            method='predict_proba', 
            n_jobs=-1, 
            verbose=2
        )
        
        # Get class predictions
        predictions = cross_val_predict(
            clf, 
            data, 
            batch, 
            cv=loo, 
            method='predict', 
            n_jobs=-1, 
            verbose=2
        )
        
        return probabilities, predictions

    def _plot_tsne_results(self, 
                          harmonization: np.ndarray,
                          noharmonization: np.ndarray,
                          batch: pd.Series,
                          block: bool = False) -> plt.Figure:
        """
        Plot t-SNE results.
        
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        for ax, data, title in zip(
            axes,
            [harmonization, noharmonization],
            ['tsne - Harmonized', 'tsne - No Harmonization']
        ):
            sns.scatterplot(ax=ax, x=data[:,0], y=data[:,1], hue=batch)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(title)
        
        plt.tight_layout()
        plt.show(block=block)
        
        return fig
    
    def _format_roc_plot(self, ax: plt.Axes) -> None:
        """
        Format the ROC plot with proper labels and legend.
        
        Args:
            ax: Matplotlib axes object
        """
        # Remove chance level from legend
        lines = ax.get_lines()
        labels = [line.get_label() for line in lines]
        legend_lines = [line for line in lines if 'chance' not in line.get_label().lower()]
        legend_labels = [label for label in labels if 'chance' not in label.lower()]
        
        # Set titles and labels
        ax.set_title('ROC Curves')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        
        # Configure legend
        ax.legend(
            legend_lines, 
            legend_labels, 
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        plt.tight_layout()


#%% Main function to run as a script
if __name__ == "__main__":

    # project_path = Path(__file__).parent.parent
    src = os.environ['INPUT_DIR']
 
    metadata_path = f'{src}/metadata.csv'
    roi_volume_path = f'{src}/roi_volume.csv'

    #%% Load data
    print(f"Loading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path, index_col='record_id')
    metadata['SITE'] = metadata.index.str[4:7]

    # Initialize DataLoader and load ROI volume data
    roi_volume = pd.read_csv(roi_volume_path, index_col='record_id')

    #%% HUM site has all controls, so we need to remove it
    mask = metadata['SITE'] != 'HUM'
    metadata = metadata[mask]
    roi_volume = roi_volume[mask]

    #%% Process data
    t1mri = T1MRIHarmonizer(roi_volume, metadata)

    dest = os.environ['OUTPUT_DIR']
    save_path = dest

    print('a')
    t1mri.harmonize_roi_volume().to_csv(f'{save_path}/harmonized_roi_volume.csv')
    print('b')
    #t1mri.to_csv('harmonized_roi_volume.csv', index=False) 
    
    
    harmonized_location = dest
    #harmonized_data = t1mri.harmonized_data.to_csv(harmonized_location)
    
    
    # Create save path
    
    
    # save_path.mkdir(parents=True, exist_ok=True)
    
    # Visualize results with save path
    t1mri.visualize_harmonization_tsne(batch=metadata['SITE'], save_path=save_path)
    t1mri.visualize_harmonization_roc(batch=metadata['SITE'], save_path=save_path)
    

#shutil.copytree(src, dest, dirs_exist_ok=True)
#print("end of processing")