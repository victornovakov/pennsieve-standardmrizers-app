import pandas as pd 
import h5py
import numpy as np

class DataLoader:
    """
    A class to handle loading and processing of connectivity and FreeSurfer data.
    """
    
    def __init__(self):
        """Initialize DataLoader instance"""
        self.connectivity = None
        self.roi_volume = None
        self.metadata = None
        

    def load_connectivity(self, connectivity_path: str, 
                         connectivity_type: str = 'sc', 
                         atlas: str = 'desikanKilliany',
                         dwi_metrics: set = {'count', 'length', 'fa', 'md', 'ad', 'rd', 'qa'}) -> pd.DataFrame:
        """
        Load connectivity data from an h5 file into a pandas DataFrame
        
        Parameters
        ----------
        connectivity_path : str
            Path to the h5 file containing connectivity data
        connectivity_type : str, optional
            Type of connectivity (default is 'sc' for structural connectivity)
        atlas : str, optional
            Brain atlas to use (default is 'desikanKilliany')
        dwi_metrics : set, optional
            Set of DWI metrics to load (default includes all available metrics)
        """
        # Initialize list to store DataFrames for better performance
        df_list = []
        
        with h5py.File(connectivity_path, 'r') as conn:
            subjects = list(conn.keys())
            for sub in subjects:
                # Add subject with valid metrics or NaN for invalid ones
                metrics_data = {'record_id': [sub]}
                for metric in dwi_metrics:
                    data = np.array(conn[f'{sub}/{connectivity_type}-{atlas}/{metric}'])
                    # Set to NaN if all values are zero, otherwise keep the data
                    metrics_data[metric] = [np.nan if np.all(data == 0) else data]
                sub_df = pd.DataFrame(metrics_data)
                df_list.append(sub_df)
            
        # Combine all DataFrames at once
        df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
        
        self.connectivity = df
        return self

    def load_roi_volume(self, freesurfer_path):
        """
        Load freesurfer data from an excel file
        
        Parameters
        ----------
        freesurfer_path : str
            Path to the excel file containing freesurfer data
        """
        subcortical_Volume = pd.read_excel(freesurfer_path, sheet_name='Subcortical_Volume')
        cortical_Volume = pd.read_excel(freesurfer_path, sheet_name='corticalVolume')
        self.roi_volume = pd.merge(cortical_Volume, subcortical_Volume)
        return self

    def filter_data(self, metadata):
        """
        Filter data based on metadata conditions
        
        Parameters
        ----------
        metadata : pandas DataFrame
            Metadata containing patient information
        
        Returns
        -------
        tuple
            Filtered connectivity, metadata, and ROI volume DataFrames
        """
        if any(df is None for df in [self.connectivity, self.roi_volume]):
            raise ValueError("Please load connectivity and ROI volume data before filtering")
            
        # Filter for Engel 1 outcomes
        engel1 = metadata['engelOutcome'].str.startswith('1').fillna(False)
        metadata_engel1 = metadata[engel1]
        
        # Filter for laser and resection cases
        laser = metadata_engel1['ResectionVsLaser_Surgery'].str.contains('laser', case=False, na=False)
        resection = metadata_engel1['ResectionVsLaser_Surgery'].str.contains('resection', case=False, na=False)
        
        # Combine filtered cases
        metadata_filt = pd.concat([
            metadata_engel1[laser],
            metadata_engel1[resection]
        ]).reset_index(drop=True)
        
        # Filter connectivity and ROI volume based on metadata
        connectivity_filt = self.connectivity[self.connectivity.record_id.isin(metadata_filt.record_id)]
        roi_volume_filt = self.roi_volume[self.roi_volume.record_id.isin(metadata_filt.record_id)]
        
        # Set index for filtered DataFrames
        connectivity_filt.set_index('record_id', inplace=True)
        roi_volume_filt.set_index('record_id', inplace=True)
        
        return connectivity_filt, metadata_filt, roi_volume_filt
    
if __name__ == "__main__":
    pass
