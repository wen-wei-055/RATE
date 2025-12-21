import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from obspy import UTCDateTime
import json

STATIONS = json.load(open('example/stations.json','r'))
TOTAL_DATA = h5py.File("example/total_CWASN_window.hdf5","r")
TOTAL_DF = pd.read_hdf("example/total_CWASN_window.hdf5", 'metadata/event_metadata')
N_SPLITS = 30
SAMPLES = 100

DATA_LIST = [TOTAL_DATA]

DF_LIST = [TOTAL_DF]
NAME_LIST = ['total']

def cal_pga(wave):
    return np.sqrt(wave[:,0]**2 + wave[:,1]**2 + wave[:,2]**2)
    

for df_i, (df, dataset_name) in enumerate(zip(DF_LIST, NAME_LIST)):
    data = DATA_LIST[df_i]
    event_datafile_list = list(df['data_file'])
    RAG_time_PGA = np.zeros((len(event_datafile_list), len(STATIONS), 31))  # (6026,249,31)
    # events
    for event_i, event in tqdm(enumerate(event_datafile_list)):
        waveforms = data['data'][event]['waveforms']
        coords = data['data'][event]['coords']
        # stations
        for st_i, st_coords in enumerate(coords):
            waveform = waveforms[st_i]
            station_key = f"{st_coords[0]},{st_coords[1]},{st_coords[2]}"
            st_position = STATIONS[station_key]
            pga = cal_pga(waveform)
            # time
            for time_idx in range(1, N_SPLITS+1):
                time_sample = time_idx*SAMPLES
                max_acc = np.max(pga[:time_sample])
                RAG_time_PGA[event_i, st_position, time_idx] = max_acc
                
        rms_values = (RAG_time_PGA**2).sum(axis=1, keepdims=True)
        rms_values = np.sqrt(rms_values)
        rms_values_expanded = np.repeat(rms_values, RAG_time_PGA.shape[1], axis=1)
        RAG_time_PGA = RAG_time_PGA / rms_values_expanded
        RAG_time_PGA[np.isnan(RAG_time_PGA)] = 0.0
    RAG_time_PGA = RAG_time_PGA.astype('float32')
    np.save(f'{dataset_name}_RAG_time_PGA', RAG_time_PGA)