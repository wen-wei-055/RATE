import h5py
import numpy as np
import pandas as pd
import obspy
import os
import time
import json
import torch
import pickle

from obspy import UTCDateTime
from tqdm import tqdm
from torch.utils.data import Dataset

def detect_location_keys(columns): 
    candidates = [['LAT', 'Latitude(°)', 'Latitude', 'source_latitude_deg'],  
                  ['LON', 'Longitude(°)', 'Longitude', 'source_longitude_deg'],
                  ['DEPTH', 'JMA_Depth(km)', 'Depth(km)', 'Depth/Km', 'source_depth_km']]

    coord_keys = []
    for keyset in candidates:
        for key in keyset:
            if key in columns:
                coord_keys += [key]
                break

    if len(coord_keys) != len(candidates):
        raise ValueError('Unknown location key format')

    return coord_keys

class PreloadedEventGenerator(Dataset):
    def __init__(self, 
                 tag,
                 datapath,
                 min_mag, 
                 limit, 
                 stations_table,
                 first_station_appearance, 
                 last_station_appearance,
                 key='MA', 
                 batch_size=5, 
                 cutout=None, 
                 sliding_window=False, 
                 windowlen=3000,
                 shuffle=True,
                 oversample=1,
                 station_blinding=False, 
                 magnitude_resampling=3,
                 adjust_mean=True,
                 current_station=None, 
                 trigger_based=None, 
                 min_upsample_magnitude=2,
                 integrate=False, 
                 sampling_rate=100.,
                 coord_keys=None, 
                 upsample_high_station_events=None,
                 **kwargs):
        if kwargs:
            print(f'Unused parameters: {", ".join(kwargs.keys())}')
        self.tag = tag
        self.data_path = datapath
        self.stations_table = stations_table
        
        self.event_metadata, self.unuse_table, reassigned_pga = self.load_events(datapath, min_mag, limit, first_station_appearance, last_station_appearance)
        self.batch_size = batch_size 
        
        self.cutout = cutout
        self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
        self.windowlen = windowlen  # Length of window for sliding window
        self.station_blinding = station_blinding
        self.adjust_mean = adjust_mean
        if current_station is None:
            current_station = batch_waveforms.shape[1]
        self.current_station = current_station
        self.trigger_based = trigger_based
        self.integrate = integrate
        self.sampling_rate = sampling_rate

        base_indexes = np.arange(len(self.event_metadata))
        if magnitude_resampling > 1:
            magnitude = self.event_metadata[key].values
            for i in np.arange(min_upsample_magnitude, 9):
                ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                base_indexes = np.concatenate(
                    (base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))

        if upsample_high_station_events is not None:
            new_indexes = []
            for ind in base_indexes:
                #n_stations = batch_waveforms[ind].shape[0]
                n_stations = len(reassigned_pga[ind])
                new_indexes += [ind for _ in range(n_stations // upsample_high_station_events + 1)]
            base_indexes = np.array(new_indexes)

        if coord_keys is None:
            self.coord_keys = detect_location_keys(self.event_metadata.columns)
        else:
            self.coord_keys = coord_keys
        
        self.indexes = np.repeat(base_indexes.copy(), oversample, axis=0)
        if shuffle:
            np.random.shuffle(self.indexes)

        self.total_coords = np.zeros(((len(self.stations_table),4)))
        for coor_i, station_coord in enumerate(list(self.stations_table.keys())):
            self.total_coords[coor_i,:] = station_coord.split(',')[:]

    def __len__(self):
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data = {}
        with h5py.File(self.data_path[0], 'r') as f:
            data_file = f['data']
            for event_idx in indexes: 
                event_name = list(self.event_metadata['KiK_File'])[int(event_idx)]
                g_event = data_file[event_name]
                for key in g_event:
                    if key not in data:
                        data[key] = []
                    data[key] += [g_event[key][()]]

        batch_p_picks = data['p_picks']
        batch_pga = data['pga']
        batch_waveforms = data['waveforms'] 
        batch_metadata = data['coords']
        
        true_batch_size = len(indexes) 
        waveforms = np.zeros((true_batch_size, self.current_station) + batch_waveforms[0].shape[1:])
        p_picks = np.zeros((true_batch_size, self.current_station))
        unuses = np.zeros((true_batch_size, 707))
        pga = np.zeros((true_batch_size, self.current_station))
        metadata = np.zeros((true_batch_size, self.current_station) + batch_metadata[0].shape[1:])

        # Find list of IDs    
        for i, idx in enumerate(indexes): 
            for staion_index in range(batch_waveforms[i].shape[0]): 
                station_key = f"{batch_metadata[i][staion_index][0]},{batch_metadata[i][staion_index][1]},{batch_metadata[i][staion_index][2]},{batch_metadata[i][staion_index][3]}"
                position = self.stations_table[station_key]
                waveforms[i,position] = batch_waveforms[i][staion_index]
                p_picks[i,position] = batch_p_picks[i][staion_index]
                pga[i,position] = batch_pga[i][staion_index] 
            metadata[i] = self.total_coords
            unuses[i] = self.unuse_table[idx] 
            
        org_waveform_length = waveforms.shape[2]
        if self.cutout:
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = np.random.randint(max(windowlen, self.cutout[0]), min(waveforms.shape[2], self.cutout[1]) + 1)
                waveforms = waveforms[:, :, window_end - windowlen: window_end]

                cutout = window_end
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms, axis=2, keepdims=True)
            else:
                cutout = np.random.randint(*self.cutout)  #cutout = 400, 3000
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms[:, :, :cutout+1], axis=2, keepdims=True) 
                waveforms[:, :, cutout:] = 0 
        else:
            cutout = waveforms.shape[2]

        if self.trigger_based:
            # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
            p_picks[p_picks <= 0] = org_waveform_length  
            waveforms[cutout < p_picks, :, :] = 0


        if self.integrate:
            waveforms = np.cumsum(waveforms, axis=2) / self.sampling_rate

        if self.station_blinding:
            mask = np.zeros(waveforms.shape[:2], dtype=bool)

            for i in range(waveforms.shape[0]):
                active = np.where((waveforms[i] != 0).any(axis=(1, 2)))[0]
                if len(active) == 0:
                    active = np.zeros(1, dtype=int)
                blind_length = np.random.randint(0, len(active))
                np.random.shuffle(active)
                blind = active[:blind_length]
                mask[i, blind] = True

            waveforms[mask] = 0
        # Avoid completely zero events, leading to NaN values in energy loss
        mask = np.logical_and((metadata == 0).all(axis=(1, 2)), (waveforms == 0).all(axis=(1, 2, 3)))
        waveforms[mask, 0, 0, 0] = 1e-9
        metadata[mask, 0, 0] = 1e-9
        
        waveforms = torch.from_numpy(waveforms.astype('float32'))
        metadata = torch.from_numpy(metadata.astype('float32'))
        unuses = torch.from_numpy(unuses)
        inputs = [waveforms, metadata, unuses]
        outputs = []

        pga[np.isinf(pga)] = 0.0  
        outputs = [torch.from_numpy(pga.astype('float32'))]

        return inputs, outputs
    
    
    def load_events(self, data_paths, min_mag, limit, first_station_appearance, last_station_appearance):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        if len(data_paths) > 1:
            raise NotImplementedError('Loading partitioned data is currently not supported')
        data_path = data_paths[0]

        event_metadata = pd.read_hdf(data_path, 'metadata/event_metadata')
        if min_mag is not None:
            event_metadata = event_metadata[event_metadata['M_J'] >= min_mag]
        for event_key in ['KiK_File']:
            if event_key in event_metadata.columns:
                break
                
        if limit:
            event_metadata = event_metadata.iloc[:limit]
        print('data_path',data_path)
        trace_filename = []
        unuse_table = []
        total_pga_dic = []
        
        with h5py.File(data_path, 'r') as f:
            for event_i, event in tqdm(event_metadata.iterrows(),total=len(event_metadata)):  
                event_name = str(event['KiK_File'])
                tmp_unuse_table = self.station_unuse_table(first_station_appearance, last_station_appearance, event_name[:14])
                if unuse_table==[]: unuse_table = [tmp_unuse_table]
                else: unuse_table.append(tmp_unuse_table)

                g_event = f['data'][event_name]
                
                reassigned_pga = g_event['pga'][()]
                total_pga_dic.append(reassigned_pga)
                
        # with open(f"tmp/{self.tag}_event_metadata.pkl", "wb") as f:
        #     pickle.dump(event_metadata, f)
        # with open(f"tmp/{self.tag}_unuse_table.pkl", "wb") as f:
        #     pickle.dump(unuse_table, f)
        # with open(f"tmp/{self.tag}_total_pga_dic.pkl", "wb") as f:
        #     pickle.dump(total_pga_dic, f)
        
        # with open(f"tmp/{self.tag}_event_metadata.pkl", "rb") as f:
        #     event_metadata = pickle.load(f)
        # with open(f"tmp/{self.tag}_unuse_table.pkl", "rb") as f:
        #     unuse_table = pickle.load(f)
        # with open(f"tmp/{self.tag}_total_pga_dic.pkl", "rb") as f:
        #     total_pga_dic = pickle.load(f)


        return event_metadata, unuse_table, total_pga_dic
    
    def station_unuse_table(self, first_station_appearance, last_station_appearance, event_time):
        unuse_table = np.ones(len(self.stations_table))

        if first_station_appearance and last_station_appearance:
            unuse_table = np.ones(len(last_station_appearance))
            
            event_t = UTCDateTime(event_time)
            for idx in range(len(last_station_appearance)):
                first_t = UTCDateTime(first_station_appearance[idx])
                last_t = UTCDateTime(last_station_appearance[idx])
                if event_t < first_t or event_t > last_t:
                    unuse_table[idx] = 0

        return unuse_table



class EvalGenerator(Dataset):
    def __init__(self, 
                 data, 
                 event_metadata, 
                 stations_table, 
                 stations_channel_boolean,
                 all_station=False,
                 key='MA', 
                 batch_size=32, 
                 cutout=None,
                 sliding_window=False, 
                 windowlen=3000, 
                 shuffle=True,
                 oversample=1, 
                 station_blinding=False,
                 magnitude_resampling=3,
                 adjust_mean=True, 
                 current_station=None, 
                 trigger_based=None, 
                 min_upsample_magnitude=2,
                 integrate=False, 
                 sampling_rate=100.,
                 pga_mode=False, 
                 coord_keys=None, 
                 upsample_high_station_events=None,
                 **kwargs):
        if kwargs:
            print(f'Unused parameters: {", ".join(kwargs.keys())}')
        self.pga_times = data['pga_times']
        self.pga = data['pga']
        self.pgv = data['pgv'] 
        self.waveforms = data['waveforms'] 
        self.metadata = data['coords']
        self.unuse_table = data['unuse_table']

        if 'p_picks' in data:
            self.p_picks = data['p_picks']  
        else:
            print('Found no picks')
            self.p_picks = [np.zeros(x.shape[0]) for x in self.waveforms]

        self.stations_channel_boolean = stations_channel_boolean
        self.stations_table = stations_table
        self.batch_size = batch_size 
        self.event_metadata = event_metadata
        
        self.cutout = cutout
        self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
        self.windowlen = windowlen  # Length of window for sliding window
        self.station_blinding = station_blinding
        self.adjust_mean = adjust_mean
        if current_station is None:
            current_station = self.waveforms.shape[1]
        self.current_station = current_station
        self.trigger_based = trigger_based
        self.integrate = integrate
        self.sampling_rate = sampling_rate

        # Extend samples to include all pga targets in each epoch
        # PGA mode is only for evaluation, as it adds zero padding to the input/pga target!
        self.pga_mode = pga_mode

        base_indexes = np.arange(len(self.event_metadata))
        reverse_index = None
        if magnitude_resampling > 1:
            magnitude = self.event_metadata[key].values
            for i in np.arange(min_upsample_magnitude, 9):
                ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                base_indexes = np.concatenate(
                    (base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))

        if upsample_high_station_events is not None:
            new_indexes = []
            for ind in base_indexes:
                n_stations = len(self.pga[ind])
                new_indexes += [ind for _ in range(n_stations // upsample_high_station_events + 1)]
            base_indexes = np.array(new_indexes)
        
        if pga_mode:  #evaluate.py在用的
            new_base_indexes = []
            reverse_index = []
            c = 0
            for idx in base_indexes: 
                # This produces an issue if there are 0 pga targets for an event.
                # As all input stations are always pga targets as well, this should not occur.
                num_samples = 1
                new_base_indexes += [(idx, i) for i in range(num_samples)]
                reverse_index += [c]
                c += num_samples
            reverse_index += [c]
            base_indexes = new_base_indexes

        if coord_keys is None:
            self.coord_keys = detect_location_keys(self.event_metadata.columns)
        else:
            self.coord_keys = coord_keys
        
        self.indexes = np.repeat(base_indexes.copy(), oversample, axis=0)
        if shuffle:
            np.random.shuffle(self.indexes)

        self.total_coords = np.zeros(((len(self.stations_table),4)))
        for coor_i, station_coord in enumerate(list(self.stations_table.keys())):
            self.total_coords[coor_i,:] = station_coord.split(',')[:]

    def __len__(self):
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index): 
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]        
        true_batch_size = len(indexes) 
        if self.pga_mode:
            pga_indexes = [x[1] for x in indexes] 
            indexes = [x[0] for x in indexes]  

        waveforms = np.zeros((true_batch_size, self.current_station) + self.waveforms[0].shape[1:])
        p_picks = np.zeros((true_batch_size, self.current_station))

        unuses = np.zeros((true_batch_size, 707))
        pga = np.zeros((true_batch_size, self.current_station))

        metadata = np.zeros((true_batch_size, self.current_station) + self.metadata[0].shape[1:])

        for i, idx in enumerate(indexes): 
            for staion_index in range(self.waveforms[idx].shape[0]): 
                station_key = f"{self.metadata[idx][staion_index][0]},{self.metadata[idx][staion_index][1]},{self.metadata[idx][staion_index][2]},{self.metadata[idx][staion_index][3]}"
                position = self.stations_table[station_key]
                waveforms[i,position] = self.waveforms[idx][staion_index]
                pga[i,position] = self.pga[idx][staion_index]
                p_picks[i,position] = self.p_picks[idx][staion_index]
            
            unuses[i] = self.unuse_table[idx]  
            metadata[i] = self.total_coords

        org_waveform_length = waveforms.shape[2]
        if self.cutout:
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = np.random.randint(max(windowlen, self.cutout[0]), min(waveforms.shape[2], self.cutout[1]) + 1)
                waveforms = waveforms[:, :, window_end - windowlen: window_end]

                cutout = window_end
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms, axis=2, keepdims=True)
            else:
                cutout = np.random.randint(*self.cutout)  #cutout = 400, 3000
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms[:, :, :cutout+1], axis=2, keepdims=True) #把隨機生成的時間點以前的波形再做一次校正 (應該是不會差到太多)
                waveforms[:, :, cutout:] = 0 
        else:
            cutout = waveforms.shape[2]
            
        # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
        p_picks[p_picks <= 0] = org_waveform_length 
        waveforms[cutout < p_picks, :, :] = 0

        if self.integrate:
            waveforms = np.cumsum(waveforms, axis=2) / self.sampling_rate

        if self.station_blinding:
            mask = np.zeros(waveforms.shape[:2], dtype=bool)

            for i in range(waveforms.shape[0]):
                active = np.where((waveforms[i] != 0).any(axis=(1, 2)))[0]
                if len(active) == 0:
                    active = np.zeros(1, dtype=int)
                blind_length = np.random.randint(0, len(active))
                np.random.shuffle(active)
                blind = active[:blind_length]
                mask[i, blind] = True

            waveforms[mask] = 0


        mask = np.logical_and((metadata == 0).all(axis=(1, 2)), (waveforms == 0).all(axis=(1, 2, 3)))
        waveforms[mask, 0, 0, 0] = 1e-9
        metadata[mask, 0, 0] = 1e-9

        waveforms = torch.from_numpy(waveforms.astype('float32'))
        metadata = torch.from_numpy(metadata.astype('float32'))
        pga = torch.from_numpy(pga.astype('float32'))
        unuses = torch.from_numpy(unuses)
        inputs = [waveforms, metadata, unuses]
        outputs = []
        outputs += [pga]

        return inputs, outputs


class CutoutGenerator(Dataset):
    def __init__(self, generator, times, sampling_rate):
        self.generator = generator
        self.times = times
        self.sampling_rate = sampling_rate
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return len(self.generator) * len(self.times)

    def __getitem__(self, index):
        time, batch_id = self.indexes[index]
        cutout = int(self.sampling_rate * (time + 5))
        self.generator.cutout = (cutout, cutout + 1)
        return self.generator[batch_id]

    def on_epoch_end(self):
        self.indexes = []
        for time in self.times:
            self.indexes += [(time, i) for i in range(len(self.generator))]
