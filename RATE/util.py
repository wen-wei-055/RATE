import h5py
import heapq
import random 
import numpy as np
import pandas as pd
import faiss
from obspy import UTCDateTime
from tqdm import tqdm
import json
import faiss
import torch
from torch.utils.data import Dataset

D2KM = 111.19492664455874

def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)


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
    def __init__(
        self, 
        datapath, 
        total_datapath, 
        train_val_boundary,
        datatype, 
        min_mag, 
        limit, 
        stations_table, 
        first_station_appearance, 
        last_station_appearance, 
        RAG_topK,
        historical_database_path,
        current_station,
        historical_station,
        peek_sample=2500,
        station_blinding=False, 
        cutout=None,
        key='MA', 
        batch_size=5, 
        shuffle=True,
        oversample=1, 
        magnitude_resampling=3,
        trigger_based=None, 
        min_upsample_magnitude=2,
        integrate=False, 
        sampling_rate=100.,
        p_pick_limit=5000, 
        coord_keys=None, 
        upsample_high_station_events=None,
        **kwargs
    ):
        if kwargs:
            print(f'Unused parameters: {", ".join(kwargs.keys())}')
        self.train_val_boundary = train_val_boundary
        self.datatype = datatype
        self.historical_database = np.load(historical_database_path)
        self.RAGindexes = self.InitialIndex(datatype, self.historical_database) 

        if datatype == 'Train':
            self.datatype_ID_diff = 0
        if datatype == 'Val':
            self.datatype_ID_diff = train_val_boundary
        self.topK_num = 2
        self.RAG_topK = RAG_topK
        self.total_datapath = total_datapath
        self.data_path = datapath
        self.stations_table = stations_table
        self.event_metadata, self.unuse_table, self.reassigned_pga = \
            self.load_events(
                datapath, 
                min_mag, 
                limit, 
                first_station_appearance, 
                last_station_appearance
            )
        self.total_event_metadata = pd.read_hdf(total_datapath[0], 'metadata/event_metadata')

        self.batch_size = batch_size 
        self.shuffle = shuffle
        
        self.key = key
        self.cutout = cutout
        self.peek_sample = peek_sample
        self.oversample = oversample 
        self.station_blinding = station_blinding
        self.left_station = current_station
        self.right_station = historical_station
        self.trigger_based = trigger_based
        self.integrate = integrate
        self.sampling_rate = sampling_rate
        self.upsample_high_station_events = upsample_high_station_events
        self.p_pick_limit = p_pick_limit

        self.base_indexes = np.arange(len(self.event_metadata))
        self.reverse_index = None
        if magnitude_resampling > 1:
            magnitude = self.event_metadata[key].values
            for i in np.arange(min_upsample_magnitude, 9):
                ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                self.base_indexes = np.concatenate(
                    (self.base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))

        if self.upsample_high_station_events is not None:
            new_indexes = []
            for ind in self.base_indexes:
                n_stations = len(self.reassigned_pga[ind])
                new_indexes += [ind for _ in range(n_stations // self.upsample_high_station_events + 1)]
            self.base_indexes = np.array(new_indexes)

        if coord_keys is None:
            self.coord_keys = detect_location_keys(self.event_metadata.columns)
        else:
            self.coord_keys = coord_keys
        
        self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.total_coords = np.zeros(((len(self.stations_table),4)))
        for coor_i, station_coord in enumerate(list(self.stations_table.keys())):
            self.total_coords[coor_i,:] = station_coord.split(',')[:]

    def __len__(self):
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):
        cutout = np.random.randint(*self.cutout)
        search_time = cutout // 100  
        
        # Generate indexes of the batch
        original_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        true_batch_size = len(original_indexes) 
        
        indexes_pairs = [] 
        original_data = {}
        pair_data = {}
        with h5py.File(self.total_datapath[0], 'r') as f:
            data_file = f['data']
            for event_idx in original_indexes: 
                event_idx = event_idx + self.datatype_ID_diff
                event_name = list(self.total_event_metadata['KiK_File'])[int(event_idx)]
                g_event = data_file[event_name]
                for key in g_event:
                    if key not in original_data:
                        original_data[key] = []
                    if key == 'p_picks':
                        tmp_pick = g_event[key][()]
                        original_data[key]+= [tmp_pick]
                        select_event = self.event_select(None, event_idx, search_time)
                        indexes_pairs.append(select_event) 
                    else:
                        original_data[key] += [g_event[key][()]]
            for indexes_pair in indexes_pairs:
                if indexes_pair == -1:
                    if 'p_picks' not in pair_data.keys():
                        pair_data['p_picks'] = [] 
                        pair_data['pga'] = []
                        pair_data['waveforms'] = []
                        pair_data['coords'] = []
                    pair_data['p_picks'] += [np.zeros((1,))]
                    pair_data['pga'] += [np.zeros((1,))]
                    pair_data['waveforms'] += [np.zeros((1,3000,3))]
                    pair_data['coords'] += [np.zeros((1,3))]
                else:
                    event_name = list(self.total_event_metadata['KiK_File'])[int(indexes_pair)]
                    g_event = data_file[event_name]
                    for key in g_event:
                        if key not in pair_data:
                            pair_data[key] = []
                        if key == 'p_picks':
                            tmp_pick = g_event[key][()]
                            tmp_pick -= np.min(tmp_pick) - 500
                            pair_data[key]+= [tmp_pick]
                        else:
                            pair_data[key] += [g_event[key][()]]

        datas = [original_data, pair_data]
        pairs = [original_indexes, indexes_pairs]
            
        total_waveforms = np.zeros((true_batch_size, self.left_station*self.topK_num, 3000, 6))
        total_pga = np.zeros((true_batch_size, self.right_station))
        left_unuse = np.zeros((true_batch_size, self.right_station))
        total_metadata = np.tile(self.total_coords, (true_batch_size,2,1))
        
        for ii, (indexes, data) in enumerate(zip(pairs,datas)):
            batch_p_picks = data['p_picks']
            batch_pga = data['pga']
            batch_waveforms = data['waveforms'] 
            batch_metadata = data['coords']
            
            waveforms = np.zeros((true_batch_size, self.left_station) + batch_waveforms[0].shape[1:])
            p_picks = np.zeros((true_batch_size, self.left_station))

            unuses = np.zeros((true_batch_size, 707))
            pga = np.zeros((true_batch_size, self.right_station))
            
            for i, idx in enumerate(indexes): 
                if idx == -1:
                    continue
                for staion_index in range(batch_waveforms[i].shape[0]):
                    station_key = f"{batch_metadata[i][staion_index][0]},{batch_metadata[i][staion_index][1]},{batch_metadata[i][staion_index][2]},{batch_metadata[i][staion_index][3]}"
                    position = self.stations_table[station_key]
                    waveforms[i,position] = batch_waveforms[i][staion_index]
                    if ii == 0:
                        pga[i,position] = batch_pga[i][staion_index] 
                    p_picks[i,position] = batch_p_picks[i][staion_index]
                if ii == 0:
                    unuses[i] = self.unuse_table[idx] 

            org_waveform_length = waveforms.shape[2]
            if ii == 0:
                waveforms[:, :, cutout:] = 0 
            else:
                waveforms[:, :, cutout+self.peek_sample:] = 0 

            if self.trigger_based:
                # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
                p_picks[p_picks <= 0] = org_waveform_length 
                waveforms[cutout < p_picks, :, :] = 0
                
            current_p_picks = p_picks[cutout > p_picks]

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

            if ii==0:
                total_waveforms[:,:self.left_station,:,:] = waveforms
                pga[np.isinf(pga)] = 0.0 
                total_pga[:,:self.left_station] = pga
                left_unuse = unuses
            else:
                total_waveforms[:,self.left_station:,:,:] = waveforms
        
        total_waveforms = torch.from_numpy(total_waveforms.astype('float32'))
        total_metadata = torch.from_numpy(total_metadata.astype('float32'))
        left_unuse = torch.from_numpy(left_unuse)
        inputs = [total_waveforms, total_metadata, left_unuse]
        outputs = [torch.from_numpy(total_pga.astype('float32'))]
        return inputs, outputs
    
    def InitialIndex(self, dtype, doc):        
        indexes = []
        for n in range(31):
            index = faiss.IndexFlatIP(doc.shape[-2])
            index.add(doc[:self.train_val_boundary,:,n])
            indexes.append(index)
        return indexes
    
    def event_select(self, district, EventID, query_time):
        query = np.expand_dims(self.historical_database[EventID,:,query_time], 0)
        _, I = self.RAGindexes[query_time].search(query, self.RAG_topK) 
        I = I[0]

        if self.RAG_topK > 1:
            np.random.shuffle(I)
            
        I = I[0]
        return I  

    
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
        unuse_table = []
        total_pga_dic = []
        
        with h5py.File(data_path, 'r') as f:
            for _, event in tqdm(event_metadata.iterrows(),total=len(event_metadata)):  
                event_name = str(event['KiK_File'])
                tmp_unuse_table = self.station_unuse_table(first_station_appearance, last_station_appearance, event_name[:14])
                if unuse_table==[]: unuse_table = [tmp_unuse_table]
                else: unuse_table.append(tmp_unuse_table)
                
                g_event = f['data'][event_name]
                reassigned_pga = g_event['pga'][()]
                total_pga_dic.append(reassigned_pga)
                        
        return event_metadata, unuse_table, total_pga_dic
    
    def station_unuse_table(self, first_station_appearance, last_station_appearance, event_time):
        unuse_table = np.ones(len(last_station_appearance))

        event_t = UTCDateTime(event_time)
        for idx in range(len(last_station_appearance)):
            first_t = UTCDateTime(first_station_appearance[idx])
            last_t = UTCDateTime(last_station_appearance[idx])
            if event_t < first_t or event_t > last_t:
                unuse_table[idx] = 0

        return unuse_table
    

class EvalGenerator(Dataset):
    def __init__(
        self, 
        event_id, 
        data, 
        data_total, 
        train_val_boundary,
        event_metadata,
        stations_table,
        validation_set, 
        current_station,
        historical_station,
        historical_database_path,
        peek_sample=2500, 
        all_station=False,
        key='MA', 
        batch_size=32,
        cutout=None,
        shuffle=True,
        oversample=1, 
        trigger_based=None, 
        min_upsample_magnitude=2,
        integrate=False, 
        sampling_rate=100.,
        p_pick_limit=5000, 
        coord_keys=None, 
        **kwargs
    ):
        if kwargs:
            print(f'Unused parameters: {", ".join(kwargs.keys())}')
        self.train_val_boundary = train_val_boundary
        self.topK_num = 2
        self.validation_set = validation_set
        self.historical_database = np.load(historical_database_path)
        self.RAGindexes = self.InitialIndex(self.historical_database) 

        self.total_waveforms = data_total['waveforms']
        self.total_p_picks = data_total['p_picks'] 
        self.total_station_coords = data_total['coords']
        self.total_unuse_table = data_total['unuse_table']
        self.total_pga = data_total['pga']
        self.event_id = event_id
        self.peek_sample = peek_sample

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


        self.stations_table = stations_table
        self.batch_size = 1 
        self.shuffle = shuffle
        self.event_metadata = event_metadata
        
        self.key = key
        self.cutout = cutout
        self.oversample = oversample  
        self.left_station = current_station
        self.right_station = historical_station
        self.trigger_based = trigger_based
        self.integrate = integrate
        self.sampling_rate = sampling_rate

        self.p_pick_limit = p_pick_limit
        
        self.base_indexes = np.arange(len(self.event_metadata))
        self.reverse_index = None

        new_base_indexes = []
        self.reverse_index = []
        c = 0
        for idx in self.base_indexes: 
            num_samples = 1
            new_base_indexes += [(idx, i) for i in range(num_samples)]
            self.reverse_index += [c]
            c += num_samples
        self.reverse_index += [c]
        self.base_indexes = new_base_indexes

        if coord_keys is None: 
            self.coord_keys = detect_location_keys(self.event_metadata.columns)
        else:
            self.coord_keys = coord_keys

        self.total_coords = np.zeros(((len(self.stations_table),4)))
        for coor_i, station_coord in enumerate(list(self.stations_table.keys())):
            self.total_coords[coor_i,:] = station_coord.split(',')[:]

        self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)

    def __len__(self):
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):
        cutout = np.random.randint(*self.cutout)
        search_time = cutout // 100
        select_event, topK_list = self.event_select(None, self.event_id, search_time)
        if select_event == -1:
            pair_waveforms = np.zeros(self.total_waveforms[self.event_id].shape)
            pair_metadata = np.zeros(self.total_station_coords[self.event_id].shape)
            pair_picks = np.zeros(self.total_p_picks[self.event_id].shape)
            
        else:
            pair_waveforms = self.total_waveforms[select_event]
            pair_metadata = self.total_station_coords[select_event]
            pair_picks = self.total_p_picks[select_event]
            pair_pga = self.total_pga[select_event]
        
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]        
        true_batch_size = len(indexes)  

        pga_indexes = [x[1] for x in indexes] 
        indexes = [x[0] for x in indexes] 

        datas = [self.waveforms, self.metadata, self.p_picks, self.unuse_table, self.pga]
        pairs = [[pair_waveforms], [pair_metadata], [pair_picks], None, [pair_pga]]

        iter_list = [datas, pairs]
        iter_index_list = [indexes, [select_event]]

        total_waveforms = np.zeros((true_batch_size, self.left_station*self.topK_num, 3000, 6))
        unuses = np.zeros((true_batch_size, self.left_station))
        total_metadata = np.tile(self.total_coords, (true_batch_size,2,1))
        pga = np.zeros((true_batch_size, self.left_station))

        
        for ii, (indexes,data) in enumerate(zip(iter_index_list, iter_list)):
            
            tmp_waveforms = data[0]
            tmp_metadata = data[1]
            tmp_p_picks = data[2]
            tmp_unuse_table = data[3]
            tmp_pga = data[4]
            
            waveforms = np.zeros((true_batch_size, self.left_station) + self.waveforms[0].shape[1:])
            p_picks = np.zeros((true_batch_size, self.left_station))

            for i, idx in enumerate(indexes): 
                if idx == -1:
                    continue
                for staion_index in range(tmp_waveforms[i].shape[0]): 
                    station_key = f"{tmp_metadata[0][staion_index][0]},{tmp_metadata[0][staion_index][1]},{tmp_metadata[0][staion_index][2]},{tmp_metadata[0][staion_index][3]}"
                    position = self.stations_table[station_key]
                    waveforms[i,position] = tmp_waveforms[i][staion_index]
                    if ii == 0:
                        pga[i,position] = tmp_pga[idx][staion_index]
                    p_picks[i,position] = tmp_p_picks[i][staion_index]
                
                if ii == 0:
                    unuses[i] = tmp_unuse_table[idx]  
                    
            org_waveform_length = waveforms.shape[2]
            
            if ii == 0:
                waveforms[:, :, cutout:] = 0 
            else:
                waveforms[:, :, cutout+self.peek_sample:] = 0

            if self.trigger_based:
                # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
                p_picks[p_picks <= 0] = org_waveform_length
                waveforms[cutout < p_picks, :, :] = 0

            if ii==0:
                total_waveforms[:,:self.left_station,:,:] = waveforms
            else:
                total_waveforms[:,self.left_station:,:,:] = waveforms
                
        total_waveforms = torch.from_numpy(total_waveforms.astype('float32'))
        total_metadata = torch.from_numpy(total_metadata.astype('float32'))
        pga = torch.from_numpy(pga.astype('float32'))
        unuses = torch.from_numpy(unuses)
        inputs = [total_waveforms, total_metadata, unuses]
        outputs = [pga, topK_list]
        return inputs, outputs
    
    def InitialIndex(self, doc):
        indexes = []
        for n in range(31):
            index = faiss.IndexFlatIP(doc.shape[-2])
            index.add(doc[:self.train_val_boundary,:,n])
            indexes.append(index)
        return indexes
    
    def event_select(self, district, EventID, query_time):
        query = np.expand_dims(self.historical_database[EventID,:,query_time], 0)
        _, I = self.RAGindexes[query_time].search(query, 10) 
        I_topK = np.array(I[0])
        I = I[0][0]
        return I, I_topK   


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

