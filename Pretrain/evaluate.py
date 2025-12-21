import argparse
import os
import json
import pickle
import h5py
import torch
import models
import loader
import util
import numpy as np
import pandas as pd

from scipy.stats import norm
from tqdm import tqdm
from geopy.distance import geodesic
from models import EvaluateModel


def load_json_if_exists(path):
    if path and os.path.isfile(path):
        with open(path) as f:
            return list(json.load(f).values())
    return None


def calculate_warning_times(config, 
                            model_list, 
                            data, 
                            event_metadata, 
                            batch_size, 
                            max_stations,
                            sampling_rate=100,
                            times=np.arange(0.5, 25, 0.8), 
                            alpha=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 
                            use_multiprocessing=False,
                            device='cuda'):
    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)

    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['upsample_high_station_events'] = None

    alpha = np.array(alpha)
    
    f = h5py.File(data_path[0], 'r')
    g_data = f['data']
    thresholds = f['metadata']['pga_thresholds'][()]
    time_before = f['metadata']['time_before'][()]

    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')

    event_key = 'KiK_File'

    full_predictions = []
    no_unuse_full_predictions = []
    coord_keys = util.detect_location_keys(event_metadata.columns)

    latlon_IDtable = latlondep_ID(stations_table)

    stations_channel_boolean = [1 for _ in list(stations_table.keys())]
    stations_channel_class = [1 for _ in list(stations_table.keys())]


    for i, _ in tqdm(enumerate(event_metadata.iterrows()), total=len(event_metadata)):
        event = event_metadata.iloc[i]
        event_metadata_tmp = event_metadata.iloc[i:i+1]
        data_tmp = {key: val[i:i+1] for key, val in data.items()}
        generator_params['translate'] = False
        # n_pga_targets = 249
        generator = util.EvalGenerator(data=data_tmp,
                                                 stations_table=stations_table,
                                                 event_metadata=event_metadata_tmp,
                                                 coords_target=True,
                                                 cutout=(0, 3000),
                                                #  pga_targets=n_pga_targets,
                                                 current_station=max_stations,
                                                 sampling_rate=sampling_rate,
                                                 select_first=True,
                                                 shuffle=False,
                                                 pga_mode=True,
                                                 stations_channel_boolean=stations_channel_boolean,
                                                 stations_channel_class=stations_channel_class,
                                                 latlon_IDtable=latlon_IDtable,
                                                 **generator_params)

        cutout_generator = util.CutoutGenerator(generator, times, sampling_rate=sampling_rate)

        # Assume PGA output at index 2
        if use_multiprocessing:
            workers = 0
            
        prediction, unuse_list = model_list.predict_generator(cutout_generator, workers=0, use_multiprocessing=False)  # (時間, (trace(超過20測站的trace會有>1個trace), 20, 50, 3))
    
        prediction = prediction.reshape((len(times), -1) + prediction.shape[2:])
        
        pga_pred = torch.Tensor(prediction)        
        pga_times_pre = np.zeros((pga_pred.shape[1], thresholds.shape[0], alpha.shape[0]), dtype=int)

        alpha = torch.Tensor(alpha)
        for j, level in enumerate(np.log10(thresholds*9.81)):
            
            prob = torch.sum(
                pga_pred[:, :, :, 0] * (1 - norm.cdf((level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                dim=-1)
            
            prob = torch.unsqueeze(prob, -1)
            
            exceedance = torch.gt(prob,alpha)  # Shape: times, stations, 1
            exceedance = np.pad(exceedance, ((1, 0), (0, 0), (0, 0)), mode='constant')
            pga_times_pre[:, j] = np.argmax(exceedance, axis=0)

        pga_times_pre -= 1
        pga_times_pred = np.zeros_like(pga_times_pre, dtype=float)
        pga_times_pred[pga_times_pre == -1] = np.nan
        pga_times_pred[pga_times_pre > -1] = times[pga_times_pre[pga_times_pre > -1]]  # 每0.2秒進入模型一次，如果有超過pga閾值就放入pga_times_pred

        g_event = g_data[str(event[event_key])]
        pga_times_true_pre = g_event['pga_times'][()]
        coords = g_event['coords'][()]
        coords_event = event[coord_keys]
        pick_offset = np.min(g_event['p_picks'][()])
        
        pga_times_true = np.zeros(pga_times_pred.shape[:2], dtype=float)
        for station_index in range(coords.shape[0]):
            station = coords[station_index]
            station_key = '{},{},{},{}'.format(station[0], station[1], station[2], station[3])
            position = stations_table[station_key]  
            pga_times_true[position] = pga_times_true_pre[station_index]
        
        pga_times_true[pga_times_true == 0] = np.nan
        pga_times_true[pga_times_true != 0] = (pga_times_true[pga_times_true != 0]) / sampling_rate - time_before  #在座資料集的時候有 + time_before
        
        coords = (np.array([[float(x) for x in row] for row in [x.split(',')[:-1] for x in stations_table]]))
        dist = np.zeros(coords.shape[0])
        for j, station_coords in enumerate(coords):
            dist[j] = geodesic(station_coords[:2], coords_event[:2]).km
        dist = np.sqrt(dist ** 2 + coords_event[2] ** 2)
        full_predictions += [(pga_times_pred, pga_times_true, dist)]

    return full_predictions
    
def latlondep_ID(stations_table):
    appear_first = {}
    appear_dic = {}
    i = 0
    for id_name, instrument_name in enumerate(stations_table):
        latlondep = ','.join(instrument_name.split(',')[:3])
        if latlondep not in appear_first.keys():
            appear_first[latlondep] = i
            appear_dic[id_name] = i
            i += 1
        if latlondep in appear_first.keys():
            appear_dic[id_name] = appear_first[latlondep]
    return appear_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--weight_file', type=str)  # If unset use latest model
    parser.add_argument('--times', type=str, default='0.5,1,2,4,8,16,25')  # Has only performance implications
    parser.add_argument('--max_stations', type=int)  # Overwrite max stations value from config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val', action='store_true')  # Evaluate on val set
    parser.add_argument('--blind_time', type=float, default=0.5)  # Time of first evaluation after first P arrival
    parser.add_argument('--alpha', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')  # 機率大於alpha以上就會發布警報 Probability thresholds alpha
    parser.add_argument('--additional_data', type=str)  # Additional data set to use for evaluation
    parser.add_argument('--loss_limit', type=float) # In ensemble model, discard members with loss above this limit
    # A combination of tensorflow multiprocessing for generators and pandas dataframes causes the code to deadlock
    # sometimes. This flag provides a workaround.
    parser.add_argument('--no_multiprocessing', action='store_true')
    parser.add_argument('--first_station_appearance_path', type=str)  # Station activation date
    parser.add_argument('--last_station_appearance_path', type=str)   # Station deactivation date
    args = parser.parse_args()
    
    torch.set_num_threads(5)

    times = [float(x) for x in args.times.split(',')]

    config = json.load(open(os.path.join(args.experiment_path, 'config.json'), 'r'))
    training_params = config['training_params']

    stations_table = json.load(open(training_params['station_json_file'], 'r'))
    max_stations = args.max_stations or config['model_params'].get('current_station')

    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")
    generator_params = training_params.get('generator_params', [training_params.copy()])[0]
    n_datasets = 1
        
    batch_size = generator_params['batch_size']
    key = generator_params.get('key', 'MA')
    pga_key = generator_params.get('pga_key', 'pga')

    if args.blind_time != 0.5:
        suffix = f'_blind{args.blind_time:.1f}'
    else:
        suffix = ''

    if args.val:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'val')
        data_path = training_params['val_data_path']
        test_set = False
    else:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'test')
        data_path = training_params['test_data_path']
        test_set = True

    if not os.path.isdir(os.path.join(args.experiment_path, f'evaluation{suffix}')):
        os.mkdir(os.path.join(args.experiment_path, f'evaluation{suffix}'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    first_station_appearance = load_json_if_exists(args.first_station_appearance_path)
    last_station_appearance = load_json_if_exists(args.last_station_appearance_path)

    custom_split = generator_params.get('custom_split', None)
    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_mag = generator_params.get('min_mag', None)
    mag_key = generator_params.get('key', 'MA')
    event_metadata, data, metadata = loader.load_events(data_path,
                                                        # limit=10,
                                                        station_len=len(stations_table),
                                                        custom_split=custom_split,
                                                        min_mag=min_mag,
                                                        mag_key=mag_key,
                                                        overwrite_sampling_rate=overwrite_sampling_rate,
                                                        first_station_appearance=first_station_appearance,
                                                        last_station_appearance=last_station_appearance)

    if args.additional_data:
        print('Loading additional data')
        event_metadata_add, data_add, _ = loader.load_events(args.additional_data,
                                                             parts=(True, True, True),
                                                             min_mag=min_mag,
                                                             mag_key=mag_key,
                                                             overwrite_sampling_rate=overwrite_sampling_rate)
        event_metadata = pd.concat([event_metadata, event_metadata_add])
        for t_key in data.keys():
            if t_key in data_add:
                data[t_key] += data_add[t_key]

    if pga_key in data:
        pga_true = data[pga_key]
    else:
        pga_true = None

    
    ensemble = config.get('ensemble', 1)
    print(ensemble)
    experiment_path = args.experiment_path
    model_list = EvaluateModel(config, 
                                experiment_path, 
                                weight_file=experiment_path+'/checkpoint_{}.pth'.format(args.weight_file), 
                                loss_limit=args.loss_limit, 
                                device=device)

    pga_stats = []
    pga_pred_full = []
    
    results = {'times': times,
               'pga_stats': np.array(pga_stats).tolist()}

    with open(os.path.join(output_dir, 'stats.json'), 'w') as stats_file:
        json.dump(results, stats_file, indent=4)

    times_pga = np.arange(args.blind_time, 25, 0.8)
    alpha = [float(x) for x in args.alpha.split(',')]
    warning_time_information = calculate_warning_times(config, 
                                                        model_list, 
                                                        data, 
                                                        event_metadata,
                                                        max_stations=max_stations,
                                                        times=times_pga,
                                                        alpha=alpha,
                                                        batch_size=batch_size,
                                                        use_multiprocessing=not args.no_multiprocessing,
                                                        device=device)


    with open(os.path.join(output_dir, f'{args.weight_file}_warning.pkl'), 'wb') as pred_file:
        pickle.dump((times, pga_pred_full, warning_time_information, alpha), pred_file)
