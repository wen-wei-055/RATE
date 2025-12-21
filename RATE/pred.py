import loader
import pickle
import numpy as np
np.set_printoptions(suppress=True)
import json
import torch
from torch.utils.data import DataLoader
from scipy.stats import norm
# from models import EnsembleEvaluateModel
import models
import util
import os
import argparse
import itertools
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)          # 導引到放pkl檔的所屬資料夾
    parser.add_argument('--weight_num', type=str)
    parser.add_argument('--experiment_path', type=str)              # main_pred用，weight_path，同evaluate
    parser.add_argument('--main_pred', action='store_true')         # 計算pred結果 --> /txt_file
    parser.add_argument('--draw_img', action='store_true')          # 畫多類別png檔 --> /confusion_matrix
    parser.add_argument('--metrics', action='store_true')           # 計算acc,pre,rec,F1 --> /metrics
    parser.add_argument('--TEAM_thresholds', type=str, default=True)   # 使用TEAM paper的threshold (預設是用thresholds變數的閾值)
    parser.add_argument('--analyze_dataset', action='store_true')   # 分析資料檔
    parser.add_argument('--detect_only', action='store_true')   # 分析資料檔
    
    parser.add_argument('--time_pred', type=str, default=True)         # 預測預警時間
    args = parser.parse_args()

    time_list = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 25.0]  # 挑最後一個時間軸(t=25.0) 
    thresholds = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    thresholds_def = ['1% ', '2% ', '5% ', '10%', '20%']  #輸出的類別，要比thresholds數量多一，放第一項
    
    if args.TEAM_thresholds:
        thresholds = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
        thresholds_def = ['<1%', '1% ', '2% ', '5% ', '10%', '20%']

    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    magnitude_class = ['[mag]3_less','[mag]3-4','[mag]4-5','[mag]5-6','[mag]6_more']  # 5個規模分類 (如果要修改這項，要連帶改--confusion_matrix的最後)


    path = '/'.join(args.path.split('/')[:-1])
    filename = (args.path.split('/')[-1]).split('.')[0]

    if args.time_pred:
        if args.detect_only:
            
            stations_table = json.load(open('dataset_configs/488_stations_category.json', 'r', encoding='utf-8'))
            sampling_rate = 100
            
            _, data, _ = loader.load_events("/mnt/nas5/william/Dataset/TEAM/taiwan/CWBSN/PGA_PGV_location_oriented/test_CWBSN_window.hdf5",
                                                        shuffle_train_dev=False,
                                                        custom_split=None,
                                                        min_mag=None,
                                                        mag_key='Magnitude',
                                                        overwrite_sampling_rate=None,
                                                        first_appearance_list=None,
                                                        last_appearance_list=None)
            trace_filename = data['trace_filename']
            coords = data['coords']
            filename += '_partial_station'

        if args.weight_num:
            with open(args.path+'/predictions_{}.pkl'.format(args.weight_num), 'rb') as f:   # [(pga_times_pred, pga_times_true, dist)]
                pre_pga_pred = pickle.load(f)  # [..., ..., ..., ..., [事件數, [(測站數, 5, 5), (測站數, 5), (測站數,)]]]
        else:
            with open(args.path, 'rb') as f:   # [(pga_times_pred, pga_times_true, dist)]
                pre_pga_pred = pickle.load(f)  # [..., ..., ..., ..., [事件數, [(測站數, 5, 5), (測站數, 5), (測站數,)]]]
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dist_list = [0, 50, 100, 200, 300]
        dist_def = ['0-50 km', '50-100 km', '100-200 km', '200-300 km', '300 more km', 'total']
        thresholds_def = ['1% ', '2% ', '5% ', '10%', '20%']
                
        total_cube = np.zeros((len(dist_def),9,4,len(thresholds_def)))  #(距離數量+1(總和), alpha, 4個指標, 6個threshold到時)  指標:tp, fp, tn, fn
        perfect_cube = np.zeros((len(thresholds_def), len(alpha_list), len(dist_def), 2)) # (threshold, alpha, 距離, 完美指標) 完美預測: tp, tn
        leading_time_cube = np.zeros((len(thresholds_def), len(alpha_list), len(dist_def), 2))  # (threshold, alpha, 距離, 2)，裝mean_leading_time, variance_leading_time
        # perfect_leading_time_cube = np.zeros((len(pre_pga_pred[4]), pre_pga_pred[4][0][0].shape[0], len(thresholds_def)))  # (事件數, 測站數, threshold)
        # leading_time_list = [[[]]*len(alpha_list) for _ in range(len(thresholds_def))]  # (threshold, alpha)
        leading_time_list = [[[[]] * (len(dist_def)) for _ in range(len(alpha_list))] for _ in range(len(thresholds_def))] # (threshold, alpha, 距離)
        for event_idx, event_pack in tqdm(enumerate(pre_pga_pred[2])):  # for每個事件
            pga_time_pred = event_pack[0]   # (249, 6, 9)
            # print('pga_time_pred',pga_time_pred.shape)
            pga_times_true = event_pack[1]  # (249, 6)
            # print('pga_times_true',pga_times_true.shape)
            dist = event_pack[2]
            dist = dist.reshape(dist.shape + (1,))

            # perfect_leading_time_cube[event_idx] = pga_times_true
            
            if args.detect_only:
                selection = []
                for station_idx in range(len(trace_filename[event_idx])):
                    channel_name = (str(trace_filename[event_idx][station_idx]).split('_'))[2]
                    station_key = f"{coords[event_idx][station_idx][0]},{coords[event_idx][station_idx][1]},{coords[event_idx][station_idx][2]},{channel_name}"
                    position = stations_table[station_key]
                    if position < 239: continue
                    else:
                        position -= 239
                        selection.append(position)
                pga_time_pred = pga_time_pred[selection]
                pga_times_true = pga_times_true[selection]
                dist = dist[selection]

            for alpha_idx in range(pga_time_pred.shape[2]):
                pre_matrix = np.zeros((pga_time_pred.shape[0], pga_time_pred.shape[1]))-99.9 #(測站數, 6個threshold到時) 裝4個指標的編號(tp:0, fp:1, tn:2, fn:3)
                
                pred = pga_time_pred[:,:,alpha_idx].copy() #(測站數, 6個threshold到時)
                true = pga_times_true.copy()               #(測站數, 6個threshold到時)

                nan_pred = pred.copy()                     #(測站數, 6個threshold到時)
                nan_true = pga_times_true.copy()           #(測站數, 6個threshold到時)

                #TP:0  FP:1  TN:2  FN:3
                #這個指標只有計算有達threshold的測站，也就是說假如true有值，預估值只要>該threshold都會算是正確TP
                ################## 處理皆有值(其中一個有nan的不管怎麼樣結果都是nan) ##################
                nan_true -= nan_pred  
                nan_true[nan_true >= 0] = 0  #TP
                nan_true[nan_true < 0] = 3   #預測時間超標
                pre_matrix[np.logical_not(np.isnan(nan_true))] = nan_true[np.logical_not(np.isnan(nan_true))]  #複製到pre_matrix

                ################## 處理nan ##################
                pre_matrix[np.isnan(pred)] = 3  #pred有nan的先設3
                pre_matrix[np.isnan(true)] = 1  #true有nan得先設1
                pre_matrix[np.logical_and(np.isnan(pred), np.isnan(true))] = 2  #pred, true都是nan的設2
                pre_matrix = pre_matrix.astype(int)
                
                dist[dist<0] = 0
                dist_exceedance = dist > dist_list
                exceedance_idx = np.sum(dist_exceedance, axis=1)-1  #得到每個測站離震央的距離在dist_def的哪個範圍
                # print('exceedance_idx',exceedance_idx)
                
                #計算leading_time
                # leading_time_cube
                for threshold_idx in range(pred.shape[1]):
                    TP_time_pred = (pred[:,threshold_idx])[pre_matrix[:,threshold_idx]==0]
                    TP_time_true = (true[:,threshold_idx])[pre_matrix[:,threshold_idx]==0]
                    sub_list = TP_time_true - TP_time_pred
                    zero_indices = np.where(pre_matrix[:,threshold_idx] == 0)[0]
                    
                    if len(sub_list) != 0:
                        # leading_time_list[threshold_idx][alpha_idx] += (TP_time_true - TP_time_pred).tolist()
                        
                        # for station_idx, dist_idx in enumerate(exceedance_idx):    #for所有距離
                        for order_id, station_idx in enumerate(zero_indices):
                            sub_value = sub_list[order_id]
                            tmp_list = list(leading_time_list[threshold_idx][alpha_idx][exceedance_idx[station_idx]])  #python要先抓出來，不然很奇怪會擴展維度
                            tmp_total_list = list(leading_time_list[threshold_idx][alpha_idx][-1])
                            
                            tmp_list.append(sub_value)
                            tmp_total_list.append(sub_value)
                            
                            leading_time_list[threshold_idx][alpha_idx][exceedance_idx[station_idx]] = tmp_list
                            leading_time_list[threshold_idx][alpha_idx][-1] = tmp_total_list
                # 另外一個matrix
                true[~np.isnan(true)] = 0 #tp
                true[np.isnan(true)] = 1 #tn
                true = true.astype(int)
                
                
                for station_idx, dist_idx in enumerate(exceedance_idx):    #for所有距離
                    for i, count_i in enumerate(pre_matrix[station_idx]):  #for所有測站
                        total_cube[dist_idx, alpha_idx, count_i, i] += 1  ##(距離數量+1(總和), 5個alpha, 4個指標, 6個threshold到時) 
                        total_cube[-1, alpha_idx, count_i, i] += 1
                    true_array = true[station_idx]  #(5個threshold,)
                    for t_idx, perfect_v in enumerate(true_array):
                        perfect_cube[t_idx, alpha_idx, dist_idx, perfect_v] += 1
            # if event_idx == 50:
            #     break 
        
        # # (事件數, 測站數, threshold) -> (事件數*測站數, threshold)
        # perfect_leading_time_cube = perfect_leading_time_cube.reshape(perfect_leading_time_cube.shape[0]*perfect_leading_time_cube.shape[1], perfect_leading_time_cube.shape[2])  
        # perfect_leading_time_cube = np.nanmean(perfect_leading_time_cube, 0)

        for ti in range(len(thresholds_def)):
            for ai in range(len(alpha_list)):
                for di in range(len(dist_def)):
                    if len(leading_time_list[ti][ai][di]) == 0: 
                        mean = np.nan
                        var = np.nan
                    else:
                        mean = np.mean(np.array(leading_time_list[ti][ai][di]))
                        var = np.var(np.array(leading_time_list[ti][ai][di]))
                        leading_time_cube[ti, ai, di, 0] = mean  #(threshold, alpha)，裝leading_time
                        leading_time_cube[ti, ai, di, 1] = var #(threshold, alpha)，裝leading_time
        del leading_time_list

                
        perfect_cube[:,:,-1,:] = np.sum(perfect_cube, axis=2)

        dir = path + '/{}/time_total/固定閾值軸'.format(filename)
        if not os.path.isdir(dir):
            os.makedirs(dir)                             #(距離數量+1(總和), 5個alpha, 4個指標, 6個threshold到時)

        ########################################################################################################################
        with open(dir+'/[混淆矩陣指標]完美預測.txt', 'w', encoding='utf-8') as f:
            f.write('threshold列表: '+str(thresholds_def)+'\n')
            f.write('alpha列表: '+str(alpha_list)+'\n')
            f.write('橫軸: tp, tn\n')
            f.write('縱軸: '+str(dist_def)+'\n\n')
            for thres_idx, thres_i in enumerate(perfect_cube): 
                f.write('======================== threshold:{} ======================== '.format(thresholds_def[thres_idx])+'\n')
                for alpha_idx, alpha_i in enumerate(thres_i):
                    f.write('alpha: '+str(alpha_list[alpha_idx])+'\n')
                    f.write(str(alpha_i.round(5))+'\n\n')
        
        ########################################################################################################################
        print_cube = np.transpose(total_cube.copy(),(3,1,0,2))  #(6個threshold到時, 5個alpha, 距離數量+1(總和), 4個指標)
        #(距離數量+1(總和), alpha, 4個指標, 6個threshold到時)  指標:tp, fp, tn, fn
        for thres_idx, thres_i in enumerate(print_cube): 
            with open(dir+'/[混淆矩陣指標]{}.txt'.format(thresholds_def[thres_idx]), 'w', encoding='utf-8') as f:
                f.write('alpha列表: '+str(alpha_list)+'\n')
                f.write('橫軸: tp, fp, tn, fn\n')
                f.write('縱軸: '+str(dist_def)+'\n\n')
                for alpha_idx, alpha_i in enumerate(thres_i):
                    f.write('alpha: '+str(alpha_list[alpha_idx])+'\n')
                    f.write(str(alpha_i.round(5))+'\n\n')
                    
        ########################################################################################################################
        for thres_idx, thres_i in enumerate(leading_time_cube):  #(threshold, alpha, 距離, 2)
            with open(dir+'/Leading_time_{}.txt'.format(thresholds_def[thres_idx]), 'w', encoding='utf-8') as f:
                f.write('alpha列表: '+str(alpha_list)+'\n')
                f.write('橫軸: mean_leading_time, variance_leading_time\n')
                f.write('縱軸: '+str(dist_def)+'\n\n')
                for alpha_idx, alpha_i in enumerate(thres_i):
                    f.write('alpha: '+str(alpha_list[alpha_idx])+'\n')
                    f.write(str(alpha_i)+'\n\n')
                    
        # with open(dir+'/Perfect_Leading_time.txt', 'w', encoding='utf-8') as f:
        #     f.write(f'threshold: {thresholds_def}\n')
        #     f.write(f'{perfect_leading_time_cube}')

        ########################################################################################################################
        # with open(dir+'/leading_time.txt'.format(thresholds_def[thres_idx]), 'w', encoding='utf-8') as f:
        #     f.write('threshold列表: '+str(thres_idx)+'\n')
        #     f.write('橫軸: mean_leading_time, variance_leading_time\n')
        #     f.write('縱軸: '+str(alpha_list)+'\n\n')
        #     for thres_idx, thres_i in enumerate(leading_time_cube): 
        #         f.write('threshold: '+str(thresholds_def[thres_idx])+'\n')
        #         f.write(str(thres_i)+'\n\n')
        
        ########################################################################################################################
        metric_cube = np.zeros((len(dist_def),9,4,len(thresholds_def)))  #(距離數量+1(總和), 5個alpha, 3+1個指標, 6個threshold到時)  指標:precision, recall, F1, 數量
                                                                #total_cube: (距離數量+1(總和), 5個alpha, 4個指標, 6個threshold到時)
        tp = total_cube[:, :, 0, :]
        fp = total_cube[:, :, 1, :]
        tn = total_cube[:, :, 2, :]
        fn = total_cube[:, :, 3, :]
        precision = tp / (tp+fp)
        precision[np.isnan(precision)] = 0.0
        recall = tp / (tp+fn)
        recall[np.isnan(recall)] = 0.0
        F1 = (2*precision*recall)/(precision+recall)
        F1[np.isnan(F1)] = 0.0
        print('F1',F1)
        print('F1',F1.shape)

        metric_cube[:, :,0,:] = precision
        metric_cube[:, :,1,:] = recall
        metric_cube[:, :,2,:] = F1
        metric_cube[:, :,3,:] = np.sum(total_cube, axis=2)
        print('metric_cube',metric_cube[:, :,2,:])
             
        ########################################################################################################################
        dir = path + '/{}/time_total'.format(filename)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        with open(dir+'/pred.txt', 'w', encoding='utf-8', newline='') as f:
            f.write('alpha列表: '+str(alpha_list)+'\n')
            f.write('橫軸: precision, recall, F1, alpha, leading_time_mean, leading_time_variance, 測站數量\n')
            f.write('縱軸: '+str(thresholds_def)+'\n\n')

            for dist_idx in range(len(dist_def)):
                best_alpha_matrix = np.zeros((len(thresholds_def), 7))  #(6個threshold到時, 5個指標) 指標:precision, recall, F1, alpha, leading_time_mean, leading_time_variance, 數量
                argmax = np.argmax(metric_cube[dist_idx, :, -2, :], axis=0)  #(距離數量+1(總和), 5個alpha, 4個指標, 6個threshold到時)
                for threshold_idx, best_alpha_idx in enumerate(argmax):
                    best_alpha_matrix[threshold_idx, :-4] = metric_cube[dist_idx, best_alpha_idx, :3, threshold_idx]
                    best_alpha_matrix[threshold_idx, -1] = metric_cube[dist_idx, best_alpha_idx, -1, threshold_idx]
                    best_alpha_matrix[threshold_idx, -3:-1] = leading_time_cube[threshold_idx, best_alpha_idx, dist_idx]
                    best_alpha_matrix[threshold_idx, -4] = alpha_list[best_alpha_idx]

                print(dist_def[dist_idx])
                print(best_alpha_matrix.round(5))

                f.write('測站距離震央: {}\n'.format(dist_def[dist_idx]))
                for row in best_alpha_matrix.round(5):
                    row_str = ', '.join([f"{val:10.5f}" for val in row])
                    f.write(row_str + '\n')
                f.write('\n\n')
                
                
        ########################################################################################################################
        dir = path + '/{}/time_total/固定距離軸'.format(filename)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        metric_cube = np.transpose(metric_cube,(0,1,3,2))  #(距離數量+1(總和), 5個alpha, 6個threshold到時, 3+1個指標)
        for dist_idx, dist_i in enumerate(metric_cube): 
            with open(dir+'/{}.txt'.format(dist_def[dist_idx]), 'w', encoding='utf-8') as f:
                f.write('alpha列表: '+str(alpha_list)+'\n')
                f.write('橫軸: precision, recall, F1, alpha, 測站數量\n')
                f.write('縱軸: '+str(thresholds_def)+'\n\n')
                for alpha_idx, alpha_i in enumerate(dist_i):
                    f.write('alpha: '+str(alpha_list[alpha_idx])+'\n')
                    f.write(str(alpha_i.round(5))+'\n\n')
                    
        ########################################################################################################################
        dir = path + '/{}/time_total/固定閾值軸'.format(filename)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        metric_cube = np.transpose(metric_cube,(2,1,0,3))  #(6個threshold到時, 5個alpha, 距離數量+1(總和), 3+1個指標)
        for thres_idx, thres_i in enumerate(metric_cube): 
            with open(dir+'/{}.txt'.format(thresholds_def[thres_idx]), 'w', encoding='utf-8') as f:
                f.write('alpha列表: '+str(alpha_list)+'\n')
                f.write('橫軸: precision, recall, F1, alpha, 測站數量\n')
                f.write('縱軸: '+str(dist_def)+'\n\n')
                for alpha_idx, alpha_i in enumerate(thres_i):
                    f.write('alpha: '+str(alpha_list[alpha_idx])+'\n')
                    f.write(str(alpha_i.round(5))+'\n\n')

        ########################################################################################################################





#===========================================================================================================================#
#===========================================================================================================================#
#===========================================================================================================================#
#===========================================================================================================================#
#===========================================================================================================================#

    if args.main_pred:
        print('============ main_pred ============')
        with open(args.path+'/predictions_{}.pkl'.format(args.weight_num), 'rb') as f:
            pre_pga_pred = pickle.load(f)  # [7個時間數, [1352個事件, [事件不同的測站數, (50, 3)]]]
        event_metadata, data, metadata = loader.load_events('japan.hdf5',
                                            parts=(False, False, True),
                                            shuffle_train_dev=False,
                                            custom_split=None,
                                            min_mag=None,
                                            mag_key='M_J',
                                            overwrite_sampling_rate=None)

        pga_true = data['pga']    # (1352,)
        magnitude = event_metadata['M_J']
        magnitude = np.array(magnitude.tolist())  #1352個事件規模

        config = json.load(open(args.experiment_path+'/config.json', 'r'))
        training_params = config['training_params']
        device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")
        ensemble = config.get('ensemble', 10)
        data_path = training_params['data_path']

        model_list = []
        print('loading weights......')
        for ens_id in range(ensemble):
            weight_path = os.path.join(args.experiment_path, str(ens_id), 'checkpoint.pth') 
            model_params = config['model_params'].copy()

            if config['training_params'].get('ensemble_rotation', False):
                model_params['rotation'] = np.pi / 4 * ens_id / (ensemble - 1)  ##前面在train是給定 p.pi / 4 * ens_id / (ensemble - 1)

            tmp_model = models.build_transformer_model(device=device, **model_params).to(device)
            tmp_model.load_state_dict(torch.load(weight_path)['model_weights'])
            model_list = model_list + [tmp_model]

        for time_i, time in enumerate(time_list):
            print('---- time:{} ----'.format(time))
            event_list = pre_pga_pred[3][time_i]

            for alpha in alpha_list:
                mag_confusion_matrix = np.zeros((len(magnitude_class), len(thresholds_def), len(thresholds_def)))
                for event_idx, event in enumerate(event_list):  #event: (測站數, 50, 3) for 1352次
                    magnitude_i = magnitude[event_idx]
                    true = pga_true[event_idx]  #(測站數,)
                    #print('1true',true)
                    true = np.reshape(true,(true.shape[0],1))
                    true = (true >= np.log10(thresholds * 9.81)).astype(int)

                    true = np.sum(true, axis=1).astype(int) # 要用此值當作查詢測站threshold時間的index (測站數,) 0:沒有警報 / 1:threshold_0 / 2:threshold_1 / 3:threshold_2 / 4:threshold_3  [0,2,1,3,2,0,......,測站數] 
                    true[true==0] = 99
                    true_time = [np.nan]*true.shape[0]
                    for station_idx, station in enumerate(head_times_pre_pga_pred[4][event_idx][1]):  # station: (5個threshold到時,)
                        if true[station_idx] >10:
                            continue
                        else: 
                            true_time[station_idx] = station[true[station_idx]-1]  #(測站數,) 裝每個測站最大震波到時
                    for station_idx in range(true.shape[0]):  #每個測站檢查，看時間有沒有超過
                        if time > true_time[station_idx] and true_time[station_idx] != np.nan:

                            cutout = int(sampling_rate * (true_time[station_idx] - 0.5 + 5))
                            cutout = (cutout, cutout + 1)
                            n_pga_targets = config['model_params'].get('n_pga_targets', 0)
                            max_stations = config['model_params']['max_stations']
                            training_params['batch_size'] = 1
                            generator = util.PreloadedEventGenerator(data=data,
                                                                    event_metadata=event_metadata,
                                                                    coords_target=True,
                                                                    cutout=cutout,
                                                                    pga_targets=n_pga_targets,
                                                                    max_stations=max_stations,
                                                                    sampling_rate=sampling_rate,
                                                                    select_first=True,
                                                                    shuffle=False,
                                                                    pga_mode=True,
                                                                    **training_params)

                            reverse_index = generator.reverse_index

                            preds = torch.tensor([], device=device)  #(10, 20, 5, 3)
                            for model_idx, model in enumerate(model_list):
                                model.eval()
                                with torch.no_grad():
                                    generator_data = generator[reverse_index[event_idx] + (station_idx // 20)]  #取該事件該測站的該編號資料就好

                                    waveforms = generator_data[0][0].to(device)
                                    inputs = generator_data[0][1].to(device)
                                    targets = generator_data[0][2].to(device)
                                    outputs = model(waveforms, inputs, targets)   #(1, 20, 5, 3)

                                    preds = torch.cat((preds, outputs), 0)
                            preds = preds.cpu().numpy()
                            preds = np.concatenate(preds, axis=-2)  #(20, 50, 3)
                            event[station_idx,:,:] = preds[(station_idx % 20),:,:]

                    pred = np.empty((event.shape[0], len(thresholds)))  #準備裝(n個測站, 七個threshold)
                    for j, log_level in enumerate(np.log10(thresholds * 9.81)):  #循環每個threshold [0.0081, 0.025, 0.081, 0.14, 0.25, 0.44]
                        prob = np.sum(
                            event[:, :, 0] * (1 - norm.cdf((log_level - event[:, :, 1]) / event[:, :, 2])),
                            axis=-1) #得到測站是log_level的機率
                        exceedance = prob >= alpha
                        pred[:,j] = exceedance
                    pred = pred.astype(int)
                    pred = np.sum(pred, axis=1)

                    true[true==99] = 0
                    
                    if magnitude_i < 3.0:
                        for idx in range(len(pred)):
                            mag_confusion_matrix[0][pred[idx]][true[idx]] += 1

                    elif 3.0 <= magnitude_i < 4.0:
                        for idx in range(len(pred)):
                            mag_confusion_matrix[1][pred[idx]][true[idx]] += 1

                    elif 4.0 <= magnitude_i < 5.0:
                        for idx in range(len(pred)):
                            mag_confusion_matrix[2][pred[idx]][true[idx]] += 1

                    elif 5.0 <= magnitude_i < 6.0:
                        for idx in range(len(pred)):
                            mag_confusion_matrix[3][pred[idx]][true[idx]] += 1

                    elif 6.0 <= magnitude_i:
                        for idx in range(len(pred)):
                            mag_confusion_matrix[4][pred[idx]][true[idx]] += 1

                dir = args.path + '/prediction/time_{}/txt_file'.format(time)
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                path = args.path + '/prediction/time_{}/txt_file/[alpha{}] mag_confusion_matrix.txt'.format(time, alpha)
                if args.TEAM_thresholds:
                    path = args.path + '/prediction/time_{}/txt_file/[TEAM_thresholds][alpha{}] mag_confusion_matrix.txt'.format(time, alpha)

                with open(path,'w',encoding='utf-8') as f:
                    f.write(str(mag_confusion_matrix.astype(int).tolist()))

    if args.draw_img or args.metrics:
        if args.draw_img and args.metrics: print('============ draw_img & metrics ============')
        else:
            if args.draw_img: print('============ draw_img ============')
            if args.metrics: print('============ metrics ============')

        metrics_name = ['accuracy', 'precision', 'recall']
        def plot_confusion_matrix(cm, title, normalize, path, cmap=plt.cm.Blues):
            if args.TEAM_thresholds:
                i0,i1,i2,i3,i4,i5=np.sum(cm, 0).astype(int)
            else:
                i0,i1,i2,i3,i4,i5,i6=np.sum(cm, 0).astype(int)
            
            if normalize:
                cm = cm / np.sum(cm, 0)

            if args.TEAM_thresholds:
                labels_name_x = ['<1%\n'+str(i0), '1%\n'+str(i1), '2%\n'+str(i2), '5%\n'+str(i3), '10%\n'+str(i4), '20%\n'+str(i5)]
                labels_name_y = ['<1%', '1% ', '2% ', '5% ', '10%', '20%']
            else:
                labels_name_x = ['<3\n'+str(i0), '3\n'+str(i1), '4\n'+str(i2), '5-\n'+str(i3), '5+\n'+str(i4), '6-\n'+str(i5), '6+\n'+str(i6)]
                labels_name_y = ['<3', '3', '4', '5-', '5+', '6-', '6+']
                
            if cmap is None:
                cmap = plt.get_cmap('Blues')
                
            plt.figure(figsize=(8,7))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title, fontsize=11)
            plt.colorbar(ticks=np.linspace(0, 1, 11),)
            
            tick_marks = np.arange(len(labels_name_x))
            plt.xticks(tick_marks, labels_name_x, rotation=0, fontsize=12)
            plt.yticks(tick_marks, labels_name_y, fontsize=12)
                
            thresh = cm.max()/ 1.5 if normalize else cm.max()/ 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, "{:0.2f}".format(cm[i,j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=11)
            plt.tight_layout()
            plt.ylabel('Pred', fontsize=12, labelpad=12)
            plt.xlabel('True', fontsize=12, labelpad=12)

            plt.tight_layout()
            plt.savefig(path)
            plt.close()

        for time_idx, time in enumerate(time_list):
            print('---- time:{} ----'.format(time))

            alpha_metrics = np.zeros((len(alpha_list), len(thresholds_def), len(metrics_name)+1)) #(acc, pre, rec, F1) 為了算best alpha

            for alpha_idx, alpha in enumerate(alpha_list):
                print('alpha: {}'.format(alpha))
                np.seterr(invalid='ignore')

                txt_name = args.path + '/prediction/time_{}/txt_file/[alpha{}] mag_confusion_matrix.txt'.format(time, alpha)
                if args.TEAM_thresholds:
                    txt_name = args.path + '/prediction/time_{}/txt_file/[TEAM_thresholds][alpha{}] mag_confusion_matrix.txt'.format(time, alpha)
                with open(txt_name,'r',encoding='utf-8') as f:
                    confusion_matrix = f.read()
                    confusion_matrix = np.array(eval(confusion_matrix))

                ##開始draw_img
                if args.draw_img:
                    dir = args.path + '/prediction/time_{}/confusion_matrix/alpha_{}'.format(time,alpha)
                    if not os.path.isdir(dir):
                        os.makedirs(dir)
                    sum_matrix = np.zeros(confusion_matrix.shape[-2:])
                    over5_matrix = np.zeros(confusion_matrix.shape[-2:])
                    for idx, conf_mat in enumerate(confusion_matrix):
                        sum_matrix += conf_mat
                        if idx > 2:  #['<3','3-4','4-5','5-6','>6']
                            over5_matrix += conf_mat
                        path = args.path + '/prediction/time_{}/confusion_matrix/alpha_{}/{}.png'.format(time, alpha, magnitude_class[idx])
                        if args.TEAM_thresholds:
                            path = args.path + '/prediction/time_{}/confusion_matrix/alpha_{}/[TEAM_thresholds]{}.png'.format(time, alpha, magnitude_class[idx])
                        plot_confusion_matrix(path=path, cm=conf_mat, title='[time{}][alpha{}]Confusion Matrix - magnitude:{}'.format(time, alpha, magnitude_class[idx]), normalize=True)
                    path = args.path + '/prediction/time_{}/confusion_matrix/alpha_{}/[mag]5_more.png'.format(time, alpha)
                    if args.TEAM_thresholds:
                        path = args.path + '/prediction/time_{}/confusion_matrix/alpha_{}/[TEAM_thresholds][mag]5_more.png'.format(time, alpha)
                    plot_confusion_matrix(path=path, cm=sum_matrix.astype(int), title='[time{}][alpha{}]Confusion Matrix - magnitude:>5'.format(time, alpha), normalize=True)

                    path = args.path + '/prediction/time_{}/confusion_matrix/alpha_{}/[mag]total.png'.format(time, alpha)
                    if args.TEAM_thresholds:
                        path = args.path + '/prediction/time_{}/confusion_matrix/alpha_{}/[TEAM_thresholds][mag]total.png'.format(time, alpha)
                    plot_confusion_matrix(path=path, cm=sum_matrix.astype(int), title='[time{}][alpha{}]Confusion Matrix - magnitude:Total'.format(time, alpha), normalize=True)

                ##開始metrics
                if args.metrics:
                    confusion = np.zeros((len(magnitude_class), len(thresholds_def), len(metrics_name)))
                    for mag_idx, conf_mat in enumerate(np.array(confusion_matrix)):
                        for idx in range(len(thresholds_def)):
                            tp = np.sum(conf_mat[idx][idx])
                            tn = np.sum(conf_mat[:idx,:idx]) + np.sum(conf_mat[:idx,idx+1:]) + np.sum(conf_mat[idx+1:,:idx]) + np.sum(conf_mat[idx+1:,idx+1:])
                            fp = np.sum(conf_mat[idx,:idx]) + np.sum(conf_mat[idx+1:,idx])
                            fn = np.sum(conf_mat[:idx,idx]) + np.sum(conf_mat[idx,idx+1:])
                            accuracy = (tp+tn)/(tp+fp+fn+tn)
                            precision = tp/(tp+fp)
                            recall = tp/(tp+fn)
                            confusion[mag_idx, idx, :] = accuracy, precision, recall
                        
                    sum_matrix = np.zeros(confusion_matrix[0].shape)
                    over5_matrix = np.zeros(confusion_matrix[0].shape)
                    for idx, mat in enumerate(confusion_matrix):  #['<3','3-4','4-5','5-6','>6']
                        sum_matrix += mat
                        if idx > 2:
                            over5_matrix += mat
                    
                    average_metrics = np.zeros(len(thresholds_def))  #算平均alpha
                    total_confusion = np.zeros((len(thresholds_def), len(metrics_name)))
                    over5_confusion = np.zeros((len(thresholds_def), len(metrics_name)))
                    for idx in range(len(thresholds_def)):

                        tp = np.sum(sum_matrix[idx][idx])
                        tn = np.sum(sum_matrix[:idx,:idx]) + np.sum(sum_matrix[:idx,idx+1:]) + np.sum(sum_matrix[idx+1:,:idx]) + np.sum(sum_matrix[idx+1:,idx+1:])
                        fp = np.sum(sum_matrix[idx,:idx]) + np.sum(sum_matrix[idx,idx+1:])
                        fn = np.sum(sum_matrix[:idx,idx]) + np.sum(sum_matrix[idx+1:,idx])
                        over5_tp = np.sum(over5_matrix[idx][idx])
                        over5_tn = np.sum(over5_matrix[:idx,:idx]) + np.sum(over5_matrix[:idx,idx+1:]) + np.sum(over5_matrix[idx+1:,:idx]) + np.sum(over5_matrix[idx+1:,idx+1:])
                        over5_fp = np.sum(over5_matrix[idx,:idx]) + np.sum(over5_matrix[idx,idx+1:])
                        over5_fn = np.sum(over5_matrix[:idx,idx]) + np.sum(over5_matrix[idx+1:,idx])
                        accuracy = (tp+tn)/(tp+fp+fn+tn) 
                        over5_accuracy = (over5_tp+over5_tn)/(over5_tp+over5_fp+over5_fn+over5_tn) 
                        precision = tp/(tp+fp) 
                        over5_precision = over5_tp/(over5_tp+over5_fp) 
                        recall = tp/(tp+fn)
                        over5_recall = over5_tp/(over5_tp+over5_fn)
                        total_confusion[idx, :] = accuracy, precision, recall
                        over5_confusion[idx, :] = over5_accuracy, over5_precision, over5_recall

                        average_metrics[idx] = 2 * precision * recall / (precision + recall)
                        if np.isnan(average_metrics[idx]): average_metrics[idx] = 0.0

                        alpha_metrics[alpha_idx, idx, :-1] = accuracy, precision, recall

                    dir = args.path + '/prediction/time_{}/metrics'.format(time)
                    if not os.path.isdir(dir):
                        os.makedirs(dir)
                    txt_name = args.path + '/prediction/time_{}/metrics/[alpha{}]metrics.txt'.format(time, alpha)
                    if args.TEAM_thresholds:
                        txt_name = args.path + '/prediction/time_{}/metrics/[TEAM_thresholds][alpha{}]metrics.txt'.format(time, alpha)
                    with open(txt_name, 'w', encoding='utf-8') as f:
                        f.write('橫軸: accuracy, precision, recall\n')
                        f.write('縱軸: '+str(thresholds_def)+'\n\n')
                        for mag_idx, mag in enumerate(magnitude_class):
                            f.write(str(mag)+'\n')
                            f.write(str(confusion[mag_idx])+'\n\n')
                        f.write('[mag]5_more\n'+str(over5_confusion))
                        f.write('\n\n')
                        f.write('[mag]total\n'+str(total_confusion))

                    alpha_metrics[alpha_idx,:, -1] = average_metrics

            best_alpha_matrix = np.zeros((len(thresholds_def), len(metrics_name)+2)) #acc, pre, rec, F1, alpha
            argmax = np.argmax(alpha_metrics[:, :, -1], axis=0)  #(alpha, threshold, 指標)
            for threshold_idx, best_alpha_idx in enumerate(argmax):
                best_alpha_matrix[threshold_idx, :-1] = alpha_metrics[best_alpha_idx, threshold_idx, :]
                best_alpha_matrix[threshold_idx, -1] = alpha_list[best_alpha_idx]

            txt_name = args.path + '/prediction/time_{}/metrics/best_alpha.txt'.format(time)
            if args.TEAM_thresholds:
                txt_name = args.path + '/prediction/time_{}/metrics/[TEAM_thresholds]best_alpha.txt'.format(time)
            with open(txt_name, 'w', encoding='utf-8') as f:
                f.write('alpha列表: '+str(alpha_list)+'\n')
                f.write('橫軸: accuracy, precision, recall, F1, alpha\n')
                f.write('縱軸: '+str(thresholds_def)+'\n\n'+'整體指標\n')
                f.write(str(best_alpha_matrix)+'\n\n\n\n=========================================================================================\n\n\n\n分項指標\n橫軸: accuracy, precision, recall, F1\n')
                f.write('縱軸: '+str(thresholds_def)+'\n\n')

                for a_i, a in enumerate(alpha_list):
                    f.write('alpha: {}\n'.format(a))
                    f.write(str(alpha_metrics[a_i, :, :])+'\n\n')

                #for i,max_idx in enumerate(argmax):
                #    f.write('震度{}級: alpha{}      --{}\n{}'.format(thresholds_def[i], str(alpha_list[max_idx]), alpha_metrics[max_idx][i], alpha_metrics[:,i].tolist())+'\n\n')


    if args.analyze_dataset:
        path = args.analyze_dataset
        with open(args.path+'/predictions.pkl', 'rb') as f:
            pre_pga_pred = pickle.load(f)  # [7個時間數, [1352個事件, [事件不同的測站數, (50, 3)]]]
        #print('一、',len(pre_pga_pred))
        #print('二、',pre_pga_pred[0])              # [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 25.0]
        #print('三、',pre_pga_pred[1])              # array([], shape=(7, 0), dtype=float64)
        #print('四、',pre_pga_pred[2])              # array([], shape=(7, 0), dtype=float64)
        #print('五、',len(pre_pga_pred[3]))
        #print('五、',len(pre_pga_pred[3][0]))
        #print('五、',len(pre_pga_pred[3][0][0]))   # 45     ndarray (45, 50, 3)    45個測站預測結果 
        #print('六、',len(pre_pga_pred[3][0][1]))   # 26     ndarray (26, 50, 3)    26個測站預測結果
        #print('七、',len(pre_pga_pred[3][0][2]))   # 23     ndarray (23, 50, 3)    23個測站預測結果
        trace_count = 0
        for event_i in pre_pga_pred[3][0]:
            trace_count += len(event_i)

        event_metadata, data, metadata = loader.load_events("japan.hdf5", parts=(False, False, True),)  #(train, val, test)
        pga_true = data['pga']    # (1352,)
        
        label_trace_count = 0
        for event_i in pga_true:
            label_trace_count += len(event_i)

        magnitude = event_metadata['M_J']
        magnitude = np.array(magnitude.tolist())  #1352個事件規模

        print('========== dataset ==========')
        if trace_count == label_trace_count and len(pre_pga_pred[3][0]) == len(pga_true): 
            print('label 吻合 dataset\n')
        else: raise  ValueError('label 不吻合 dataset:   dataset事件數:{}/ label事件數:{}   dataset trace數:{}/ label trace數:{}'.format(len(pre_pga_pred[3][0]), len(pga_true), trace_count, label_trace_count))
        
        print('時間數: ',len(pre_pga_pred[3]))         # 7      List       7個時間點
        print('事件數: ',len(pre_pga_pred[3][0]))      # 1352   List       1352個事件
        print('trace數: ',trace_count)
        print('規模: ')
        print('   <1: ',np.sum(magnitude<1))
        print('   1-2: ',np.sum(magnitude<2)-np.sum(magnitude<1))
        print('   2-3: ',np.sum(magnitude<3)-np.sum(magnitude<2))
        print('   3-4: ',np.sum(magnitude<4)-np.sum(magnitude<3))
        print('   4-5: ',np.sum(magnitude<5)-np.sum(magnitude<4))
        print('   5-6: ',np.sum(magnitude<6)-np.sum(magnitude<5))
        print('   6-7: ',np.sum(magnitude<7)-np.sum(magnitude<6))
        print('   7-8: ',np.sum(magnitude<8)-np.sum(magnitude<7))
        print('   8-9: ',np.sum(magnitude<9)-np.sum(magnitude<8))
        print('   >9: ',np.sum(magnitude>=9))
        


        #print('一、', len(pga_true))       # 1352
        #print('二、', pga_true[0].shape)   # (45,)
        #print('三、', pga_true[1].shape)   # (26,) 