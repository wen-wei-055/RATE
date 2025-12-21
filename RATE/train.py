import numpy as np
import torch
import os
import argparse
import json
import time
import sys
import util
import loader
import logging
import shutil
import models

from models import FullModel
from tqdm import tqdm
from scipy.stats import norm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_np_pt(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed) #CPU seed
    torch.cuda.manual_seed(seed) #GPU seed


def load_json_if_exists(path):
    if path and os.path.isfile(path):
        with open(path) as f:
            return list(json.load(f).values())
    return None


def freeze_model(
    model, 
    to_freeze_dict, 
    keep_step=None
    ):
    for (name, param) in model.named_parameters():
        if name.split('.')[0] in to_freeze_dict:
            param.requires_grad = False
        else: pass
    return model


def transfer_weights(
    model, 
    weights_path, 
    sleeptime=600
    ):
    print("weights_path")

    if os.path.isdir(weights_path):
        last_weight = sorted([x for x in os.listdir(weights_path) if x[:11] == 'checkpoint_'])[-1] 
        weights_path = os.path.join(weights_path, last_weight)
        
    print(weights_path)
    own_state = model.state_dict()
    state_dict = torch.load(weights_path)['model_weights']
    
    for name, param in state_dict.items():
        if name not in own_state.keys():
            print(f"{name} is not load weight")
            continue
        else:
            print(name)
            own_state[name].copy_(param)
            
    full_model.load_state_dict(own_state)
    return full_model


def gaussian_confusion_matrix(
    type,
    status, 
    confusion_matrix, 
    exceedance_prob=0.2,
    targets_pga=None, 
    pred=None, 
    thresholds=None, 
    loop=None,
    total_loss=None, 
    optimizer=None
    ):
    if status == 'accumulate':
        pred_matrix = np.empty((pred.shape[0], len(thresholds)))
        targets_pga = np.reshape(targets_pga,(targets_pga.shape[0],1))
        targets_pga = (targets_pga >= thresholds).astype(int)
        targets_pga = np.sum(targets_pga, axis=1)
        for j, level in enumerate(thresholds):
            prob = np.sum(
                pred[:, :, 0] * (1 - norm.cdf((level - pred[:, :, 1]) / pred[:, :, 2])),
                axis=-1) 
            exceedance = prob >= exceedance_prob
            pred_matrix[:,j] = exceedance
        pred_matrix = pred_matrix.astype(int)
        pred_matrix = np.sum(pred_matrix, axis=1)
        for idx in range(len(pred_matrix)):
            confusion_matrix[pred_matrix[idx]][targets_pga[idx]] += 1
            
    elif status == 'write_txt':
        print(confusion_matrix)
        with open(os.path.join(training_params['weight_path'], '{}_confusion_matrix.txt'.format(type)), 'a+', encoding='utf-8') as f:
            f.write(str(epoch)+'\n')
            f.write(str(confusion_matrix)+'\n\n')
            f.write('loss: {},   lr: {}'.format((total_loss / len(loop)), optimizer.param_groups[0]['lr'])+'\n\n')


def training(
    model, 
    optimizer, 
    loader, 
    epoch, 
    epochs, 
    device, 
    training_params, 
    pga_loss, 
    train_loss_record, 
    logger,
    current_station
    ):

    train_loop = tqdm(loader)
    model.train()
    model.TotalEmbedding.eval()
    model.To_GaussianDistribution.eval()
    total_train_loss = 0.0

    thresholds = np.log10(np.array([0.01, 0.02, 0.05, 0.1, 0.2])*9.81)  # Depends on dataset settings
    confusion_matrix = np.zeros((len(thresholds)+1, len(thresholds)+1)).astype(np.int32)
    for x,y in train_loop:
        inputs_waveforms, inputs_coords, unuse_list, targets_pga = \
                x[0].to(device), x[1].to(device), x[2].to(device).long(), y[0].to(device)

        targets_pga[targets_pga==0] = -1.5
        targets_pga = targets_pga[:,:current_station]
        targets_pga = targets_pga.contiguous().view(-1)

        # model running
        pred = model(inputs_waveforms, inputs_coords, unuse_list, None)   
        pred = pred[:,:current_station]
        pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1])
        
        unuse_list = unuse_list.contiguous().view(-1)
        selected_indices = torch.nonzero(unuse_list==1, as_tuple=False).squeeze(dim=1)
        
        pred = pred[selected_indices]
        targets_pga = targets_pga[selected_indices]
        
        # loss
        targets_pga = torch.unsqueeze(targets_pga, 0)  
        pred = torch.unsqueeze(pred,0)
        train_loss = pga_loss(targets_pga, pred)   

        total_train_loss = train_loss.item() + total_train_loss
        train_loss.backward()   
        
        clip_grad_norm_(model.parameters(), training_params['clipnorm'])
        optimizer.step()       
        optimizer.zero_grad()   

        train_loop.set_description(f"[Train Epoch {epoch+1}/{epochs}]")
        train_loop.set_postfix(loss=train_loss.detach().cpu().item())
        
        # confusion matrix 
        targets_pga = torch.squeeze(targets_pga, 0)
        pred = torch.squeeze(pred, 0)
        
        targets_pga = targets_pga.contiguous().view(-1, (pred.shape[-1] - 1) // 2).cpu().numpy() 
        pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1]).detach().cpu().numpy()
        gaussian_confusion_matrix('train', 'accumulate', confusion_matrix, targets_pga=targets_pga, pred=pred, thresholds=thresholds)
    gaussian_confusion_matrix('train', 'write_txt', confusion_matrix, loop=train_loop, total_loss=total_train_loss, optimizer=optimizer)
    

    train_loss_record.append(total_train_loss / len(train_loop))  
    
    logger.info('[Train] epoch: %d -> loss: %.4f' %(epoch, total_train_loss / len(train_loop)))
    return model, optimizer,train_loss_record


def validating(
    model, 
    optimizer,
    loader, 
    epoch, 
    epochs, 
    device, 
    pga_loss, 
    scheduler, 
    val_loss_record, 
    logger,
    current_station
    ):
    valid_loop = tqdm(loader)
    model.eval()
    total_val_loss = 0.0
    
    thresholds = np.log10(np.array([0.01, 0.02, 0.05, 0.1, 0.2])*9.8)
    confusion_matrix = np.zeros((len(thresholds)+1, len(thresholds)+1)).astype(np.int32)
    with torch.no_grad():
        for x,y in valid_loop:
            inputs_waveforms, inputs_coords, unuse_list, targets_pga = \
                    x[0].to(device), x[1].to(device), x[2].to(device).long(), y[0].to(device)
        
            targets_pga[targets_pga==0] = -1.5
            targets_pga = targets_pga[:,:current_station]
            targets_pga = targets_pga.contiguous().view(-1)
            # model running
            pred = model(inputs_waveforms, inputs_coords, unuse_list, None)     
            pred = pred[:,:current_station]
            pred = pred.contiguous().view(-1, pred.shape[2], pred.shape[3] ) 

            # loss
            targets_pga = torch.unsqueeze(targets_pga,0)
            pred = torch.unsqueeze(pred,0) 
            val_loss = pga_loss(targets_pga, pred)
            
            total_val_loss = val_loss.item() + total_val_loss
            valid_loop.set_description(f"[Eval Epoch {epoch+1}/{epochs}]")
            valid_loop.set_postfix(loss=val_loss.detach().cpu().item())

            # confusion matrix
            targets_pga = torch.squeeze(targets_pga, 0)
            pred = torch.squeeze(pred, 0)
            
            unuse_list = unuse_list.contiguous().view(-1)
            selected_indices = torch.nonzero(unuse_list==1, as_tuple=False).squeeze(dim=1)
            
            pred = pred[selected_indices]
            targets_pga = targets_pga[selected_indices]
            targets_pga = targets_pga.contiguous().view(-1, (pred.shape[-1] - 1) // 2).cpu().numpy() 
            pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1]).detach().cpu().numpy() 

            gaussian_confusion_matrix('val', 'accumulate', confusion_matrix, targets_pga=targets_pga, pred=pred, thresholds=thresholds)
        gaussian_confusion_matrix('val', 'write_txt', confusion_matrix, loop=valid_loop, total_loss=total_val_loss, optimizer=optimizer)
        

    val_loss_record.append((total_val_loss / len(valid_loop)))
    scheduler.step(total_val_loss/len(valid_loop))
    
    logger.info('[Eval] epoch: %d -> loss: %.4f\n---' %(epoch, total_val_loss/ len(valid_loop)))
    return val_loss_record,scheduler


def my_custom_logger(
    logger_name, 
    level=logging.DEBUG
    ):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
                    "%(lineno)d — %(message)s")
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))

    torch.set_num_threads(6)
    seed_np_pt(config.get('seed', 42))

    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])
    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")
    stations_table = json.load(open(training_params['station_json_file'], 'r'))
    current_station    = config['model_params']['current_station']
    historical_station = config['model_params']['historical_station']
    sampling_rate = training_params.get('sampling_rate', 100)
    noise_seconds = training_params.get('noise_seconds', 5)
    train_val_boundary = training_params['train_val_boundary']
    cutout = (
        sampling_rate * (noise_seconds + generator_params[0]['cutout_start']),
        sampling_rate * (noise_seconds + generator_params[0]['cutout_end'])
    )

    if not os.path.isdir(training_params['weight_path']): os.mkdir(training_params['weight_path'])
    listdir = os.listdir(training_params['weight_path'])

    with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    copy_filedir = f"{training_params['weight_path']}/exec_code"
    os.makedirs(copy_filedir, exist_ok=True)

    for doc_name in ['train.py','util.py','models.py','loader.py']:
        shutil.copyfile(doc_name, f"{copy_filedir}/{doc_name}")

    print('Loading data')
    if args.test_run: limit = 10
    else: limit = None

    assert len(generator_params) == len(training_params['train_data_path'])
    assert len(generator_params) == len(training_params['val_data_path'])

    # For training use only
    first_station_appearance = load_json_if_exists(training_params.get("first_station_appearance"))
    last_station_appearance = load_json_if_exists(training_params.get("last_station_appearance"))
    print('==============================================================')
    print('===================     Start Training     ===================')
    print('==============================================================')
    if not os.path.isdir(training_params['weight_path']):
        os.mkdir(training_params['weight_path'])

    with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    full_model = FullModel(**config['model_params'], device=device).to(device)
    full_model = freeze_model(full_model, ['TotalEmbedding', 'To_GaussianDistribution']).to(device)

    if 'transfer_model_path' in training_params:
        print('Transfering model weights')
        full_model = transfer_weights(full_model, training_params['transfer_model_path'])
    
    train_datas = []
    val_datas = []               

    for i, generator_param_set in enumerate(generator_params): 
        noise_seconds = generator_param_set.get('noise_seconds', 5)
        cutout = (sampling_rate * (noise_seconds + generator_param_set['cutout_start']), sampling_rate * (noise_seconds + generator_param_set['cutout_end']))
        train_datas += [
            util.PreloadedEventGenerator(
                datapath=training_params['train_data_path'],
                total_datapath=training_params['total_data_path'],
                train_val_boundary=train_val_boundary,
                datatype='Train',
                min_mag=generator_param_set.get('min_mag', 0),
                limit=limit,
                stations_table=stations_table,
                station_blinding=True,
                cutout=cutout,
                current_station=current_station,
                historical_station=historical_station,
                sampling_rate=sampling_rate,
                first_station_appearance=first_station_appearance,
                last_station_appearance=last_station_appearance,
                **generator_param_set
            )
        ]
        val_datas += [
            util.PreloadedEventGenerator(
                datapath=training_params['val_data_path'],
                total_datapath=training_params['total_data_path'],
                train_val_boundary=train_val_boundary,
                datatype='Val',
                min_mag=generator_param_set.get('min_mag', None),
                limit=limit,
                stations_table=stations_table,
                station_blinding=True,
                cutout=cutout,
                current_station=current_station,
                historical_station=historical_station,
                sampling_rate=sampling_rate,
                first_station_appearance=first_station_appearance,
                last_station_appearance=last_station_appearance,
                **generator_param_set
            )
        ]


    filepath = os.path.join(training_params['weight_path'], 'event-{epoch:02d}.hdf5')
    workers = training_params.get('workers', 0)
    
    
    optimizer = torch.optim.Adam(full_model.parameters(), lr=training_params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=4)

    weighted_loss = training_params.get('weighted_loss_threshold', None)
    weighted_loss_threshold = None
    weight = None
    if weighted_loss:
        weighted_loss_threshold = weighted_loss['threshold']
        weight = weighted_loss['weight']

    def pga_loss(y_true, y_pred):
        return models.time_distributed_loss(
            y_true,
            y_pred, 
            models.mixture_density_loss, 
            weight=weight, 
            weighted_loss_threshold=weighted_loss_threshold,
            device=device, 
            mean=True, 
            kwloss={'mean': False})
        
    losses = {}
    losses['pga'] = pga_loss
        
    
    num_epochs = training_params['epochs_full_model']
    metrics_record = {}
    train_loss_record = []
    val_loss_record = []
    lr_record = []
    log_path = training_params['weight_path']+'/train.log'
    logger = my_custom_logger(log_path)
    logger.info('start training')
    
    train_generators = DataLoader(
        train_datas[0], 
        shuffle=True, 
        batch_size=None, 
        collate_fn=models.my_collate, 
        pin_memory=False, 
        num_workers=workers
    )
    val_generators = DataLoader(
        val_datas[0], 
        shuffle=False, 
        batch_size=None, 
        collate_fn=models.my_collate, 
        pin_memory=False, 
        num_workers=workers
    )
        
    for epoch in range(num_epochs):

        full_model, optimizer,train_loss_record = training(
            full_model, 
            optimizer, 
            train_generators, 
            epoch,num_epochs, 
            device,
            training_params,
            pga_loss,
            train_loss_record,
            logger,
            current_station
        )
        val_loss_record,scheduler = validating(
            full_model, 
            optimizer, 
            val_generators, 
            epoch,
            num_epochs, 
            device,
            pga_loss,
            scheduler,
            val_loss_record,
            logger,
            current_station
        )

        lr_record.append(scheduler.optimizer.param_groups[0]['lr'])
        
        record_num = 20 
        #save model
        if epoch < record_num:
            after_thre_min = -999 
        else:
            after_thre_min = min(val_loss_record[record_num-1:-1])
                
        if epoch%9==0 or (epoch>record_num-1 and epoch<record_num+11) or val_loss_record[-1]<after_thre_min: 
            metrics_record['train_loss'] = train_loss_record
            metrics_record['val_loss']  = val_loss_record
            metrics_record['lr_record'] = lr_record
            with open (os.path.join(training_params['weight_path'], 'metrics.txt'), 'w', encoding='utf-8') as f:
                f.write(str(metrics_record))

            print("-----Saving checkpoint-----")
            torch.save({
                'model_weights' : full_model.state_dict(), 
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
            }, 
            os.path.join(training_params['weight_path'], f'checkpoint_{epoch:02d}.pth'))