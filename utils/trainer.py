import time
import copy
import torch
import torch.nn as nn
from torch import optim

from utils.accumulator import accumulator
from utils.set_seed import set_seed

def test(model, data_loader, config):

    task_type = config.task_type
    device = torch.device('cuda:'+ config.device_num)
    test_metric = accumulator()

    criterion = nn.MSELoss()

    model.eval()

    for batch_x, batch_y in data_loader['test']:
        if task_type == 'forecasting':
            batch_y = batch_y[:,:,-1]


        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        with torch.no_grad():
                
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            test_metric.add(outputs, batch_y)
            test_metric.loss_update(loss, batch_x.size(0))

    return test_metric.running_metric()


def train(model, data_loader, config, log_path):
    print(config, file=open(log_path, "a"))
    device = torch.device('cuda:'+ config.device_num)
    task_type = config.task_type

    set_seed(config.seed)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_loss = 1e+10
    patience = 0
    

    model.float().to(device)
    for epoch in range(config.n_epoch):

        epoch_time = time.time()
        best_model_update = False

        train_metric = accumulator()
        val_metric = accumulator()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch_x, batch_y in data_loader[phase]:
                if task_type == 'forecasting':
                    batch_y = batch_y[:,:,-1]

                optimizer.zero_grad()

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
  
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(batch_x)
                    
                    loss = criterion(outputs, batch_y)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    train_metric.add(outputs, batch_y)
                    train_metric.loss_update(loss, batch_x.size(0))

                if phase == 'val':
                    val_metric.add(outputs, batch_y)
                    val_metric.loss_update(loss, batch_x.size(0))

        tr_metrics = train_metric.running_metric()
        val_metrics = val_metric.running_metric()

        if val_metrics[0] < best_loss:
            best_model = copy.deepcopy(model.state_dict())
            best_loss = val_metrics[0]
            best_epoch = epoch
            best_val_metrics = val_metrics
            patience = 0
            best_model_update = True
        else:
            patience += 1

        if (epoch + 1) % 1 == 0:
            train_log = 'Epoch: {}/{} | Train loss: {:.4f} | Valid loss: {:.4f} | Patience: {} | Time per epoch: {:.2f} (sec)'.format(epoch + 1, config.n_epoch, tr_metrics[0], val_metrics[0], patience, round(time.time() - epoch_time, 4))
            print(train_log)
            print(train_log, file=open(log_path, "a"))
            if best_model_update:
                best_train_log = '==> Find best valid at epoch {} with valid loss of {:.4f}'.format(best_epoch + 1, best_val_metrics[0])
                print(best_train_log)
                print(best_train_log, file=open(log_path, "a"))

        if patience >= config.patience:
            print('Early stop!\n')
            return best_model, best_val_metrics

    return best_model, best_val_metrics
