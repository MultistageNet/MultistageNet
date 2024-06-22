import argparse
import torch

from utils.dotdict import dotdict
from utils.set_seed import set_seed
from utils.trainer import train, test
from utils.datautils import Load_liquidsugar_data
from utils.dataloader import get_dataloader
from utils.write_results import write_results

from model.MultistageNet import MultistageNet

def get_argument_parser():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='LiquidSugar', help='dataset name')
    parser.add_argument('--target', type=str, default='yield_flow', help='target')
    parser.add_argument('--task_type', type=str, default='forecasting', help='task type: forecasting or prediction')
    parser.add_argument('--collection_interval', type=int, default=15, help='Data collection interval (minutes)')
    parser.add_argument('--n_stage', type=int, default=3, help='number of stages')
    parser.add_argument('--seq_len', type=int, default=12, help='sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction length')
    # Dataloader
    parser.add_argument('--batch_size', type=int, default=64, help='pretrain batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
    # Train set
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--n_epoch', type=int, default=10, help='train_epoch')
    # Model
    parser.add_argument('--model_name', type=str, default='MultistageNet', help='model name')
    parser.add_argument('--d_model', type=int, default=64, help='d_model of variable dimension')
    parser.add_argument('--d_ff', type=int, default=256, help='feed foward dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='number of head in attention')
    parser.add_argument('--kernel_size', type=int, default=3, help='number of head in attention')
    parser.add_argument('--num_causal_layers', type=int, default=3, help='number of stack in DepthwiseCausalConv')
    parser.add_argument('--num_mmp_layers', type=int, default=3, help='number of stack in Multistage Layer')
    parser.add_argument('--n_regressor_layer', type=int, default=3, help='number of stack in Regressor')
    parser.add_argument('--dropout_p', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--activation', type=str, default='GELU', help='activation function')
    # Else
    parser.add_argument('--seed', type=int, default=0, help='seet seed')
    parser.add_argument('--device_num', type=str, default='0', help='set gpu number')

    #args = parser.parse_args()
    args = parser.parse_args(args=[])

    return args

if __name__ == "__main__":
    args = get_argument_parser()
    print('\n', args)

    configs = dotdict(vars(args))

    # Define path
    log_path = 'save_log/train_log_{}_{}_{}_{}.txt'.format(configs.model_name, configs.dataset, configs.seed, configs.seq_len)
    model_save_path = 'save_model/{}_{}_{}_{}.pth'.format(configs.model_name, configs.dataset, configs.seed, configs.seq_len)
    result_path = 'save_result/{}_{}_{}_{}.csv'.format(configs.model_name, configs.dataset, configs.seed, configs.seq_len)

    # Load data
    set_seed(configs.seed)
    data, stage_vars, segment_index, slices, scaler = Load_liquidsugar_data(configs)

    # Dataloader
    data_loader = {}
    data_loader['train'] = get_dataloader(data, configs, segment_index, slices, flag='train', shuffle = True, drop_last=True)
    data_loader['val'] = get_dataloader(data, configs, segment_index, slices, flag='val', shuffle = True, drop_last=True)
    data_loader['test'] = get_dataloader(data, configs, segment_index, slices, flag='test')

    # Define model
    model = MultistageNet(configs, stage_vars)

    # Model train
    set_seed(configs.seed)
    best_model, val_metric = train(model, data_loader, configs, log_path)
    torch.save(best_model, model_save_path)

    # Test
    model.load_state_dict(best_model)
    test_metric = test(model, data_loader, configs, 'forecasting')

    results = {
        'val_loss': val_metric[0],
        'test_mse': test_metric[0],
        'test_rmse': test_metric[1],
        'test_mae': test_metric[2]
    }

    # write result
    write_results(result_path, results, configs)
