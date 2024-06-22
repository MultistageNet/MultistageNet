import os
import csv
import numpy as np

def write_results(results_file, metrics, configs):

    write_dict = metrics.copy()
    write_dict.update(configs)

    fieldnames = list(write_dict.keys())

    if os.path.isfile(results_file) == False:
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writeheader()

    f = open(results_file, mode = 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(write_dict)
    f.close()

def write_mean_std(results_file, config):
    val_loss_list = []
    test_mse_list, test_rmse_list, test_mae_list = [], [], []
    if os.path.isfile(results_file):
        with open(results_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if all(row[key] == str(config[key]) for key in config if key != 'seed'):
                    val_loss_list.append(float(row['val_loss']))
                    test_mse_list.append(float(row['test_mse']))
                    test_rmse_list.append(float(row['test_rmse']))
                    test_mae_list.append(float(row['test_mae']))

    mean_result = {
        'val_loss': '{:.4f} ({:.4f})'.format(np.mean(val_loss_list),np.std(val_loss_list)),
        'test_mse': str(round(np.mean(test_mse_list),4)) + '±' + str(round(np.std(test_mse_list),4)),
        'test_rmse':str(round(np.mean(test_rmse_list),4)) + '±' + str(round(np.std(test_rmse_list),4)),
        'test_mae':str(round(np.mean(test_mae_list),4)) + '±' + str(round(np.std(test_mae_list),4)),
        'seed':'=> mean&std'
    }

    f = open(results_file, mode = 'a',newline='')
    writer = csv.DictWriter(f, fieldnames=mean_result.keys())
    writer.writerow(mean_result)
    f.close()