from sklearn.preprocessing import StandardScaler, MinMaxScaler

from datetime import timedelta
import numpy as np
import pandas as pd


def Load_liquidsugar_data(configs, scale = True):
    data_path = 'Dataset/{}.csv'.format(configs.dataset) # load data
    data = pd.read_csv(data_path, parse_dates=['TimeStamp'])

    # Define each stage variables
    stage_vars = { 
        'Stage1': ['feed_1', 'feed_2', 'feed_3', 'feed_4', 'feed_5', 'feed_6', 'feed_7', 'stage1_var1', 'stage1_var2'],
        'Stage2': ['stage2_var1', 'stage2_var2', 'stage2_var3', 'stage2_var4', 'stage2_var5', 'stage2_var6', 'stage2_var7', 'stage2_var8', 'stage2_var9', 'stage2_var10', 'stage2_var11', 'stage2_var12'],
        'Stage3': ['stage3_var1', 'stage3_var2',  'stage3_var3', 'yield_flow']}

    # Define x, y
    y_var = [configs.target] 
    x_vars = [i for j in stage_vars.values() for i in j if i != configs.target]
    
    # variable sort
    data = data[['TimeStamp'] + x_vars + y_var]

    # Index actually used according to seq_len and pred_len
    segment_index = get_segment_index(data, configs)

    # Data split
    num_train = int(len(segment_index) * 0.8)
    num_test = int(len(segment_index) * 0.1)
    num_val = len(segment_index) - num_train - num_test

    train_slice = slice(None, num_train)
    val_slice = slice(num_train, num_train + num_val)
    test_slice = slice(num_train+num_val, None)

    # Stadardization
    if scale:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        train_data = data[0:min(segment_index[num_train:])]
        train_x = train_data[x_vars].values
        train_y = train_data[y_var].values

        x_scaler.fit(train_x)
        y_scaler.fit(train_y)

        data_x = x_scaler.transform(data[x_vars].values)
        data_y = y_scaler.transform(data[y_var].values)

        scale_data = np.concatenate((data_x,data_y), axis = 1)

        data = pd.concat([data['TimeStamp'], pd.DataFrame(scale_data, columns= x_vars + y_var)], axis = 1)

        return data, stage_vars, segment_index, (train_slice, val_slice, test_slice), (x_scaler, y_scaler)

    else:
        return data, stage_vars, segment_index, (train_slice, val_slice, test_slice), None


def get_segment_index(data, configs):
    index_list = []
    for i in range((configs.seq_len + configs.pred_len), len(data)):
        temp = data.iloc[i-configs.seq_len-configs.pred_len:i]
        if (max(temp.TimeStamp) - min(temp.TimeStamp)).seconds == timedelta(minutes=configs.collection_interval * (configs.seq_len + configs.pred_len - 1)).seconds:
            index_list.append(i)

    return index_list

