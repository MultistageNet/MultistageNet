from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, configs, segment_index, slices, flag):

        """
        Each train/val/test data are segmented based on seq_len, pred_len

        In prediction task,
        X_dim = [batch_size, seq_len, stage_variables] ==> without target variable
        y_dim = [batch_size, 1] ==> the target value at last time points in X

        In forecasting task,
        X_dim = [batch_size, seq_len, stage_variables] ==> with target variable
        y_dim = [batch_size, pred_len] ==> the target value after last time points in X

        (Note that, in out setting, the number of target variable is one, but you can customize it)
        """

        # set task type
        self.task_type = configs.task_type 
        self.seq_len = configs.seq_len

        assert configs.task_type in ['forecasting', 'prediction']
        if configs.task_type == 'forecasting': self.pred_len = configs.pred_len
        if configs.task_type == 'prediction': self.pred_len = configs.pred_len -1
        
        # set train, val, test
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # delete timestamp in data
        if 'TimeStamp' in data.columns: data = data.drop(columns='TimeStamp')
        self.data = data.values
        self.slices = slices # train/val/test ratio
        self.segment_index = segment_index # Index actually used

        # read data
        self.__read_data__()

    def __read_data__(self):
        # Get train, val, test segmented points
        train_border = self.segment_index[self.slices[0]]
        val_border = self.segment_index[self.slices[1]]
        test_border = self.segment_index[self.slices[2]]

        # Define train/val/test start, end points 
        ## border1s = [train start, val start, test start]
        ## border2s = [train end, val end, test end]
        border1s = [0, min(val_border) - self.seq_len - self.pred_len, min(test_border) - self.seq_len - self.pred_len]
        border2s = [min(val_border), min(test_border), len(self.data)]

        # Set train or val or test border
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Train or val or test data
        self.data_x = self.data[border1:border2]
        self.data_y = self.data[border1:border2]

        if self.set_type == 0:
            self.set_segment_index = train_border
        elif self.set_type == 1:
            self.set_segment_index = val_border
        else:
            self.set_segment_index = test_border


    def __getitem__(self, index):
        
        # segment y end
        r_end = self.set_segment_index[index] - min(self.set_segment_index) + self.seq_len + self.pred_len 
        # segment y start
        r_begin = r_end - self.pred_len 
        # segment x end
        s_end = r_begin 
        # segment x start
        s_begin = s_end - self.seq_len 

        if self.task_type == 'prediction':
            # segmented X => [seq_len, stage_variables]
            seg_x = self.data_x[s_begin:s_end][:,:-1] # without target variable
            # segmented y => [1, 1]
            seg_y = self.data_y[r_begin-1:r_end][:,-1]

        if self.task_type == 'forecasting':
            # segmented X => [seq_len, stage_variables]
            seg_x = self.data_x[s_begin:s_end] # with target variable
            # segmented y => [pred_len, 1]
            seg_y = self.data_y[r_begin:r_end]

        return seg_x, seg_y

    def __len__(self):
        return len(self.set_segment_index)


def get_dataloader(data, configs, segment_index, slices, flag, shuffle = True, drop_last = True):

    dataset = CustomDataset(data, configs, segment_index, slices, flag)

    if flag == 'test': 
        drop_last = False
        shuffle = False


    data_loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=shuffle,
        num_workers=configs.n_workers,
        drop_last=drop_last)

    return data_loader