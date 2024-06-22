from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, configs, segment_index, slices, flag):

        self.task_type = configs.task_type # set task type

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len 

        assert configs.task_type in ['forecasting', 'prediction']
        if configs.task_type == 'forecasting': self.pred_len = configs.pred_len
        if configs.task_type == 'prediction': self.pred_len = configs.pred_len -1

        if type(self.label_len) is not int: self.label_len = 0
        
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        if 'TimeStamp' in data.columns: data = data.drop(columns='TimeStamp')
        self.data = data.values
        self.slices = slices # train/val/test ratio
        self.segment_index = segment_index # Index actually used

        self.__read_data__()

    def __read_data__(self):
        train_border = self.segment_index[self.slices[0]] # train indexes
        val_border = self.segment_index[self.slices[1]] # val indexes
        test_border = self.segment_index[self.slices[2]] # test indexes

        border1s = [0, min(val_border) - self.seq_len - self.pred_len, min(test_border) - self.seq_len - self.pred_len] # train start / val start / test start
        border2s = [min(val_border), min(test_border), len(self.data)] # train end / val end / test end

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = self.data[border1:border2]
        self.data_y = self.data[border1:border2]

        if self.set_type == 0:
            self.set_segment_index = train_border
        elif self.set_type == 1:
            self.set_segment_index = val_border
        else:
            self.set_segment_index = test_border


    def __getitem__(self, index):
        if self.task_type == 'prediction':
            r_end = self.set_segment_index[index] - min(self.set_segment_index) + self.seq_len + self.pred_len # segment_y end
            r_begin = r_end - self.pred_len - self.label_len # segment_y start
            s_end = r_begin + self.label_len # segment_x end
            s_begin = s_end - self.seq_len # segment_x start

            seg_x = self.data_x[s_begin:s_end][:,:-1] # segment_x / dim:(seq_len x variabels)
            seg_y = self.data_y[r_begin-1:r_end][:,-1] # segment_y / dim:(pred_len x 1)

        if self.task_type == 'forecasting':
            r_end = self.set_segment_index[index] - min(self.set_segment_index) + self.seq_len + self.pred_len # segment_y end
            r_begin = r_end - self.pred_len - self.label_len # segment_y start
            s_end = r_begin + self.label_len # segment_x end
            s_begin = s_end - self.seq_len # segment_x start

            seg_x = self.data_x[s_begin:s_end] # segment_x / dim:(seq_len x variabels)
            seg_y = self.data_y[r_begin:r_end] # segment_y / dim:(pred_len x 1)

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
        num_workers=configs.num_workers,
        drop_last=drop_last)

    return data_loader