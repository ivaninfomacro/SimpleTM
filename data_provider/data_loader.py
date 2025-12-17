import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        input_cols = [col for col in df_raw.columns if col != 'date']
        if self.target not in input_cols:
            raise ValueError(f"Target column {self.target} not found in data")
        ot_idx = input_cols.index(self.target)
        post_target_cols = input_cols[ot_idx + 1:]
        output_cols = input_cols[:ot_idx + 1]
        if output_cols[-1] != self.target:
            raise ValueError(
                f"Target column {self.target} must be the last column in output_cols; found {output_cols[-1]} at index {len(output_cols)-1}"
            )
        mode_like = [
            col for col in post_target_cols
            if ('mode' in col.lower())
            or ('imf' in col.lower())
            or ('svmd' in col.lower())
            or col.lower().startswith(f"{self.target.lower()}_")
        ]
        if mode_like and any(col in output_cols for col in mode_like):
            raise ValueError(
                f"Detected possible mode columns before target ({mode_like}); please append SVMD/IMF/mode columns after {self.target}"
            )
        self.ot_idx = ot_idx
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.output_ot_idx = output_cols.index(self.target)

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            df_x = df_raw[input_cols]
            df_y = df_raw[output_cols]
        elif self.features == 'MS':
            df_x = df_raw[input_cols]
            df_y = df_raw[[self.target]]
        elif self.features == 'S':
            df_x = df_raw[[self.target]]
            df_y = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported feature type: {self.features}")

        if self.scale:
            train_data = df_x[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_x = self.scaler.transform(df_x.values)
            col_to_idx = {col: idx for idx, col in enumerate(input_cols)}
            if self.features == 'M':
                y_indices = [col_to_idx[c] for c in output_cols]
                mean = self.scaler.mean_[y_indices]
                scale = self.scaler.scale_[y_indices]
                data_y = (df_y.values - mean) / scale
            elif self.features == 'MS':
                target_mean = self.scaler.mean_[col_to_idx[self.target]]
                target_scale = self.scaler.scale_[col_to_idx[self.target]]
                data_y = (df_y.values - target_mean) / target_scale
            else:
                data_y = self.scaler.transform(df_y.values)
        else:
            data_x = df_x.values
            data_y = df_y.values

        self.train_data = df_x[border1s[0]:border2s[0]].values
        self.N = data_x.shape[1]
        self.out_dim = data_y.shape[1]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        input_cols = [col for col in df_raw.columns if col != 'date']
        if self.target not in input_cols:
            raise ValueError(f"Target column {self.target} not found in data")
        ot_idx = input_cols.index(self.target)
        post_target_cols = input_cols[ot_idx + 1:]
        output_cols = input_cols[:ot_idx + 1]
        if output_cols[-1] != self.target:
            raise ValueError(
                f"Target column {self.target} must be the last column in output_cols; found {output_cols[-1]} at index {len(output_cols)-1}"
            )
        mode_like = [
            col for col in post_target_cols
            if ('mode' in col.lower())
            or ('imf' in col.lower())
            or ('svmd' in col.lower())
            or col.lower().startswith(f"{self.target.lower()}_")
        ]
        if mode_like and any(col in output_cols for col in mode_like):
            raise ValueError(
                f"Detected possible mode columns before target ({mode_like}); please append SVMD/IMF/mode columns after {self.target}"
            )
        self.ot_idx = ot_idx
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.output_ot_idx = output_cols.index(self.target)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            df_x = df_raw[input_cols]
            df_y = df_raw[output_cols]
        elif self.features == 'MS':
            df_x = df_raw[input_cols]
            df_y = df_raw[[self.target]]
        elif self.features == 'S':
            df_x = df_raw[[self.target]]
            df_y = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported feature type: {self.features}")

        if self.scale:
            train_data = df_x[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_x = self.scaler.transform(df_x.values)
            col_to_idx = {col: idx for idx, col in enumerate(input_cols)}
            if self.features == 'M':
                y_indices = [col_to_idx[c] for c in output_cols]
                mean = self.scaler.mean_[y_indices]
                scale = self.scaler.scale_[y_indices]
                data_y = (df_y.values - mean) / scale
            elif self.features == 'MS':
                target_mean = self.scaler.mean_[col_to_idx[self.target]]
                target_scale = self.scaler.scale_[col_to_idx[self.target]]
                data_y = (df_y.values - target_mean) / target_scale
            else:
                data_y = self.scaler.transform(df_y.values)
        else:
            data_x = df_x.values
            data_y = df_y.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

        self.train_data = df_x[border1s[0]:border2s[0]].values
        self.N = self.data_x.shape[1]
        self.out_dim = self.data_y.shape[1]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        input_cols = [col for col in df_raw.columns if col != 'date']
        if self.target not in input_cols:
            raise ValueError(f"Target column {self.target} not found in data")
        ot_idx = input_cols.index(self.target)
        post_target_cols = input_cols[ot_idx + 1:]
        output_cols = input_cols[:ot_idx + 1]
        if output_cols[-1] != self.target:
            raise ValueError(
                f"Target column {self.target} must be the last column in output_cols; found {output_cols[-1]} at index {len(output_cols)-1}"
            )
        mode_like = [
            col for col in post_target_cols
            if ('mode' in col.lower())
            or ('imf' in col.lower())
            or ('svmd' in col.lower())
            or col.lower().startswith(f"{self.target.lower()}_")
        ]
        if mode_like and any(col in output_cols for col in mode_like):
            raise ValueError(
                f"Detected possible mode columns before target ({mode_like}); please append SVMD/IMF/mode columns after {self.target}"
            )
        self.ot_idx = ot_idx
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.output_ot_idx = output_cols.index(self.target)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            df_x = df_raw[input_cols]
            df_y = df_raw[output_cols]
        elif self.features == 'MS':
            df_x = df_raw[input_cols]
            df_y = df_raw[[self.target]]
        elif self.features == 'S':
            df_x = df_raw[[self.target]]
            df_y = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported feature type: {self.features}")

        out_dim = df_y.shape[1]
        if self.scale:
            train_data = df_x[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_x = self.scaler.transform(df_x.values)
            col_to_idx = {col: idx for idx, col in enumerate(input_cols)}
            if self.features == 'M':
                y_indices = [col_to_idx[c] for c in output_cols]
                mean = self.scaler.mean_[y_indices]
                scale = self.scaler.scale_[y_indices]
                data_y = (df_y.values - mean) / scale
            elif self.features == 'MS':
                target_mean = self.scaler.mean_[col_to_idx[self.target]]
                target_scale = self.scaler.scale_[col_to_idx[self.target]]
                data_y = (df_y.values - target_mean) / target_scale
            else:
                data_y = self.scaler.transform(df_y.values)
        else:
            data_x = df_x.values
            data_y = df_y.values

        self.train_data = df_x[border1s[0]:border2s[0]].values
        self.N = data_x.shape[1]
        self.out_dim = out_dim

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        print('data file:', data_file)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df
        self.train_data = train_data
        self.N = self.data_x.shape[1]
        self.out_dim = self.data_y.shape[1]

    def __getitem__(self, index):
        if self.set_type == 2:  
            s_begin = index * 12
        else:
            s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 2: 
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) // 12
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.train_data = train_data if self.scale else df_data
        self.N = self.data_x.shape[1]
        self.out_dim = self.data_y.shape[1]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, pred_y_from='x'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.pred_y_from = pred_y_from
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.cols:
            input_cols = [col for col in self.cols if col != 'date']
        else:
            input_cols = [col for col in df_raw.columns if col != 'date']

        if self.target not in input_cols:
            raise ValueError(f"Target column {self.target} not found in data")
        ot_idx = input_cols.index(self.target)
        post_target_cols = input_cols[ot_idx + 1:]
        output_cols = input_cols[:ot_idx + 1]
        if output_cols[-1] != self.target:
            raise ValueError(
                f"Target column {self.target} must be the last column in output_cols; found {output_cols[-1]} at index {len(output_cols)-1}"
            )
        mode_like = [
            col for col in post_target_cols
            if ('mode' in col.lower())
            or ('imf' in col.lower())
            or ('svmd' in col.lower())
            or col.lower().startswith(f"{self.target.lower()}_")
        ]
        if mode_like and any(col in output_cols for col in mode_like):
            raise ValueError(
                f"Detected possible mode columns before target ({mode_like}); please append SVMD/IMF/mode columns after {self.target}"
            )
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.ot_idx = ot_idx
        self.output_ot_idx = output_cols.index(self.target)

        df_raw = df_raw[['date'] + input_cols]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M':
            df_x = df_raw[input_cols]
            df_y = df_raw[output_cols]
        elif self.features == 'MS':
            df_x = df_raw[input_cols]
            df_y = df_raw[[self.target]]
        elif self.features == 'S':
            df_x = df_raw[[self.target]]
            df_y = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported feature type: {self.features}")

        out_dim = df_y.shape[1]

        if self.scale:
            self.scaler.fit(df_x.values)
            data_x = self.scaler.transform(df_x.values)
            col_to_idx = {col: idx for idx, col in enumerate(input_cols)}
            if self.features == 'M':
                y_indices = [col_to_idx[c] for c in output_cols]
                mean = self.scaler.mean_[y_indices]
                scale = self.scaler.scale_[y_indices]
                data_y = (df_y.values - mean) / scale
            elif self.features == 'MS':
                target_mean = self.scaler.mean_[col_to_idx[self.target]]
                target_scale = self.scaler.scale_[col_to_idx[self.target]]
                data_y = (df_y.values - target_mean) / target_scale
            else:
                data_y = self.scaler.transform(df_y.values)
        else:
            data_x = df_x.values
            data_y = df_y.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data_x[border1:border2]
        if self.inverse:
            warnings.warn(
                "Dataset_Pred inverse=True returns seq_y derived from encoder inputs (N channels). "
                "If your model expects decoder inputs of size c_out, consider setting pred_y_from='y' or disabling inverse.")
            self.data_y = df_y.values[border1:border2]
        else:
            self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

        self.N = self.data_x.shape[1]
        self.out_dim = out_dim

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            if self.pred_y_from == 'y':
                seq_y = self.data_y[r_begin:r_begin + self.label_len]
            else:
                seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
