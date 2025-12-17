import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_provider.data_loader import Dataset_Custom, Dataset_Pred
from model.SimpleTM import Model


def build_sample_csv(tmpdir: str, filename: str):
    dates = pd.date_range("2024-01-01", periods=64, freq="H")
    base_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    mode_cols = [f'OT_mode{i}' for i in range(1, 11)]
    columns = ['date'] + base_cols + mode_cols
    data = np.random.randn(len(dates), len(columns) - 1)
    df = pd.DataFrame(data, columns=base_cols + mode_cols)
    df.insert(0, 'date', dates)
    csv_path = os.path.join(tmpdir, filename)
    df.to_csv(csv_path, index=False)
    return csv_path, base_cols, mode_cols


def build_model(seq_len: int, pred_len: int, enc_in: int, c_out: int):
    configs = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        output_attention=False,
        use_norm=False,
        geomattn_dropout=0.0,
        alpha=1.0,
        kernel_size=None,
        c_out=c_out,
        factor=1,
        dropout=0.1,
        requires_grad=True,
        wv='db1',
        m=1,
        d_model=16,
        d_ff=16,
        e_layers=1,
        activation='gelu',
        dec_in=c_out,
    )
    return Model(configs)


def main():
    tmpdir = tempfile.mkdtemp()
    csv_path, base_cols, mode_cols = build_sample_csv(tmpdir, 'sample.csv')
    size = [32, 8, 8]

    train_ds = Dataset_Custom(
        root_path=tmpdir,
        flag='train',
        size=size,
        features='M',
        data_path=os.path.basename(csv_path),
        target='OT',
        timeenc=0,
        freq='h',
    )
    pred_ds = Dataset_Pred(
        root_path=tmpdir,
        flag='pred',
        size=size,
        features='M',
        data_path=os.path.basename(csv_path),
        target='OT',
        timeenc=0,
        freq='h',
    )

    assert train_ds.input_cols == pred_ds.input_cols, "Input columns differ between train and pred loaders"
    assert train_ds.output_cols == pred_ds.output_cols, "Output columns differ between train and pred loaders"

    sample_x, sample_y, _, _ = train_ds[0]
    assert sample_x.shape[1] == len(train_ds.input_cols)
    assert sample_y.shape[1] == len(train_ds.output_cols)

    model = build_model(seq_len=size[0], pred_len=size[2], enc_in=train_ds.N, c_out=train_ds.out_dim)
    x = torch.tensor(sample_x).unsqueeze(0).float()
    with torch.no_grad():
        y_pred, _ = model(x, None, None, None)
    assert list(y_pred.shape) == [1, size[2], train_ds.out_dim], f"Unexpected output shape {y_pred.shape}"

    print(
        "Shapes OK",
        {
            "input_cols": train_ds.input_cols,
            "output_cols": train_ds.output_cols,
            "train_sample_x": sample_x.shape,
            "train_sample_y": sample_y.shape,
            "pred_data_x": pred_ds.data_x.shape,
            "pred_data_y": pred_ds.data_y.shape,
            "model_out": y_pred.shape,
        },
    )


if __name__ == "__main__":
    main()
