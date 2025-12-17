import csv
import datetime as dt
import os
import random
import sys
import tempfile
from types import SimpleNamespace


def build_sample_csv(tmpdir: str, filename: str):
    base_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    mode_cols = [f'OT_mode{i}' for i in range(1, 11)]
    columns = ['date'] + base_cols + mode_cols
    start = dt.datetime(2024, 1, 1, 0, 0, 0)
    path = os.path.join(tmpdir, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for step in range(80):
            stamp = start + dt.timedelta(hours=step)
            row = [stamp.isoformat()]
            for _ in base_cols:
                row.append(round(random.random(), 6))
            for _ in mode_cols:
                row.append(round(random.random(), 6))
            writer.writerow(row)
    return path, base_cols, mode_cols


def pure_header_check(csv_path: str, target: str):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    input_cols = [c for c in header if c != 'date']
    if target not in input_cols:
        raise AssertionError(f"Target {target} not in columns {input_cols}")
    ot_idx = input_cols.index(target)
    output_cols_expected = input_cols[: ot_idx + 1]
    post_target_cols = input_cols[ot_idx + 1 :]
    if output_cols_expected[-1] != target:
        raise AssertionError("output_cols must end with target")
    mode_like = [
        c
        for c in post_target_cols
        if ('mode' in c.lower())
        or ('imf' in c.lower())
        or ('svmd' in c.lower())
        or c.lower().startswith(f"{target.lower()}_")
    ]
    if mode_like and any(col in output_cols_expected for col in mode_like):
        raise AssertionError("Mode-like columns leaked into outputs")
    print("[E1] header check passed")
    print(f"  target index: {ot_idx}")
    print(f"  N: {len(input_cols)}, M: {len(output_cols_expected)}")
    if mode_like:
        print(f"  detected mode-like columns: {mode_like[:3]}")
    return input_cols, output_cols_expected


def safe_imports(repo_root):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None
    try:
        from data_provider.data_loader import Dataset_Custom, Dataset_Pred
        from model.SimpleTM import Model
    except Exception as exc:  # pragma: no cover - informative guard
        print(f"Dependencies missing for dataset/model imports: {exc}")
        return torch, None, None, None
    return torch, Dataset_Custom, Dataset_Pred, Model


def build_model_configs(seq_len: int, pred_len: int, c_out: int, dec_in: int):
    return SimpleNamespace(
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
        dec_in=dec_in,
    )


def run_case(model_cls, torch_lib, sample_x, pred_len, c_out, dec_in):
    configs = build_model_configs(seq_len=sample_x.shape[0], pred_len=pred_len, c_out=c_out, dec_in=dec_in)
    model = model_cls(configs)
    x = torch_lib.tensor(sample_x).unsqueeze(0).float()
    with torch_lib.no_grad():
        y_pred, _ = model(x, None, None, None)
    return y_pred


def main():
    tmpdir = tempfile.mkdtemp()
    csv_path, base_cols, mode_cols = build_sample_csv(tmpdir, 'sample.csv')
    target = 'OT'

    # E1: header-only checks
    input_cols, output_cols_expected = pure_header_check(csv_path, target)

    # Optional: dataset/model imports
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch_lib, Dataset_Custom, Dataset_Pred, Model = safe_imports(repo_root)
    if Dataset_Custom is None:
        print("[E2] Skipped dataset/model checks due to missing deps")
        return

    size = [32, 8, 8]
    train_ds = Dataset_Custom(
        root_path=tmpdir,
        flag='train',
        size=size,
        features='M',
        data_path=os.path.basename(csv_path),
        target=target,
        timeenc=0,
        freq='h',
    )
    pred_ds = Dataset_Pred(
        root_path=tmpdir,
        flag='pred',
        size=size,
        features='M',
        data_path=os.path.basename(csv_path),
        target=target,
        timeenc=0,
        freq='h',
        inverse=True,
    )

    assert train_ds.input_cols == pred_ds.input_cols, "Input columns differ between train and pred loaders"
    assert train_ds.output_cols == pred_ds.output_cols, "Output columns differ between train and pred loaders"
    assert train_ds.output_cols == output_cols_expected, "Output columns mismatch expected header-derived set"
    assert all(col not in train_ds.output_cols for col in mode_cols), "Output columns should not include modes"
    assert len(train_ds.input_cols) == len(base_cols) + len(mode_cols)
    assert len(train_ds.output_cols) == len(base_cols)

    sample_x, sample_y, _, _ = train_ds[0]
    assert sample_x.shape[1] == len(train_ds.input_cols)
    assert sample_y.shape[1] == len(train_ds.output_cols)

    print("[E2] dataset column checks passed")
    print("  N (inputs)", train_ds.N)
    print("  M (outputs)", train_ds.out_dim)
    print("  input_cols", train_ds.input_cols)
    print("  output_cols", train_ds.output_cols)
    print("  Dataset_Pred inverse=True shapes", pred_ds.data_x.shape, pred_ds.data_y.shape)

    if torch_lib is None:
        print("Torch not available; skipping model forward tests")
        return

    # Case A: dec_in = N (recommended)
    y_pred_a = run_case(Model, torch_lib, sample_x, pred_len=size[2], c_out=train_ds.out_dim, dec_in=train_ds.N)
    print("Case A (dec_in=N) output shape", tuple(y_pred_a.shape))

    # Case B: dec_in = M (should highlight mismatch if any)
    try:
        y_pred_b = run_case(Model, torch_lib, sample_x, pred_len=size[2], c_out=train_ds.out_dim, dec_in=train_ds.out_dim)
        print("Case B (dec_in=M) output shape", tuple(y_pred_b.shape))
        print("Case B passed; prefer dec_in=N for channel semantics")
    except Exception as exc:  # pragma: no cover - diagnostic output
        print("Case B (dec_in=M) failed:", exc)
        print("Use dec_in=N to match encoder channel count")


if __name__ == "__main__":
    main()
