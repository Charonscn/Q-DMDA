import os
import warnings

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset


def z_score(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (mean - x) / (std + 1e-9), mean, std


def normalize(x, mean, std):
    return (mean - x) / (std + 1e-9)


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class PairedData:
    def __init__(self, source_loaders, target_loader, num_source_domains):
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.num_source_domains = num_source_domains

    def __iter__(self):
        self.source_iters = [iter(loader) for loader in self.source_loaders]
        self.target_iter = iter(self.target_loader)
        self.source_done = [False] * self.num_source_domains
        self.target_done = False
        return self

    def __next__(self):
        source_x, source_y = [], []
        all_sources_done = True

        for i in range(self.num_source_domains):
            try:
                x_i, y_i = next(self.source_iters[i])
            except StopIteration:
                self.source_done[i] = True
                self.source_iters[i] = iter(self.source_loaders[i])
                x_i, y_i = next(self.source_iters[i])

            source_x.append(x_i)
            source_y.append(y_i)
            if not self.source_done[i]:
                all_sources_done = False

        try:
            target_x, target_y = next(self.target_iter)
        except StopIteration:
            self.target_done = True
            self.target_iter = iter(self.target_loader)
            target_x, target_y = next(self.target_iter)

        if all_sources_done and self.target_done:
            raise StopIteration()

        batch = {}
        for i in range(self.num_source_domains):
            batch[f"Sx{i + 1}"] = source_x[i]
            batch[f"Sy{i + 1}"] = source_y[i]
        batch["Tx"] = target_x
        batch["Ty"] = target_y
        return batch


class UnalignedDataLoader:
    def initialize(self, num_domains, sx, sy, tx, ty, target_subject,
                   batch_size_src, batch_size_trg, drop_last_testing,
                   shuffle_testing):
        source_loaders = []
        self.source_datasets = []
        num_source_domains = num_domains - 1
        print("[*] Target subject", target_subject)

        for subject_idx in range(num_domains):
            if subject_idx == target_subject:
                continue
            x_tr = np.asarray(sx[subject_idx])
            y_tr = np.asarray(sy[subject_idx])
            x_tr, _, _ = z_score(x_tr)

            print(
                "Subject", subject_idx + 1,
                "Total:", len(y_tr),
                "# classes:", len(np.unique(y_tr))
            )
            dataset = ArrayDataset(x_tr, y_tr)
            self.source_datasets.append(dataset)
            source_loaders.append(
                DataLoader(
                    dataset,
                    batch_size=batch_size_src,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True,
                )
            )

        self.target_dataset = ArrayDataset(tx, ty)
        target_loader = DataLoader(
            self.target_dataset,
            batch_size=batch_size_trg,
            shuffle=shuffle_testing,
            num_workers=0,
            drop_last=drop_last_testing,
        )
        self.paired_data = PairedData(source_loaders, target_loader, num_source_domains)
        self.num_source_domains = num_source_domains

    def load_data(self):
        return self.paired_data


class PairedDataTesting:
    def __init__(self, target_loader):
        self.target_loader = target_loader

    def __iter__(self):
        self.target_iter = iter(self.target_loader)
        return self

    def __next__(self):
        target_x, target_y = next(self.target_iter)
        return {"Tx": target_x, "Ty": target_y}


class UnalignedDataLoaderTesting:
    def initialize(self, tx, ty, batch_size_trg, drop_last_testing, shuffle_testing):
        self.target_dataset = ArrayDataset(tx, ty)
        target_loader = DataLoader(
            self.target_dataset,
            batch_size=batch_size_trg,
            shuffle=shuffle_testing,
            num_workers=0,
            drop_last=drop_last_testing,
        )
        self.paired_data = PairedDataTesting(target_loader)

    def load_data(self):
        return self.paired_data


def fine_tuning_load_XY_MI(args, x_subjects, y_subjects):
    if args.dataset not in ["seed", "seed-iv", "bnci2014012", "mi"]:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    selected_subject = args.target - 1
    source_x, source_y = [], []
    target_key = None

    for idx, subject_key in enumerate(x_subjects.keys()):
        if idx == selected_subject:
            target_key = subject_key
            continue
        x_s = np.asarray(x_subjects[subject_key])
        y_s = np.asarray(y_subjects[subject_key])
        x_s, _, _ = z_score(x_s)
        source_x.append(x_s)
        source_y.append(y_s)

    if target_key is None:
        raise ValueError(f"Target subject {args.target} was not found.")

    sx = np.concatenate(source_x, axis=0)
    sy = np.concatenate(source_y, axis=0)
    tx = np.asarray(x_subjects[target_key])
    ty = np.asarray(y_subjects[target_key])

    vx = tx.copy()
    vy = ty.copy()
    tx, mean, std = z_score(tx)
    vx = normalize(vx, mean=mean, std=std)

    print("[+] Target subject:", target_key)
    print("Sx_train:", sx.shape, "Sy_train:", sy.shape)
    print("Tx_train:", tx.shape, "Ty_train:", ty.shape)
    print("Tx_test:", vx.shape, "Ty_test:", vy.shape)

    source_dataset = TensorDataset(
        torch.as_tensor(sx, dtype=torch.float32),
        torch.as_tensor(sy, dtype=torch.long),
    )
    target_dataset = TensorDataset(
        torch.as_tensor(tx, dtype=torch.float32),
        torch.as_tensor(ty, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.as_tensor(vx, dtype=torch.float32),
        torch.as_tensor(vy, dtype=torch.long),
    )

    return {
        "source": DataLoader(
            source_dataset,
            batch_size=args.batch_size_fine,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        ),
        "target": DataLoader(
            target_dataset,
            batch_size=args.batch_size_fine,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=200,
            shuffle=False,
            num_workers=0,
        ),
    }


def _get_first_existing_key(container, candidates):
    for key in candidates:
        if key in container:
            return key
    raise KeyError(f"None of these variables exists: {candidates}")


def _safe_scalar(container, key, default=None, is_h5=False):
    if key not in container:
        return default
    try:
        value = container[key][()] if is_h5 else container[key]
        value = np.asarray(value).squeeze()
        if value.size == 0:
            return default
        return int(value.reshape(-1)[0])
    except Exception:
        return default


def _mat_numeric_to_real_numpy(x, var_name="array", imag_tol=1e-6):
    x = np.asarray(x)
    if x.dtype.names is not None:
        names = set(x.dtype.names)
        if "real" not in names or "imag" not in names:
            raise TypeError(f"{var_name} has unsupported structured dtype: {x.dtype}")
        real = np.asarray(x["real"], dtype=np.float32)
        imag = np.asarray(x["imag"], dtype=np.float32)
        imag_max = float(np.max(np.abs(imag))) if imag.size > 0 else 0.0
        if imag_max > imag_tol:
            warnings.warn(
                f"{var_name} has non-zero imaginary part; only the real part is used.",
                RuntimeWarning,
            )
        return real.astype(np.float32)

    if np.iscomplexobj(x):
        imag_max = float(np.max(np.abs(np.imag(x)))) if x.size > 0 else 0.0
        if imag_max > imag_tol:
            warnings.warn(
                f"{var_name} is complex; only the real part is used.",
                RuntimeWarning,
            )
        return np.real(x).astype(np.float32)

    return np.asarray(x, dtype=np.float32)


def _fix_x_shape_mi1(x, n_windows=4, feat_dim_hint=None, n_samples_hint=None):
    x = _mat_numeric_to_real_numpy(x, var_name="X")
    while x.ndim > 3 and 1 in x.shape:
        x = np.squeeze(x, axis=list(x.shape).index(1))

    if x.ndim == 3:
        permutations = [
            x,
            np.transpose(x, (0, 2, 1)),
            np.transpose(x, (1, 0, 2)),
            np.transpose(x, (1, 2, 0)),
            np.transpose(x, (2, 0, 1)),
            np.transpose(x, (2, 1, 0)),
        ]
        candidates = []
        for arr in permutations:
            if arr.shape[1] != n_windows:
                continue
            score = 0
            if feat_dim_hint is not None and arr.shape[2] == feat_dim_hint:
                score += 10
            if n_samples_hint is not None and arr.shape[0] == n_samples_hint:
                score += 10
            candidates.append((score, arr))
        if not candidates:
            raise ValueError(
                f"Cannot infer X order from shape={x.shape}, "
                f"n_windows={n_windows}, feat_dim_hint={feat_dim_hint}, "
                f"n_samples_hint={n_samples_hint}"
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1].astype(np.float32)

    if x.ndim == 2:
        if n_windows != 1:
            raise ValueError(f"2-D X is only supported when n_windows=1, got {n_windows}")
        candidates = [x[:, None, :], x.T[:, None, :]]
        scored = []
        for arr in candidates:
            score = 0
            if feat_dim_hint is not None and arr.shape[2] == feat_dim_hint:
                score += 10
            if n_samples_hint is not None and arr.shape[0] == n_samples_hint:
                score += 10
            scored.append((score, arr))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1].astype(np.float32)

    raise ValueError(f"X must be a 2-D or 3-D array, got shape={x.shape}")


def _read_mi1_loso_mat(full_path, n_windows=4):
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Feature file not found: {full_path}")

    meta = {}
    try:
        data = scipy.io.loadmat(full_path)
        x_key = _get_first_existing_key(data, ["X_split", "X_all"])
        y_key = _get_first_existing_key(data, ["Y_split", "Y_all"])
        s_key = _get_first_existing_key(data, ["subj_split", "subject_id"])
        t_key = _get_first_existing_key(data, ["is_target"])

        x_raw = data[x_key]
        y_all = np.asarray(data[y_key]).squeeze()
        subject_id = np.asarray(data[s_key]).squeeze()
        is_target = np.asarray(data[t_key]).squeeze()
        meta["feat_dim"] = _safe_scalar(data, "feat_dim")
        meta["feat_dim_out"] = _safe_scalar(data, "feat_dim_out")
        meta["k"] = _safe_scalar(data, "k")
        meta["USE_GFK"] = _safe_scalar(data, "USE_GFK")
    except (NotImplementedError, ValueError, OSError):
        with h5py.File(full_path, "r") as data:
            x_key = _get_first_existing_key(data, ["X_split", "X_all"])
            y_key = _get_first_existing_key(data, ["Y_split", "Y_all"])
            s_key = _get_first_existing_key(data, ["subj_split", "subject_id"])
            t_key = _get_first_existing_key(data, ["is_target"])

            x_raw = data[x_key][()]
            y_all = np.asarray(data[y_key][()]).squeeze()
            subject_id = np.asarray(data[s_key][()]).squeeze()
            is_target = np.asarray(data[t_key][()]).squeeze()
            meta["feat_dim"] = _safe_scalar(data, "feat_dim", is_h5=True)
            meta["feat_dim_out"] = _safe_scalar(data, "feat_dim_out", is_h5=True)
            meta["k"] = _safe_scalar(data, "k", is_h5=True)
            meta["USE_GFK"] = _safe_scalar(data, "USE_GFK", is_h5=True)

    feat_dim_hint = meta.get("feat_dim_out") or meta.get("feat_dim")
    x_all = _fix_x_shape_mi1(
        x_raw,
        n_windows=n_windows,
        feat_dim_hint=feat_dim_hint,
        n_samples_hint=len(y_all),
    )
    y_all = np.asarray(y_all).reshape(-1)
    subject_id = np.asarray(subject_id).reshape(-1).astype(np.int64)
    is_target = np.asarray(is_target).reshape(-1).astype(np.int64)

    if x_all.shape[0] != len(y_all):
        raise ValueError(f"X/Y sample mismatch: X={x_all.shape}, Y={y_all.shape}")
    if x_all.shape[0] != len(subject_id):
        raise ValueError("X and subject_id sample counts do not match.")
    if x_all.shape[0] != len(is_target):
        raise ValueError("X and is_target sample counts do not match.")

    return x_all.astype(np.float32), y_all, subject_id, is_target, meta


def _feature_file_name(args, n_windows, k, mode, use_gfk):
    prefix = getattr(args, "feature_file_prefix", "Blankertz")
    suffix = "logmap_gfk" if use_gfk else "logmap"
    if use_gfk:
        return f"{prefix}_LOSO_target_{args.target:02d}_{mode}_W{n_windows}_{suffix}_k{k}.mat"
    return f"{prefix}_LOSO_target_{args.target:02d}_{mode}_W{n_windows}_{suffix}.mat"


def load_mi1(args, path, n_windows=4, k=25):
    feature_mode = getattr(args, "feature_mode", "spat")
    use_gfk = getattr(args, "feature_use_gfk", True)
    file_name = _feature_file_name(args, n_windows, k, feature_mode, use_gfk)
    full_path = os.path.join(path, file_name)
    print("LOSO feature file load:", full_path)

    shape_windows = getattr(args, "feature_shape_windows", n_windows)
    x_all, y_all, subject_id, is_target, meta = _read_mi1_loso_mat(
        full_path,
        n_windows=shape_windows,
    )

    extra_path = getattr(args, "feature_path_extra", "")
    if extra_path:
        extra_mode = getattr(args, "feature_mode_extra", feature_mode)
        extra_windows = getattr(args, "feature_windows_extra", n_windows)
        extra_shape_windows = getattr(args, "feature_shape_windows_extra", extra_windows)
        extra_k = getattr(args, "feature_k_extra", k)
        extra_use_gfk = getattr(args, "feature_use_gfk_extra", use_gfk)
        extra_file = _feature_file_name(
            args,
            extra_windows,
            extra_k,
            extra_mode,
            extra_use_gfk,
        )
        extra_full_path = os.path.join(extra_path, extra_file)
        print("Extra LOSO feature file load:", extra_full_path)
        x_extra, y_extra, subject_extra, target_extra, extra_meta = _read_mi1_loso_mat(
            extra_full_path,
            n_windows=extra_shape_windows,
        )
        if not np.array_equal(y_all, y_extra):
            raise ValueError("Extra feature labels do not match primary feature labels.")
        if not np.array_equal(subject_id, subject_extra):
            raise ValueError("Extra feature subject ids do not match primary feature subject ids.")
        if not np.array_equal(is_target, target_extra):
            raise ValueError("Extra feature target masks do not match primary feature target masks.")

        if x_extra.shape[1] < x_all.shape[1]:
            pad = np.zeros(
                (x_extra.shape[0], x_all.shape[1] - x_extra.shape[1], x_extra.shape[2]),
                dtype=x_extra.dtype,
            )
            x_extra = np.concatenate([x_extra, pad], axis=1)
        elif x_extra.shape[1] > x_all.shape[1]:
            pad = np.zeros(
                (x_all.shape[0], x_extra.shape[1] - x_all.shape[1], x_all.shape[2]),
                dtype=x_all.dtype,
            )
            x_all = np.concatenate([x_all, pad], axis=1)
        x_all = np.concatenate([x_all, x_extra], axis=2).astype(np.float32)
        meta["extra_feature"] = {
            "path": extra_full_path,
            "meta": extra_meta,
            "combined_feat_dim": x_all.shape[2],
        }

    print("Raw loaded shapes:")
    print("  X_all      :", x_all.shape)
    print("  Y_all      :", y_all.shape)
    print("  subject_id :", subject_id.shape)
    print("  is_target  :", is_target.shape)
    print("  meta       :", meta)
    print("Loaded feature dim:", x_all.shape[2])

    unique_labels = np.unique(y_all)
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    y_all = np.asarray([label_map[label] for label in y_all], dtype=np.int64)

    x_subjects = {}
    y_subjects = {}
    num_subjects = int(subject_id.max())
    for subject in range(1, num_subjects + 1):
        mask = subject_id == subject
        x_subjects[subject - 1] = x_all[mask].astype(np.float32)
        y_subjects[subject - 1] = y_all[mask].astype(np.int64)
        print(
            f"Subject {subject}: "
            f"X={x_subjects[subject - 1].shape}, "
            f"Y={y_subjects[subject - 1].shape}"
        )

    target_subject = args.target - 1
    if target_subject < 0 or target_subject >= num_subjects:
        raise ValueError(f"args.target={args.target} is out of range for {num_subjects} subjects.")

    tx = x_subjects[target_subject].copy()
    ty = y_subjects[target_subject].copy()
    tx, _, _ = z_score(tx)

    train_loader = UnalignedDataLoader()
    train_loader.initialize(
        num_subjects,
        x_subjects,
        y_subjects,
        tx,
        ty,
        target_subject,
        args.batch_size,
        args.batch_size,
        shuffle_testing=True,
        drop_last_testing=True,
    )

    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(
        tx,
        ty,
        10,
        shuffle_testing=False,
        drop_last_testing=False,
    )

    return train_loader.load_data(), test_loader.load_data(), x_subjects, y_subjects
