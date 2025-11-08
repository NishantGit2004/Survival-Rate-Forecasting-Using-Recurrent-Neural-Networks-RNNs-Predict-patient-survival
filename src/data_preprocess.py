#!/usr/bin/env python3
"""
Preprocessing script for MIMIC-III demo to produce time-series sequences per ICU stay.

Outputs train/val/test .npz files containing:
 - X: (n_samples, seq_len, n_features)
 - lengths: (n_samples,) actual lengths before padding
 - y: (n_samples,) binary survival label (1=survived discharge, 0=died)
 - ids: list of icustay_ids
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd


def safe_read_csv(path, **kwargs):
    """Safely read CSV with dtype=str and low memory off."""
    return pd.read_csv(path, dtype=str, low_memory=False, **kwargs)


def parse_dates(df, cols):
    """Convert listed columns to datetime if present."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def aggregate_events_to_timeseries(ce, le, features_itemids, time_index):
    """
    Aggregate CHARTEVENTS + LABEVENTS to fixed hourly bins.
    Returns array (len(time_index), len(features_itemids)).
    """
    rows = []
    for df in (ce, le):
        if df.empty:
            continue

        # --- Find time column ---
        time_col = None
        for c in df.columns:
            if "TIME" in c.upper():
                time_col = c
                break
        if time_col is None:
            continue

        # --- Find numeric value column ---
        value_col = None
        for c in ["VALUENUM", "VALUE"]:
            if c in df.columns:
                value_col = c
                break
        if value_col is None:
            continue

        # --- Find item identifier column ---
        itemid_col = None
        for c in ["ITEMID", "ITEMLABEL", "LABEL", "ITEM_NAME", "NAME"]:
            if c in df.columns:
                itemid_col = c
                break
        if itemid_col is None:
            continue

        df2 = df[[time_col, itemid_col, value_col]].copy()
        df2.columns = ["CHARTTIME", "ITEMID_COL", "VALUE_COL"]
        df2["VALUENUM"] = pd.to_numeric(df2["VALUE_COL"], errors="coerce")
        df2 = df2.dropna(subset=["VALUENUM"])
        df2["CHARTTIME"] = pd.to_datetime(df2["CHARTTIME"], errors="coerce")
        df2 = df2.dropna(subset=["CHARTTIME"])
        rows.append(df2[["CHARTTIME", "ITEMID_COL", "VALUENUM"]])

    if not rows:
        return np.full((len(time_index), len(features_itemids)), np.nan)

    events = pd.concat(rows, ignore_index=True)
    events["ITEMID_COL"] = events["ITEMID_COL"].astype(str)

    bins = pd.IntervalIndex.from_breaks(list(time_index) + [time_index[-1] + pd.Timedelta(hours=1)])
    events["bin"] = pd.cut(events["CHARTTIME"], bins=bins)

    arr = np.full((len(time_index), len(features_itemids)), np.nan)
    itemid_to_idx = {str(item): i for i, item in enumerate(features_itemids)}

    for b_idx, t in enumerate(time_index):
        bin_events = events[events["bin"].notna() & (events["bin"].apply(lambda x: x.left == t))]
        if bin_events.empty:
            continue
        grouped = bin_events.groupby("ITEMID_COL")["VALUENUM"].mean()
        for itemid, val in grouped.items():
            if itemid in itemid_to_idx:
                arr[b_idx, itemid_to_idx[itemid]] = val

    return arr


def pad_truncate_seq(arr, max_len, pad_value=np.nan):
    """Pad or truncate sequence to max_len."""
    seq_len = arr.shape[0]
    n_features = arr.shape[1]
    if seq_len == max_len:
        return arr, seq_len
    if seq_len < max_len:
        pad = np.full((max_len - seq_len, n_features), pad_value)
        return np.vstack([arr, pad]), seq_len
    return arr[-max_len:, :], max_len


def preprocess_all(raw_dir, out_dir, timestep_hours=1, max_seq_len=48, val_frac=0.15, test_frac=0.15):
    os.makedirs(out_dir, exist_ok=True)
    print("🚀 Loading CSVs (this may take some time)...")

    # --- Load data ---
    patients = safe_read_csv(os.path.join(raw_dir, "PATIENTS.csv"))
    admissions = safe_read_csv(os.path.join(raw_dir, "ADMISSIONS.csv"))
    icustays = safe_read_csv(os.path.join(raw_dir, "ICUSTAYS.csv"))
    chartevents = safe_read_csv(os.path.join(raw_dir, "CHARTEVENTS.csv"))
    labevents = safe_read_csv(os.path.join(raw_dir, "LABEVENTS.csv"))

    # --- Normalize column names ---
    for df in (patients, admissions, icustays, chartevents, labevents):
        df.columns = [c.upper() for c in df.columns]

    # --- Parse datetime columns ---
    patients = parse_dates(patients, ["DOB", "DOD"])
    admissions = parse_dates(admissions, ["ADMITTIME", "DISCHTIME", "DEATHTIME"])
    if "INTIME" in icustays.columns:
        icustays["INTIME"] = pd.to_datetime(icustays["INTIME"], errors="coerce")
    if "OUTTIME" in icustays.columns:
        icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"], errors="coerce")

    # --- Label: survived or died ---
    death_col = None
    for possible in ["DEATHTIME", "DOD", "DEATH_DATE", "HOSPITAL_EXPIRE_FLAG"]:
        if possible in admissions.columns:
            death_col = possible
            break

    if death_col is not None:
        if death_col == "HOSPITAL_EXPIRE_FLAG":
            admissions["SURVIVED"] = (admissions["HOSPITAL_EXPIRE_FLAG"].astype(float) == 0).astype(int)
        else:
            admissions["SURVIVED"] = admissions[death_col].isna().astype(int)
    else:
        print("⚠️ Warning: No death column found — assuming all patients survived.")
        admissions["SURVIVED"] = 1

    # --- Feature selection ---
    print("🔍 Detecting common ITEMIDs...")
    if "ITEMID" in chartevents.columns:
        top_itemids = chartevents["ITEMID"].value_counts().head(20).index.astype(str).tolist()
    else:
        label_col = next((c for c in chartevents.columns if "LABEL" in c or "NAME" in c or "ITEM" in c), None)
        if label_col:
            top_itemids = chartevents[label_col].value_counts().head(20).index.astype(str).tolist()
        else:
            print("⚠️ No identifiable ITEMID/LABEL column found, using default placeholder features.")
            top_itemids = [f"F{i}" for i in range(20)]

    selected_itemids = list(dict.fromkeys(top_itemids))
    print(f"✅ Selected {len(selected_itemids)} ITEMIDs for tracking.")

    # --- Construct sequences ---
    rows_X, rows_y, rows_len, rows_ids = [], [], [], []
    print("⚙️ Constructing sequences for each ICU stay...")

    for _, icu in tqdm(icustays.iterrows(), total=icustays.shape[0]):
        sub = icu.get("SUBJECT_ID")
        hadm = icu.get("HADM_ID")
        icu_id = icu.get("ICUSTAY_ID") or icu.get("ICUSTAY")
        adm_row = admissions[(admissions["SUBJECT_ID"] == sub) & (admissions["HADM_ID"] == hadm)]
        survived = int(adm_row.iloc[0]["SURVIVED"]) if not adm_row.empty else 1

        start = pd.to_datetime(icu.get("INTIME"), errors="coerce")
        end = pd.to_datetime(icu.get("OUTTIME"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue

        ce = chartevents[chartevents["SUBJECT_ID"] == sub]
        le = labevents[labevents["SUBJECT_ID"] == sub]

        time_index = pd.date_range(start=start.floor("H"), end=end.ceil("H"), freq=f"{timestep_hours}H")
        if len(time_index) == 0:
            continue

        arr = aggregate_events_to_timeseries(ce, le, selected_itemids, time_index)
        arr2, length = pad_truncate_seq(arr, max_seq_len)

        rows_X.append(arr2)
        rows_y.append(survived)
        rows_len.append(min(length, max_seq_len))
        rows_ids.append(str(icu_id))

    if not rows_X:
        raise RuntimeError("❌ No sequences constructed. Check your CSV paths or column names.")

    X = np.stack(rows_X)
    y = np.array(rows_y, dtype=int)
    lengths = np.array(rows_len, dtype=int)
    ids = np.array(rows_ids, dtype=object)

    # --- Impute missing values ---
    print("🧩 Imputing missing values (forward-fill → median)...")
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i])
        df = df.fillna(method="ffill").fillna(method="bfill")
        X[i] = df.values

    feat_medians = np.nanmedian(X.reshape(-1, X.shape[2]), axis=0)
    inds = np.where(np.isnan(X))
    for s, t, f in zip(*inds):
        X[s, t, f] = feat_medians[f]

    # --- Normalize ---
    print("📊 Normalizing features (min-max)...")
    X_flat = X.reshape(-1, X.shape[2])
    fmin = np.nanmin(X_flat, axis=0)
    fmax = np.nanmax(X_flat, axis=0)
    denom = fmax - fmin
    denom[denom == 0] = 1.0
    X = (X - fmin) / denom

    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    print("✅ Final check: NaNs =", np.isnan(X).sum(), "| Infs =", np.isinf(X).sum())

    # --- Split ---
    n = X.shape[0]
    idxs = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idxs)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idxs[:n_test]
    val_idx = idxs[n_test:n_test + n_val]
    train_idx = idxs[n_test + n_val:]
    print(f"✅ Total samples: {n}; train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    def save_split(name, indices):
        np.savez_compressed(
            os.path.join(out_dir, f"{name}.npz"),
            X=X[indices].astype(np.float32),
            y=y[indices].astype(np.int32),
            lengths=lengths[indices].astype(np.int32),
            ids=ids[indices],
        )
        print(f"💾 Saved {name}: {len(indices)} samples")

    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)

    with open(os.path.join(out_dir, "features.txt"), "w") as f:
        for it in selected_itemids:
            f.write(str(it) + "\n")

    print("🎯 Preprocessing complete! Data ready for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw csvs")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Path to save processed npz")
    parser.add_argument("--timestep_hours", type=int, default=1, help="Timestep in hours for binning")
    parser.add_argument("--max_seq_len", type=int, default=48, help="Max timesteps (pad/truncate)")
    args = parser.parse_args()

    preprocess_all(args.raw_dir, args.out_dir, args.timestep_hours, args.max_seq_len)