import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings

try:
    from config import Config
except ImportError:
    pass

warnings.filterwarnings("ignore")

def load_and_merge_metadata():
    print("--- 1. Loading Metadata & Applying 3-Class Mapping ---")
    try:
        diag = pd.read_csv(Config.DIAG_FILE, names=['pid', 'disease'], header=None)
        diag['pid'] = pd.to_numeric(diag['pid'], errors='coerce')
    except Exception as e:
        raise ValueError(f"Error reading diagnosis: {e}")

    try:
        demo = pd.read_csv(Config.DEMO_FILE, sep='\t', names=['pid', 'age', 'sex', 'bmi', 'wt', 'ht'])
        demo['pid'] = pd.to_numeric(demo['pid'], errors='coerce')
    except Exception as e:
        raise ValueError(f"Error reading demographics: {e}")

    meta = pd.merge(demo, diag, on='pid', how='inner').dropna(subset=['pid'])
    
    # --- NEW: Apply 3-Class Mapping Here ---
    # Map raw strings (e.g., 'Asthma') to aggregated classes (e.g., 'Other')
    meta['target_class'] = meta['disease'].map(Config.CLASS_MAPPING)
    
    # Drop rows that somehow didn't map (though CLASS_MAPPING covers all ICBHI classes)
    meta = meta.dropna(subset=['target_class'])
    
    # Create Numeric Label Index (0, 1, 2)
    class_to_idx = {name: idx for idx, name in enumerate(Config.CLASSES)}
    meta['label_idx'] = meta['target_class'].map(class_to_idx)
    
    print(f"✅ Metadata ready: {len(meta)} patients.")
    print(f"   Class Counts: {meta['target_class'].value_counts().to_dict()}")
    return meta

def check_annotation_overlap(start_sec, end_sec, event_df):
    if event_df.empty: return 0, 0
    overlaps = event_df[(event_df['start'] < end_sec) & (event_df['end'] > start_sec)]
    if overlaps.empty: return 0, 0
    c = 1 if overlaps['c'].sum() > 0 else 0
    w = 1 if overlaps['w'].sum() > 0 else 0
    return c, w

def process_audio(meta_df):
    print(f"--- 2. Slicing Audio (Overlap: {Config.OVERLAP*100}%) ---")
    
    files = sorted([f for f in os.listdir(Config.AUDIO_DIR) if f.endswith('.wav')])
    records = []
    
    for filename in tqdm(files, desc="Processing WAVs"):
        try:
            pid = int(filename.split('_')[0])
        except ValueError: continue

        if pid not in meta_df['pid'].values: continue

        # Retrieve the aggregated class label for this patient
        patient_meta = meta_df[meta_df['pid'] == pid].iloc[0]
        label_idx = patient_meta['label_idx']
        disease_str = patient_meta['target_class']

        file_path = os.path.join(Config.AUDIO_DIR, filename)
        try:
            audio, _ = librosa.load(file_path, sr=Config.SAMPLE_RATE)
        except: continue

        txt_path = file_path.replace('.wav', '.txt')
        event_df = pd.DataFrame()
        if os.path.exists(txt_path):
            try:
                event_df = pd.read_csv(txt_path, sep='\t', names=['start', 'end', 'c', 'w'])
            except: pass
        
        # Helper to add record
        def add_record(slice_name, c, w):
            records.append({
                'filename': slice_name, 
                'pid': pid, 
                'crackles': c, 
                'wheezes': w,
                'label_idx': label_idx,
                'disease': disease_str # Stores 'COPD', 'Healthy', or 'Other'
            })

        total_samples = len(audio)
        if total_samples < Config.TARGET_LEN:
            padding = Config.TARGET_LEN - total_samples
            slice_audio = np.pad(audio, (0, padding), mode='constant')
            slice_name = f"{filename.replace('.wav', '')}_0.wav"
            sf.write(os.path.join(Config.PROCESSED_DIR, slice_name), slice_audio, Config.SAMPLE_RATE)
            c, w = check_annotation_overlap(0, Config.DURATION, event_df)
            add_record(slice_name, c, w)
        else:
            for start_idx in range(0, total_samples - Config.TARGET_LEN + 1, Config.STEP_SIZE):
                end_idx = start_idx + Config.TARGET_LEN
                slice_audio = audio[start_idx:end_idx]
                
                start_sec = start_idx / Config.SAMPLE_RATE
                end_sec = end_idx / Config.SAMPLE_RATE
                c, w = check_annotation_overlap(start_sec, end_sec, event_df)
                
                slice_name = f"{filename.replace('.wav', '')}_{start_idx}.wav"
                sf.write(os.path.join(Config.PROCESSED_DIR, slice_name), slice_audio, Config.SAMPLE_RATE)
                add_record(slice_name, c, w)

    return pd.DataFrame(records)

def assign_folds(records_df, meta_df):
    print(f"--- 3. Assigning {Config.NUM_FOLDS} patient-stratified folds ---")
    
    # Just merge demographics (age/sex) back in; disease/label is already in records_df
    meta_subset = meta_df[['pid', 'age', 'sex']] 
    full_df = pd.merge(records_df, meta_subset, on='pid', how='left')
    patients = meta_df[['pid', 'label_idx']].drop_duplicates('pid').reset_index(drop=True)
    splitter = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    fold_by_pid = {}
    for fold_idx, (_, val_idx) in enumerate(
        splitter.split(np.zeros(len(patients)), patients['label_idx'])
    ):
        for pid in patients.iloc[val_idx]['pid']:
            fold_by_pid[int(pid)] = fold_idx
    full_df['fold'] = full_df['pid'].map(fold_by_pid).astype(int)
    full_df['cohort'] = 'ICHBI'
    full_df['split_role'] = 'development'
    if not (full_df.groupby('pid')['fold'].nunique() == 1).all():
        raise RuntimeError('Patient leakage detected across ICHBI folds')
        
    full_df.to_csv(Config.FULL_CSV, index=False)
    print(f"✅ Saved Master Dataset to {Config.FULL_CSV}")

if __name__ == "__main__":
    Config.set_seed(Config.SEED)
    Config.check_audio_paths()
    meta = load_and_merge_metadata()
    records = process_audio(meta)
    assign_folds(records, meta)
