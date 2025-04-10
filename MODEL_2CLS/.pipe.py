import glob
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc, accuracy_score
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.model_selection import train_test_split
import os
from joblib import Parallel, delayed  # Add parallel processing
import time  # Add time module for tracking execution time

# ================== 配置参数 ==================
DATA_DIR = '/home/wanyiyang/AImodel/'
REPORT_PATH = f'{DATA_DIR}model_performance_report_{datetime.now().strftime("%Y%m%d")}.csv'
N_SPLITS = 5  # Cross Validation 折数

BASE_SAVE_PATH = '/home/wanyiyang/AImodel/'  # 基础存储路径
BASE_SP_COLS = [  # 固定18个物种特征列
    'Clostridium_leptum', 'Fusicatenibacter_saccharivorans',
    'Gemmiger_formicilis', 'Odoribacter_splanchnicus',
    'Ruminococcus_torques', 'Bilophila_wadsworthia',
    'Actinomyces_sp_oral_taxon_181', 'Blautia_hansenii',
    'Clostridium_spiroforme', 'Gemella_morbillorum',
    'Dorea_formicigenerans', 'Roseburia_inulinivorans',
    'Roseburia_intestinalis', 'Blautia_obeum',
    'Lawsonibacter_asaccharolyticus', 'Eubacterium_sp_CAG_274',
    'Bacteroides_fragilis', 'Escherichia_coli'
]
META_COLS = ['Age', 'Gender', 'Smoke', 'Alcohol']  # 元数据特征



# ================== 评估指标计算函数 ==================
def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        'AUC': auc(fpr, tpr),
        'ACC': accuracy_score(y_true, y_pred),
        'SEN': tp / (tp + fn),
        'SPE': tn / (tn + fp),
        'PPV': tp / (tp + fp),
        'NPV': tn / (tn + fn)
    }

# ================== 最佳阈值搜索函数 ==================
def find_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

# ================== 深度学习训练函数（CV优化阈值） ==================
def train_dl_with_cv(X_train, y_train):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train.values.ravel())
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    thresholds = []
    for train_idx, val_idx in skf.split(X_train, y_encoded):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_tr.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_tr, y_tr, epochs=20, batch_size=64, verbose=0)
        y_prob = model.predict(X_val).flatten()
        best_thresh = find_best_threshold(y_val, y_prob)
        thresholds.append(best_thresh)
    optimal_threshold = np.mean(thresholds)
    scaler_final = StandardScaler().fit(X_train)
    X_train_scaled = scaler_final.transform(X_train)
    final_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    final_model.fit(X_train_scaled, y_encoded, epochs=20, batch_size=64, verbose=0)
    return final_model, scaler_final, optimal_threshold, le

# ================== 测试集可视化函数 ==================
def generate_visualizations(y_test, y_pred, y_prob, model_type, dataset_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.2f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f'ROC Curve - {model_type} ({dataset_name})')
    plt.legend()
    plt.savefig(f'{DATA_DIR}{dataset_name}_{model_type}_Test_ROC.png')
    plt.close()

# ================== 通用模型训练函数（CV优化阈值） ==================
def train_model_with_cv(model, X_train, y_train):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train.values.ravel())
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    thresholds = []
    for train_idx, val_idx in skf.split(X_train, y_encoded):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]
        best_thresh = find_best_threshold(y_val, y_prob)
        thresholds.append(best_thresh)
    optimal_threshold = np.mean(thresholds)
    scaler_final = StandardScaler().fit(X_train)
    X_train_scaled = scaler_final.transform(X_train)
    final_model = model.fit(X_train_scaled, y_encoded)
    return final_model, scaler_final, optimal_threshold, le

# ================== 对比可视化函数 ==================
def generate_comparison_visualizations(y_test, y_prob_svm, y_prob_dl, dataset_name):
    plt.figure(figsize=(10, 8))
    # 计算SVM的ROC曲线和AUC
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    auc_svm = auc(fpr_svm, tpr_svm)
    # 计算DL的ROC曲线和AUC
    fpr_dl, tpr_dl, _ = roc_curve(y_test, y_prob_dl)
    auc_dl = auc(fpr_dl, tpr_dl)
    # 绘制ROC曲线
    plt.plot(fpr_svm, tpr_svm, 'b-', linewidth=2, label=f'SVM (AUC = {auc_svm:.3f})')
    plt.plot(fpr_dl, tpr_dl, 'r-', linewidth=2, label=f'Deep Learning (AUC = {auc_dl:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # 对角线
    # 设置图表属性
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison - {dataset_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    # 保存图表
    plt.savefig(f'{DATA_DIR}{dataset_name}_SVM_DL_ROC_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成 {dataset_name} 的SVM和DL模型ROC曲线对比图")


def generate_feature_combinations(data):
    """Generate all valid feature combinations containing at least one species feature."""
    species_combinations = [
        list(itertools.combinations(BASE_SP_COLS, i))
        for i in range(1, len(BASE_SP_COLS) + 1)
    ]
    meta_combinations = [
        list(itertools.combinations(META_COLS, j))
        for j in range(0, len(META_COLS) + 1)
    ]
    species_combinations = list(itertools.chain.from_iterable(species_combinations))
    meta_combinations = list(itertools.chain.from_iterable(meta_combinations))

    combinations = [
        {
            'name': f"{len(sp_combo) + len(meta_combo)}feat({len(sp_combo)}sp{'+' + '+'.join(meta_combo) if meta_combo else ''})",
            'features': list(sp_combo) + list(meta_combo),
        }
        for sp_combo in species_combinations
        for meta_combo in meta_combinations
        if all(col in data.columns for col in sp_combo + meta_combo)
    ]
    return combinations

def process_combination(X_train_raw, X_test_raw, features, combo_name):
    """Process feature combinations with caching for efficiency."""
    try:
        # Extract and preprocess features
        X_train = X_train_raw[features].copy()
        X_test = X_test_raw[features].copy()

        # Encode categorical variables
        categorical_cols = list(set(features) & {'Gender', 'Smoke', 'Alcohol'})
        if categorical_cols:
            X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=categorical_cols)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Standardize numeric columns
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        if numeric_cols.any():
            scaler = StandardScaler().fit(X_train[numeric_cols])
            X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        return X_train.values, X_test.values
    except Exception as e:
        print(f"Error in {combo_name}: {str(e)}")
        return None, None

def train_and_evaluate(combo, X_train, X_test, y_train, y_test):
    """Train and evaluate models for a given feature combination."""
    try:
        # Extract species abbreviation
        species_abbrev = '_'.join([
            '.'.join([part[:2] for part in f.split('_')])  # Take the first two letters of each part
            for f in combo['features'] if f in BASE_SP_COLS
        ])

        current_metrics = []

        # SVM Model
        svm = SVC(probability=True, random_state=42)
        svm_model, svm_scaler, svm_thresh, svm_le = train_model_with_cv(svm, X_train, y_train)
        X_test_scaled = svm_scaler.transform(X_test)
        y_test_encoded = svm_le.transform(y_test.values.ravel())
        y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
        y_pred_svm = (y_prob_svm >= svm_thresh).astype(int)
        metrics_svm = calculate_metrics(y_test_encoded, y_pred_svm, y_prob_svm)
        metrics_svm.update({
            'Model': 'SVM',
            'Species_abbrev': species_abbrev,
            'Feature_set': ', '.join(combo['features']),
            'Threshold': svm_thresh
        })
        current_metrics.append(metrics_svm)

        # Deep Learning Model
        dl_model, dl_scaler, dl_thresh, dl_le = train_dl_with_cv(X_train, y_train)
        X_test_scaled_dl = dl_scaler.transform(X_test)
        y_test_encoded_dl = dl_le.transform(y_test.values.ravel())
        y_prob_dl = dl_model.predict(X_test_scaled_dl).flatten()
        y_pred_dl = (y_prob_dl >= dl_thresh).astype(int)
        metrics_dl = calculate_metrics(y_test_encoded_dl, y_pred_dl, y_prob_dl)
        metrics_dl.update({
            'Model': 'Deep Learning',
            'Species_abbrev': species_abbrev,
            'Feature_set': ', '.join(combo['features']),
            'Threshold': dl_thresh
        })
        current_metrics.append(metrics_dl)

        # Random Forest Model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model, rf_scaler, rf_thresh, rf_le = train_model_with_cv(rf, X_train, y_train)
        X_test_scaled_rf = rf_scaler.transform(X_test)
        y_test_encoded_rf = rf_le.transform(y_test.values.ravel())
        y_prob_rf = rf_model.predict_proba(X_test_scaled_rf)[:, 1]
        y_pred_rf = (y_prob_rf >= rf_thresh).astype(int)
        metrics_rf = calculate_metrics(y_test_encoded_rf, y_pred_rf, y_prob_rf)
        metrics_rf.update({
            'Model': 'Random Forest',
            'Species_abbrev': species_abbrev,
            'Feature_set': ', '.join(combo['features']),
            'Threshold': rf_thresh
        })
        current_metrics.append(metrics_rf)

        # Check if any AUC meets threshold and generate ROC plot
        if max(metrics_svm['AUC'], metrics_dl['AUC'], metrics_rf['AUC']) >= 0.8:
            plt.figure(figsize=(10, 8))
            fpr_svm, tpr_svm, _ = roc_curve(y_test_encoded, y_prob_svm)
            fpr_dl, tpr_dl, _ = roc_curve(y_test_encoded_dl, y_prob_dl)
            fpr_rf, tpr_rf, _ = roc_curve(y_test_encoded_rf, y_prob_rf)

            plt.plot(fpr_svm, tpr_svm, 'b-', linewidth=2, label=f'SVM (AUC = {metrics_svm["AUC"]:.3f})')
            plt.plot(fpr_dl, tpr_dl, 'r-', linewidth=2, label=f'Deep Learning (AUC = {metrics_dl["AUC"]:.3f})')
            plt.plot(fpr_rf, tpr_rf, 'g-', linewidth=2, label=f'Random Forest (AUC = {metrics_rf["AUC"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve Comparison - {species_abbrev}')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)

            # Include species abbreviation, meta information, and number of features in the filename
            feature_count = len(combo['features'])
            species_features = [f for f in combo['features'] if f in BASE_SP_COLS]
            meta_features = [f for f in combo['features'] if f in META_COLS]
            species_abbrev_part = '_'.join([
                '.'.join([part[:2] for part in f.split('_')]) for f in species_features
            ])
            meta_full_names = '_'.join(meta_features).replace(' ', '_')
            plt.savefig(f'{DATA_DIR}{species_abbrev_part}_{meta_full_names}_{feature_count}features_ROC_Comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        return current_metrics
    except Exception as e:
        print(f"Error processing {combo['name']}: {str(e)}")
        return []

if __name__ == "__main__":
    start_time = time.time()  # Record start time

    train_files = glob.glob(f'{DATA_DIR}X_train_*feat*.npy')
    performance_data = []

    # Load raw data and split once
    data = pd.read_csv('/home/wanyiyang/AImodel/MODEL_2CLS/data_18markers_byMP3_metadata_3367samples_20250326.csv')
    data['Group'] = data['Group'].apply(lambda x: 'IBD' if x != 'Controls' else x)
    data.fillna({'Age': data['Age'].median()}, inplace=True)
    for col in ['Gender', 'Smoke', 'Alcohol']:
        data[col] = data[col].fillna('Unknown')
    data.dropna(subset=BASE_SP_COLS, inplace=True)

    X_raw = data.drop('Group', axis=1)
    y_raw = data['Group']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    y_train.to_csv(f"{BASE_SAVE_PATH}y_train_all.csv", index=False)
    y_test.to_csv(f"{BASE_SAVE_PATH}y_test_all.csv", index=False)

    # Generate combinations and process in parallel
    n_jobs = 4  # Limit the number of parallel threads to 4
    all_combinations = generate_feature_combinations(X_raw)
    processed_data = Parallel(n_jobs=n_jobs)(
        delayed(process_combination)(X_train_raw, X_test_raw, combo['features'], combo['name'])
        for combo in all_combinations
    )

    # Load labels once
    y_train = pd.read_csv(f"{BASE_SAVE_PATH}y_train_all.csv")
    y_test = pd.read_csv(f"{BASE_SAVE_PATH}y_test_all.csv")

    # Train models in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate)(combo, X_train, X_test, y_train, y_test)
        for combo, (X_train, X_test) in zip(all_combinations, processed_data)
        if X_train is not None and X_test is not None
    )

    # Flatten results and save to CSV
    performance_data = [metric for result in results for metric in result]
    pd.DataFrame(performance_data).to_csv(REPORT_PATH, index=False)
    print(f"\nFinal report saved: {REPORT_PATH}")

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Program completed in {elapsed_time:.2f} seconds.")
