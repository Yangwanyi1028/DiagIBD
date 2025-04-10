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


# ================== 配置参数 ==================
DATA_DIR = '/home/wanyiyang/AImodel/'
REPORT_PATH = f'{DATA_DIR}model_performance_report_{datetime.now().strftime("%Y%m%d")}.csv'
N_SPLITS = 5  # Cross Validation 折数

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
        model.fit(X_tr, y_tr, epochs=20, batch_size=32, verbose=0)

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
    final_model.fit(X_train_scaled, y_encoded, epochs=20, batch_size=32, verbose=0)

    return final_model, scaler_final, optimal_threshold, le

# ================== 测试集可视化函数 ==================
def generate_visualizations(y_test, y_pred, y_prob, model_type, dataset_name):
    # plt.figure(figsize=(6,4))
    # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    # plt.title(f'Confusion Matrix - {model_type} ({dataset_name})')
    # plt.savefig(f'{DATA_DIR}{dataset_name}_{model_type}_Test_CM.png')
    # plt.close()

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

if __name__ == "__main__":
    train_files = glob.glob(f'{DATA_DIR}X_train_*feat*.npy')
    performance_data = []
    
    for train_file in train_files:
        dataset_name = train_file.split('X_train_')[1].split('.npy')[0]
        print(f"\nProcessing {dataset_name}...")

        try:
            X_train = np.load(train_file, allow_pickle=True)
            X_test = np.load(train_file.replace('train', 'test'), allow_pickle=True)
            y_train = pd.read_csv('/home/wanyiyang/AImodel/y_train_all.csv')
            y_test = pd.read_csv('/home/wanyiyang/AImodel/y_test_all.csv')

            # SVM模型训练与评估
            svm = SVC(probability=True, random_state=42)
            svm_model, svm_scaler, svm_thresh, svm_le = train_model_with_cv(svm, X_train, y_train)
            X_test_scaled = svm_scaler.transform(X_test)
            y_test_encoded = svm_le.transform(y_test.values.ravel())
            y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
            y_pred_svm = (y_prob_svm >= svm_thresh).astype(int)
            metrics_svm = calculate_metrics(y_test_encoded, y_pred_svm, y_prob_svm)
            metrics_svm.update({'Model': 'SVM', 'Dataset': dataset_name, 'Threshold': svm_thresh})
            performance_data.append(metrics_svm)

            # 深度学习模型训练与评估
            dl_model, dl_scaler, dl_thresh, dl_le = train_dl_with_cv(X_train, y_train)
            X_test_scaled_dl = dl_scaler.transform(X_test)
            y_test_encoded_dl = dl_le.transform(y_test.values.ravel())
            y_prob_dl = dl_model.predict(X_test_scaled_dl).flatten()
            y_pred_dl = (y_prob_dl >= dl_thresh).astype(int)
            metrics_dl = calculate_metrics(y_test_encoded_dl, y_pred_dl, y_prob_dl)
            metrics_dl.update({'Model': 'Deep Learning', 'Dataset': dataset_name, 'Threshold': dl_thresh})
            performance_data.append(metrics_dl)

            # 随机森林模型训练与评估
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model, rf_scaler, rf_thresh, rf_le = train_model_with_cv(rf, X_train, y_train)
            X_test_scaled_rf = rf_scaler.transform(X_test)
            y_test_encoded_rf = rf_le.transform(y_test.values.ravel())
            y_prob_rf = rf_model.predict_proba(X_test_scaled_rf)[:, 1]
            y_pred_rf = (y_prob_rf >= rf_thresh).astype(int)
            metrics_rf = calculate_metrics(y_test_encoded_rf, y_pred_rf, y_prob_rf)
            metrics_rf.update({'Model': 'Random Forest', 'Dataset': dataset_name, 'Threshold': rf_thresh})
            performance_data.append(metrics_rf)

            # 绘制三种模型的ROC曲线对比图
            plt.figure(figsize=(10, 8))
            fpr_svm, tpr_svm, _ = roc_curve(y_test_encoded, y_prob_svm)
            fpr_dl, tpr_dl, _ = roc_curve(y_test_encoded_dl, y_prob_dl)
            fpr_rf, tpr_rf, _ = roc_curve(y_test_encoded_rf, y_prob_rf)

            auc_svm = auc(fpr_svm, tpr_svm)
            auc_dl = auc(fpr_dl, tpr_dl)
            auc_rf = auc(fpr_rf, tpr_rf)

            plt.plot(fpr_svm, tpr_svm, 'b-', linewidth=2, label=f'SVM (AUC = {auc_svm:.3f})')
            plt.plot(fpr_dl, tpr_dl, 'r-', linewidth=2, label=f'Deep Learning (AUC = {auc_dl:.3f})')
            plt.plot(fpr_rf, tpr_rf, 'g-', linewidth=2, label=f'Random Forest (AUC = {auc_rf:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve Comparison - {dataset_name}')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{DATA_DIR}{dataset_name}_SVM_DL_RF_ROC_Comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"已生成 {dataset_name} 的SVM、DL和RF模型ROC曲线对比图")

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")

    report_df = pd.DataFrame(performance_data)
    report_df.to_csv(REPORT_PATH, index=False)
    print(f"\n报告已生成：{REPORT_PATH}")

