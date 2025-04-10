import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ================== 配置参数 ==================
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





def process_combination(X_train_raw, X_test_raw, features, combo_name):
    """针对固定划分的数据处理"""
    try:
        # 特征有效性验证
        missing_train = set(features) - set(X_train_raw.columns)
        missing_test = set(features) - set(X_test_raw.columns)
        if missing_train or missing_test:
            print(f"组合 {combo_name} 缺失特征")
            return

        # 提取特征子集
        X_train = X_train_raw[features].copy()
        X_test = X_test_raw[features].copy()

        # 分类变量编码（仅在训练集上拟合）
        categorical_cols = list(set(features) & {'Gender', 'Smoke', 'Alcohol'})
        if categorical_cols:
            X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
            # 保持测试集列与训练集对齐
            X_test = pd.get_dummies(X_test, columns=categorical_cols)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # 标准化处理（仅在训练集上拟合）
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        if numeric_cols.any():
            scaler = StandardScaler().fit(X_train[numeric_cols])
            X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        # 保存数据集
        np.save(f"{BASE_SAVE_PATH}X_train_{combo_name}.npy", X_train.values)
        np.save(f"{BASE_SAVE_PATH}X_test_{combo_name}.npy", X_test.values)
        
    except Exception as e:
        print(f"Error in {combo_name}: {str(e)}")

    return(X_train.values,X_test.values)

def generate_feature_combinations(data):
    """生成包含独立18物种的所有有效特征组合"""
    # 添加独立18物种组合
    combinations = [{
        'name': "18feat(18sp)",
        'features': BASE_SP_COLS
    }]
    
    # 生成元数据组合：从0个到全部元数据的组合
    meta_combinations = []
    for i in range(0, len(META_COLS)+1):  # 修改range起始为0
        meta_combinations += list(itertools.combinations(META_COLS, i))
    
    # 构建完整组合列表（过滤空元数据组合）
    for meta in meta_combinations:
        if len(meta) == 0:
            continue  # 跳过空组合，已单独添加
        
        total_features = 18 + len(meta)
        combo_name = f"{total_features}feat(18sp+{'+'.join(meta)})"
        
        # 检查特征有效性
        valid_features = [col for col in (BASE_SP_COLS + list(meta)) 
                         if col in data.columns]
        if len(valid_features) != total_features:
            print(f"警告：组合 {combo_name} 存在无效特征，已跳过")
            continue
            
        combinations.append({
            'name': combo_name,
            'features': valid_features
        })
    
    return combinations


def main_pipeline(raw_data):
    """重构后的主流程"""
    # 第一阶段：统一划分数据集
    X_raw = raw_data.drop('Group', axis=1)
    y_raw = raw_data['Group']
    
    # 关键修改：先统一划分数据集
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    # 保存原始划分结果
    y_train.to_csv(f"{BASE_SAVE_PATH}y_train_all.csv", index=False)
    y_test.to_csv(f"{BASE_SAVE_PATH}y_test_all.csv", index=False)

    # 第二阶段：生成特征组合（基于完整列信息）
    all_combinations = generate_feature_combinations(X_raw)
    print(f'一共有{len(all_combinations)}个组合！\n')
    # 第三阶段：并行处理所有特征组合
    for combo in all_combinations:
        process_combination(
            X_train_raw, X_test_raw,
            combo['features'], combo['name']
        )

# ================== 主流程 ==================
if __name__ == "__main__":
    # 加载原始数据
    data = pd.read_csv('/home/wanyiyang/AImodel/MODEL_2CLS/data_18markers_byMP3_metadata_3367samples_20250326.csv')
    data['Group'] = data['Group'].apply(lambda x: 'IBD' if x != 'Controls' else x)
    
    # 数据预处理
    data.fillna({'Age': data['Age'].median()}, inplace=True)
    for col in ['Gender', 'Smoke', 'Alcohol']:
        data[col] = data[col].fillna('Unknown')
    data.dropna(subset=BASE_SP_COLS, inplace=True)
    
    main_pipeline(data)



# ================== 生成结果示例 ==================
"""
生成的文件命名规则示例：
- X_train_19feat(18sp+Age).npy
- y_train_22feat(18sp+Age+Gender+Smoke+Alcohol).csv
- X_test_20feat(18sp+Gender+Alcohol).npy 等
"""
