import os
import pandas as pd
from datetime import datetime
from pipe1 import generate_feature_combinations, chunkify, process_chunk, DATA_DIR, BASE_SAVE_PATH
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv('/home/wanyiyang/AImodel/MODEL_2CLS/data_18markers_byMP3_metadata_3367samples_20250326.csv')
data['Group'] = data['Group'].apply(lambda x: 'IBD' if x != 'Controls' else x)
data.fillna({'Age': data['Age'].median()}, inplace=True)
for col in ['Gender', 'Smoke', 'Alcohol']:
    data[col] = data[col].fillna('Unknown')
data.dropna(subset=data.columns, inplace=True)

# 生成所有组合
all_combinations = generate_feature_combinations(data)

# 选择范围 19201 到 137088 的组合
start_index = 19201 - 1  # Python 索引从 0 开始
end_index = 137088
selected_combinations = all_combinations[start_index:end_index]

# 分块处理
chunk_size = 100  # 每块包含的组合数量
combination_chunks = list(chunkify(selected_combinations, chunk_size))

# 加载训练和测试数据
X_raw = data.drop('Group', axis=1)
y_raw = data['Group']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# 保存标签以便后续使用
y_train.to_csv(f"{BASE_SAVE_PATH}y_train_all.csv", index=False)
y_test.to_csv(f"{BASE_SAVE_PATH}y_test_all.csv", index=False)

# 处理选定的组合块
for chunk_idx, chunk in enumerate(combination_chunks, start=1):
    print(f"Processing chunk {chunk_idx}...")
    process_chunk(chunk, chunk_idx, X_train_raw, X_test_raw, y_train, y_test)

print("Processing completed.")


# def main():
#     # 加载数据
#     data_path = '/home/wanyiyang/AImodel/MODEL_2CLS/data_18markers_byMP3_metadata_3367samples_20250326.csv'
#     data = pd.read_csv(data_path)
#     data['Group'] = data['Group'].apply(lambda x: 'IBD' if x != 'Controls' else x)
#     data.fillna({'Age': data['Age'].median()}, inplace=True)
#     for col in ['Gender', 'Smoke', 'Alcohol']:
#         data[col] = data[col].fillna('Unknown')
#     data.dropna(subset=BASE_SP_COLS, inplace=True)

#     # 生成所有组合
#     all_combinations = generate_feature_combinations(data)
#     total_combinations = len(all_combinations)
#     print(f"总组合数: {total_combinations}")

#     # 保存所有组合到文件
#     combinations_file = os.path.join(DATA_DIR, "all_combinations.csv")
#     with open(combinations_file, "w") as f:
#         f.write("Index,Combination Name,Features\n")
#         for idx, combo in enumerate(all_combinations, start=1):
#             combo_name = combo['name']
#             features = ", ".join(combo['features'])
#             f.write(f"{idx},{combo_name},{features}\n")
#     print(f"所有组合已保存到文件: {combinations_file}")

# if __name__ == "__main__":
#     main()

# from itertools import combinations

# BASE_SP_COLS = [
#     'Clostridium_leptum', 'Fusicatenibacter_saccharivorans',
#     'Gemmiger_formicilis', 'Odoribacter_splanchnicus',
#     'Ruminococcus_torques', 'Bilophila_wadsworthia',
#     'Actinomyces_sp_oral_taxon_181', 'Blautia_hansenii',
#     'Clostridium_spiroforme', 'Gemella_morbillorum',
#     'Dorea_formicigenerans', 'Roseburia_inulinivorans',
#     'Roseburia_intestinalis', 'Blautia_obeum',
#     'Lawsonibacter_asaccharolyticus', 'Eubacterium_sp_CAG_274',
#     'Bacteroides_fragilis', 'Escherichia_coli'
# ]
# META_COLS = ['Age', 'Gender', 'Smoke', 'Alcohol']

# # Calculate combinations of 5 features from BASE_SP_COLS
# base_combinations = list(combinations(BASE_SP_COLS, 5))
# num_base_combinations = len(base_combinations)

# # Calculate all combinations of META_COLS (including empty set)
# meta_combinations = sum(1 for r in range(len(META_COLS) + 1) for _ in combinations(META_COLS, r))
# num_meta_combinations = meta_combinations

# # Total combinations
# total_combinations = num_base_combinations * num_meta_combinations

# print(f"Number of combinations from BASE_SP_COLS: {num_base_combinations}")
# print(f"Number of combinations from META_COLS: {num_meta_combinations}")
# print(f"Total number of combinations: {total_combinations}")



