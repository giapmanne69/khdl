import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix
import math

# --- CẤU HÌNH ---
SONG_PATH = 'Data Processing - Python\\final_datasets\processed_songs.csv'
TRAIN_PATH = 'Data Processing - Python\\final_datasets\processed_train.csv'
MAPPING_PATH = 'Data Processing - Python\\final_datasets\song_mapping.csv'
MEMBER_PATH = 'Data Processing - Python\\final_datasets\members_mapping.csv'

print("--- 1. CHUẨN BỊ MÔI TRƯỜNG THÍ NGHIỆM (EXPERIMENTAL SETUP) ---")
# Load dữ liệu
train_df = pd.read_csv(TRAIN_PATH)
song_features = pd.read_csv(SONG_PATH).sort_values('song_id_encoded').reset_index(drop=True)
song_map = pd.read_csv(MAPPING_PATH).drop_duplicates('song_id_encoded')
member_df = pd.read_csv(MEMBER_PATH).drop_duplicates('user_id_encoded')

# Dictionary Metadata & Tuổi
song_info_dict = song_map.set_index('song_id_encoded')[['artist_name', 'genre_ids', 'language', 'song_length']].to_dict('index')
member_df['real_age'] = (member_df['bd'] * 50 + 15).round().astype(int)
user_age_dict = member_df.set_index('user_id_encoded')['real_age'].to_dict()

# [QUAN TRỌNG] CHIA TẬP TRAIN/TEST (80% Học, 20% Thi)
# Random_state=42 để đảm bảo kết quả có thể tái lập
train_split, test_split = train_test_split(train_df, test_size=0.2, random_state=42)

print(f"Tổng dữ liệu: {len(train_df)} dòng")
print(f"-> Tập học (Train Set - 80%): {len(train_split)} dòng (Dùng để xây model & thống kê)")
print(f"-> Tập thi (Test Set - 20%): {len(test_split)} dòng (Dùng để chấm điểm)")

# --- 2. LOGIC TIỀN XỬ LÝ TRÊN TẬP TRAIN (PHASE 4 LOGIC REPLICATION) ---
print("\n--- 2. TÍNH TOÁN CÁC THAM SỐ CHIẾN THUẬT (DỰA TRÊN TRAIN SET) ---")

# 2.1. High-Trust Artists Logic
# Lưu ý: Chỉ được tính thống kê trên tập Train_split để tránh rò rỉ dữ liệu (Data Leakage)
train_merged = train_split.merge(song_map[['song_id_encoded', 'artist_name']], on='song_id_encoded', how='left')
artist_stats = train_merged.groupby('artist_name')['target'].agg(['count', 'mean'])

# Điều kiện: > 50 lượt nghe và tỷ lệ nghe lại >= 65% (Theo Phase 4)
high_trust_artists = artist_stats[
    (artist_stats['count'] > 50) & 
    (artist_stats['mean'] >= 0.65)
].index.tolist()

if "Jason Mraz" not in high_trust_artists: high_trust_artists.append("Jason Mraz")
print(f"-> Phát hiện {len(high_trust_artists)} High-Trust Artists (Replay > 65%).")

# --- 3. HUẤN LUYỆN CORE MODELS ---
print("\n--- 3. HUẤN LUYỆN MÔ HÌNH ---")

# 3.1. CF Model (SVD)
n_users = train_df['user_id_encoded'].max() + 1
n_songs = train_df['song_id_encoded'].max() + 1

rows = train_split['user_id_encoded'].values
cols = train_split['song_id_encoded'].values
data = train_split['target'].values 
R_sparse = coo_matrix((data, (rows, cols)), shape=(n_users, n_songs))

print("Training SVD Model...")
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(R_sparse)
item_factors = svd.components_.T

def predict_cf_raw(u_idx, i_idx):
    try:
        if u_idx >= user_factors.shape[0] or i_idx >= item_factors.shape[0]: return 0.5
        score = np.dot(user_factors[u_idx], item_factors[i_idx])
        return np.clip(score, 0, 1)
    except: return 0.5

# 3.2. Content-Based Model
print("Calculating Cosine Similarity...")
feature_cols = [c for c in song_features.columns if c != 'song_id_encoded']
feature_matrix = song_features[feature_cols].values
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# --- 4. CHẤM ĐIỂM CƠ BẢN (RMSE & AUC) ---
print("\n--- 4. CHẤM ĐIỂM KỸ THUẬT (METRICS) ---")
# Phần này đánh giá khả năng dự đoán thô của CF Model

test_users = test_split['user_id_encoded'].values
test_items = test_split['song_id_encoded'].values
true_ratings = test_split['target'].values

pred_ratings = []
for u, i in zip(test_users, test_items):
    pred_ratings.append(predict_cf_raw(u, i))

rmse = math.sqrt(mean_squared_error(true_ratings, pred_ratings))
auc = roc_auc_score(true_ratings, pred_ratings)

print(f"✅ RMSE (Sai số): {rmse:.4f} (Càng thấp càng tốt)")
print(f"✅ ROC-AUC (Phân loại): {auc:.4f} (Càng gần 1 càng tốt, >0.6 là tạm ổn)")

# --- 5. CHẤM ĐIỂM CHIẾN THUẬT HYBRID (RANKING SIMULATION) ---
print("\n--- 5. SIMULATION: ĐÁNH GIÁ HIỆU QUẢ CHIẾN THUẬT PHASE 4 (HIT RATE) ---")
print("(Mô phỏng 100 phiên nghe thử nghiệm áp dụng các luật Penalty/Boost)")

# Helper function cho Language 31 Strategy
def check_user_lang_31_affinity_train(u_id):
    # Chỉ check trong tập Train để công bằng
    history = train_split[(train_split['user_id_encoded'] == u_id) & (train_split['target'] == 1)]
    # Lấy ra danh sách ngôn ngữ user đã nghe
    songs = history['song_id_encoded'].tolist()
    for s in songs:
        lang = song_info_dict.get(s, {}).get('language', -1)
        if lang == 31.0: return True
    return False

# Hàm tính điểm Hybrid (Cập nhật Logic Phase 4)
def hybrid_predict_score_v4(u_id, s_id, context_sid, has_lang31_affinity):
    # 1. Content Score
    try: cb_score = cosine_sim[context_sid][s_id]
    except: cb_score = 0
    
    # 2. CF Score
    cf_score = predict_cf_raw(u_id, s_id)
    
    # Base Weighted Score
    raw_score = 0.4 * cb_score + 0.6 * cf_score
    
    # 3. APPLY TACTICS (PHASE 4)
    info = song_info_dict.get(s_id, {})
    artist = str(info.get('artist_name', ''))
    genre = str(info.get('genre_ids', ''))
    length_ms = info.get('song_length', 0)
    language = info.get('language', -1)
    
    # Tactic 1: Penalty Various Artists (x0.7)
    if artist == "Various Artists": 
        raw_score *= 0.7
        
    # Tactic 1b: Penalty Genre 465 (x0.5) - Giảm điểm để tăng đa dạng
    if '465' in genre: 
        raw_score *= 0.5
        
    # Tactic 2: High Trust Bonus (x1.25)
    if artist in high_trust_artists: 
        raw_score *= 1.25
        
    # Tactic 3: Gold Duration (3.5 - 4.2 mins) -> (x1.1)
    # 3.5p = 210000ms, 4.2p = 252000ms
    if 210000 <= length_ms <= 252000: 
        raw_score *= 1.1
        
    # Tactic 4: Language 31 Affinity (x1.5)
    if language == 31.0 and has_lang31_affinity:
        raw_score *= 1.5
        
    return raw_score

# --- THỰC HIỆN TEST PRECISION@10 ---
sample_test_users = np.unique(test_users)
np.random.shuffle(sample_test_users)
sample_test_users = sample_test_users[:100] # Test trên 100 user ngẫu nhiên

hits = 0
total_cases = 0

print("Running simulation...")
for u_id in sample_test_users:
    # 1. Lấy đáp án thật (Ground Truth) từ tập Test
    user_test_logs = test_split[(test_split['user_id_encoded'] == u_id) & (test_split['target'] == 1)]
    if user_test_logs.empty: continue 
    
    target_songs = set(user_test_logs['song_id_encoded'].values)
    
    # 2. Lấy ngữ cảnh (Context) từ tập Train
    # Giả sử user đang nghe bài hát cuối cùng họ tương tác trong quá khứ
    user_train_logs = train_split[train_split['user_id_encoded'] == u_id]
    if user_train_logs.empty: continue
    context_sid = user_train_logs.iloc[0]['song_id_encoded']
    
    # 3. Check Affinity (Lang 31)
    has_lang31 = check_user_lang_31_affinity_train(u_id)
    
    # 4. Tạo danh sách Candidates (100 bài ngẫu nhiên + các bài Target)
    # Trong thực tế ta sẽ rank tất cả, nhưng simulation thì lấy mẫu cho nhanh
    negatives = np.random.choice(n_songs, 100)
    candidates = list(target_songs) + list(negatives)
    
    # 5. Chấm điểm từng candidate
    scored_candidates = []
    for cid in candidates:
        score = hybrid_predict_score_v4(u_id, cid, context_sid, has_lang31)
        scored_candidates.append((cid, score))
    
    # 6. Xếp hạng & Lấy Top 10
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_10 = [x[0] for x in scored_candidates[:10]]
    
    # 7. Kiểm tra độ trúng (Hit)
    # Nếu trong Top 10 có ít nhất 1 bài nằm trong Target -> Hit
    if any(s in target_songs for s in top_10):
        hits += 1
    
    total_cases += 1

hit_rate = hits / total_cases if total_cases > 0 else 0

print(f"\n📊 KẾT QUẢ SIMULATION TRÊN {total_cases} CASES:")
print(f"✅ Hit Rate@10: {hit_rate:.2%}")
print(f"   (Tỷ lệ model gợi ý trúng ít nhất 1 bài user thực sự thích trong top 10)")

# --- 6. TỔNG KẾT ---
print("\n--- TỔNG KẾT PHASE 5 ---")
threshold_hit_rate = 0.35 # Ngưỡng kỳ vọng (35%)

if hit_rate > threshold_hit_rate:
    print(f"🌟 SUCCESS: Hit Rate ({hit_rate:.2%}) vượt ngưỡng {threshold_hit_rate:.0%}.")
    print("   Chiến thuật Penalty (VA, 465) và Boost (HighTrust, Lang31) hoạt động hiệu quả.")
    print("   -> SẴN SÀNG CHO PHASE 6 (DEPLOYMENT).")
else:
    print(f"⚠️ WARNING: Hit Rate ({hit_rate:.2%}) thấp hơn kỳ vọng.")
    print("   -> Cần quay lại Phase 3/4 để tinh chỉnh trọng số Content-Based hoặc nới lỏng Penalty.")