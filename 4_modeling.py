import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix

# --- CẤU HÌNH ---
# Hãy đảm bảo đường dẫn đúng với máy của bạn
SONG_PATH = 'Data Processing - Python\\final_datasets\processed_songs.csv'
TRAIN_PATH = 'Data Processing - Python\\final_datasets\processed_train.csv'
MAPPING_PATH = 'Data Processing - Python\\final_datasets\song_mapping.csv'
MEMBER_PATH = 'Data Processing - Python\\final_datasets\members_mapping.csv'

print("--- 1. LOADING DATA ---")
song_features = pd.read_csv(SONG_PATH)
train_df = pd.read_csv(TRAIN_PATH)
song_map = pd.read_csv(MAPPING_PATH)
member_df = pd.read_csv(MEMBER_PATH)

# --- QUAN TRỌNG: SẮP XẾP ĐỂ INDEX KHỚP VỚI ID ---
# Đảm bảo row 0 là ID 0, row 1 là ID 1... để ma trận tính toán đúng
song_features = song_features.sort_values('song_id_encoded').reset_index(drop=True)

# Fix lỗi trùng lặp index (nếu có)
song_map = song_map.drop_duplicates('song_id_encoded')
member_df = member_df.drop_duplicates('user_id_encoded')

# Dictionary tra cứu Metadata
# Key: song_id_encoded -> Value: {artist, genre, ...}
song_info_dict = song_map.set_index('song_id_encoded')[['artist_name', 'genre_ids', 'language', 'song_length']].to_dict('index')

# Dictionary tra cứu Tuổi User
member_df['real_age'] = (member_df['bd'] * 50 + 15).round().astype(int)
user_age_dict = member_df.set_index('user_id_encoded')['real_age'].to_dict()

print(f"Loaded: Songs {song_features.shape}, Train {train_df.shape}")

# --- 2. LOGIC PRE-CALCULATION (CHIẾN THUẬT) ---
print("\n--- 2. ANALYZING HIGH-TRUST ARTISTS ---")
train_merged = train_df.merge(song_map[['song_id_encoded', 'artist_name']], on='song_id_encoded', how='left')
artist_stats = train_merged.groupby('artist_name')['target'].agg(['count', 'mean'])
significant_artists = artist_stats[artist_stats['count'] > 20]
high_trust_artists = significant_artists[significant_artists['mean'] >= 0.65].index.tolist()

print(f"Detected {len(high_trust_artists)} high-trust artists.")

# --- 3. BUILDING CONTENT-BASED ENGINE ---
print("\n--- 3. BUILDING CONTENT-BASED ENGINE ---")
feature_cols = [c for c in song_features.columns if c != 'song_id_encoded']
feature_matrix = song_features[feature_cols].values

print("Calculating Cosine Similarity Matrix...")
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

def get_content_candidates(song_id_encoded, top_k=50):
    try:
        # Kiểm tra biên (tránh lỗi index out of bounds)
        if song_id_encoded >= len(cosine_sim): return []
        
        sim_scores = list(enumerate(cosine_sim[song_id_encoded]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return sim_scores[1:top_k+1] 
    except IndexError:
        return []

# --- 4. BUILDING CF ENGINE (SKLEARN SVD) ---
print("\n--- 4. BUILDING CF ENGINE (SKLEARN SVD) ---")

n_users = train_df['user_id_encoded'].max() + 1
n_songs = train_df['song_id_encoded'].max() + 1

rows = train_df['user_id_encoded'].values
cols = train_df['song_id_encoded'].values
data = train_df['target'].values 

R_sparse = coo_matrix((data, (rows, cols)), shape=(n_users, n_songs))

svd_sklearn = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd_sklearn.fit_transform(R_sparse)
item_factors = svd_sklearn.components_.T

def predict_cf_score(user_idx, song_idx):
    try:
        if user_idx >= user_factors.shape[0] or song_idx >= item_factors.shape[0]:
            return 0.5 
        score = np.dot(user_factors[user_idx], item_factors[song_idx])
        return np.clip(score, 0, 1) 
    except:
        return 0.5

# --- 5. HYBRID ENGINE ---
print("\n--- 5. HYBRID RECOMMENDATION ENGINE (7 TACTICS) ---")

def check_user_language_affinity(user_id_encoded, target_lang=17.0):
    """Kiểm tra user có thích ngôn ngữ target không (Chiến thuật 4)"""
    user_history = train_df[(train_df['user_id_encoded'] == user_id_encoded) & (train_df['target'] == 1)]
    liked_songs = user_history['song_id_encoded'].tolist()
    for song in liked_songs:
        lang = song_info_dict.get(song, {}).get('language', -1)
        if lang == target_lang: return True
    return False

def hybrid_recommendation(user_id_encoded, current_song_id_encoded, top_k=10):
    candidates = get_content_candidates(current_song_id_encoded, top_k=60)
    final_scores = []
    
    # Lấy thông tin user
    user_age = user_age_dict.get(user_id_encoded, 25)
    user_likes_lang_31 = check_user_language_affinity(user_id_encoded, target_lang=31.0)
    
    # --- PHASE A: SCORING ---
    for song_idx, cb_score in candidates:
        cf_prediction = predict_cf_score(user_id_encoded, song_idx)
        
        # Lấy metadata an toàn bằng .get() để tránh lỗi KeyError
        info = song_info_dict.get(song_idx, {})
        artist = str(info.get('artist_name', 'Unknown'))
        genre = str(info.get('genre_ids', 'Unknown'))
        language = info.get('language', -1)
        length_ms = info.get('song_length', 0)
        
        # ÁP DỤNG 7 CHIẾN THUẬT
        
        # 1. Various Artists Penalty
        if artist == "Various Artists": 
            cb_score *= 0.7; cf_prediction *= 0.7
            
        # 2. High-Trust Bonus
        if artist in high_trust_artists: 
            cf_prediction += 0.2
            
        # 4. Language 17 Bonus
        if language == 31.0 and user_likes_lang_31: 
            cf_prediction += 0.3

        # 5. Gold Duration Bonus (3.5 - 4.5 phút)
        if 210000 <= length_ms <= 252000: 
            cf_prediction += 0.05 

        # 6. Demographic Alignment (21-30 tuổi thích Pop)
        if (21 <= user_age <= 30) and ('465' in genre): 
            cf_prediction += 0.15 
        
        # Weighted Sum
        final_score = 0.4 * cb_score + 0.6 * cf_prediction
        final_scores.append((song_idx, final_score, genre, artist))
    
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    # --- PHASE B: FILTERING & INTERLEAVING ---
    recommendations = []
    count_genre_465 = 0
    MAX_GENRE_465 = 5 # 3. Giới hạn Genre 465
    consecutive_465 = 0 # 7. Xen kẽ thông minh
    
    for item in final_scores:
        if len(recommendations) >= top_k: break
        
        s_idx, score, genre, artist = item
        is_465 = '465' in genre
        
        if is_465:
            if count_genre_465 >= MAX_GENRE_465: continue
            if consecutive_465 >= 2: continue 
            
            count_genre_465 += 1
            consecutive_465 += 1
        else:
            consecutive_465 = 0 
            
        recommendations.append((s_idx, score, artist, genre))
        
    return recommendations

# --- 6. DEMO (ĐÃ FIX LỖI TỰ ĐỘNG CHỌN BÀI) ---
print("\n--- KẾT QUẢ DEMO (SKLEARN VERSION) ---")
try:
    # Tìm User 21-30 tuổi
    target_users = [u for u, age in user_age_dict.items() if 21 <= age <= 30]
    
    # 1. Chọn User ID 
    u_id = 1234
    if u_id not in user_age_dict: 
        if target_users:
            u_id = target_users[0]
            print(f"⚠️ User không có trong dữ liệu. Chuyển sang User ID: {u_id}")
        else:
            u_id = train_df.iloc[0]['user_id_encoded']
            print(f"⚠️ Không tìm thấy user 21-30 tuổi. Lấy user bất kỳ: {u_id}")
    
    # 2. Chọn Song ID (Ngữ cảnh)
    # Tìm trong lịch sử của user này xem họ đã nghe bài nào
    user_logs = train_df[train_df['user_id_encoded'] == u_id]
    
    if not user_logs.empty:
        # Lấy bài đầu tiên user này nghe làm ngữ cảnh
        s_id = user_logs.iloc[0]['song_id_encoded']
    else:
        # Nếu user mới tinh chưa nghe bài nào, lấy bài hát đầu tiên trong kho nhạc
        s_id = list(song_info_dict.keys())[0]
        print("⚠️ User này chưa có lịch sử. Lấy bài hát đầu tiên trong kho làm ngữ cảnh.")

    # Lấy thông tin bài hát để in ra
    ctx_info = song_info_dict.get(s_id, {'artist_name': 'Unknown', 'genre_ids': 'Unknown'})
    
    print(f"User ID: {u_id} | Age: {user_age_dict.get(u_id)}")
    print(f"Context Song (ID {s_id}): {ctx_info['artist_name']} (Genre: {ctx_info['genre_ids']})")
    
    # Chạy gợi ý
    recs = hybrid_recommendation(u_id, s_id, top_k=10)
    
    # In kết quả có cột ID
    print(f"\n{'ID':<6} | {'Artist':<25} | {'Genre':<10} | {'Score':<6} | {'Tactics Applied'}")
    print("-" * 75)
    
    for r in recs:
        s_idx, score, art, gen = r
        notes = []
        if art == "Various Artists": notes.append("Penalized")
        if art in high_trust_artists: notes.append("HighTrust")
        if '465' in gen: notes.append("PopBoost")
        
        # IN CỘT ID (s_idx) ĐẦU TIÊN
        print(f"{s_idx:<6} | {art[:23]:<25} | {gen[:10]:<10} | {score:.4f} | {', '.join(notes)}")
        
except Exception as e:
    print(f"Lỗi Demo: {e}")