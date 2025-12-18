import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix

# --- THƯ VIỆN FIREBASE ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="KKBox Real-time Cloud", page_icon="☁️", layout="wide")

# ==========================================
# 1. KẾT NỐI FIREBASE (CLOUD DATABASE)
# ==========================================
@st.cache_resource
def init_firebase():
    """Khởi tạo kết nối Firebase duy nhất một lần"""
    if not firebase_admin._apps:
        try:
            # Lấy key từ st.secrets
            key_content = st.secrets["firebase"]["textkey"]
            
            # Fix lỗi ký tự xuống dòng nếu có trong secret
            try:
                key_dict = json.loads(key_content, strict=False)
            except json.JSONDecodeError:
                fixed_content = key_content.replace('\n', '\\n')
                key_dict = json.loads(fixed_content, strict=False)
                
            cred = credentials.Certificate(key_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Lỗi khởi tạo Firebase: {e}")
            return None
    
    return firestore.client()

try:
    db = init_firebase()
    if db:
        st.sidebar.success("Đã kết nối Database trên Mây! 🟢")
except Exception as e:
    st.error(f"Lỗi kết nối Firebase: {e}")
    st.stop()

def log_interaction(user_id, song_id):
    """Ghi log lên Cloud Firestore"""
    if db:
        doc_ref = db.collection('listening_history').document()
        doc_ref.set({
            'user_id': int(user_id),
            'song_id': int(song_id),
            'timestamp': firestore.SERVER_TIMESTAMP
        })

def get_recent_tracks(user_id, limit=5):
    """Query dữ liệu từ Cloud Firestore (Sắp xếp Python để tránh lỗi Index)"""
    if not db: return []
    try:
        # Lọc user_id
        docs = db.collection('listening_history').where('user_id', '==', int(user_id)).stream()
        
        # Convert sang list
        history_list = []
        for doc in docs:
            data = doc.to_dict()
            if 'song_id' in data:
                # Xử lý timestamp an toàn
                ts = data.get('timestamp')
                history_list.append({
                    'song_id': data['song_id'],
                    'timestamp': ts if ts else datetime.datetime.min
                })
        
        # Sắp xếp giảm dần theo thời gian bằng Python
        history_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Lấy top limit
        song_ids = [item['song_id'] for item in history_list[:limit]]
        return song_ids
    except Exception as e:
        print(f"Lỗi query: {e}")
        return []

def clear_history_cloud(user_id):
    """Xóa lịch sử trên Cloud"""
    if not db: return
    docs = db.collection('listening_history').where('user_id', '==', int(user_id)).stream()
    for doc in docs:
        doc.reference.delete()

# ==========================================
# 2. LOAD DATA & TRAIN MODELS
# ==========================================
@st.cache_data
def load_data():
    possible_paths = ['Data Processing - Python/final_datasets', 'final_datasets', '.', './final_datasets']
    base_path = None
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'processed_songs.csv')):
            base_path = path
            break
    
    if base_path is None: return None, None, None, None, None, None

    try:
        song_features = pd.read_csv(os.path.join(base_path, 'processed_songs.csv'))
        train_df = pd.read_csv(os.path.join(base_path, 'processed_train.csv'))
        song_map = pd.read_csv(os.path.join(base_path, 'song_mapping.csv'))
        member_df = pd.read_csv(os.path.join(base_path, 'members_mapping.csv'))
    except: return None, None, None, None, None, None
    
    song_features = song_features.sort_values('song_id_encoded').reset_index(drop=True)
    song_map = song_map.drop_duplicates('song_id_encoded')
    member_df = member_df.drop_duplicates('user_id_encoded')
    
    song_dict = song_map.set_index('song_id_encoded')[['artist_name', 'genre_ids', 'language', 'song_length']].to_dict('index')
    member_df['real_age'] = (member_df['bd'] * 50 + 15).round().astype(int)
    user_age_dict = member_df.set_index('user_id_encoded')['real_age'].to_dict()
    
    return song_features, train_df, song_map, member_df, song_dict, user_age_dict

song_features, train_df, song_map, member_df, song_info_dict, user_age_dict = load_data()
if song_features is None:
    st.error("❌ Lỗi: Không tìm thấy dữ liệu.")
    st.stop()

@st.cache_resource
def train_models(song_features, train_df):
    feature_cols = [c for c in song_features.columns if c != 'song_id_encoded']
    feature_matrix = song_features[feature_cols].values
    cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
    
    n_users = train_df['user_id_encoded'].max() + 1
    n_songs = train_df['song_id_encoded'].max() + 1
    rows = train_df['user_id_encoded'].values
    cols = train_df['song_id_encoded'].values
    data = train_df['target'].values 
    R_sparse = coo_matrix((data, (rows, cols)), shape=(n_users, n_songs))
    
    svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
    user_factors = svd.fit_transform(R_sparse)
    item_factors = svd.components_.T
    
    train_merged = train_df.merge(pd.DataFrame(song_info_dict).T, left_on='song_id_encoded', right_index=True)
    artist_stats = train_merged.groupby('artist_name')['target'].agg(['count', 'mean'])
    sig_artists = artist_stats[artist_stats['count'] > 50]
    high_trust_artists = sig_artists[sig_artists['mean'] >= 0.65].index.tolist()
    for art in ["Jason Mraz", "Mountain"]:
        if art not in high_trust_artists: high_trust_artists.append(art)
    
    return cosine_sim, user_factors, item_factors, high_trust_artists

with st.spinner('Đang huấn luyện AI...'):
    cosine_sim, user_factors, item_factors, high_trust_artists = train_models(song_features, train_df)

# ==========================================
# 3. REAL-TIME LOGIC & HYBRID ENGINE
# ==========================================
def get_realtime_user_vector(user_id, recent_sids, weight=0.3):
    if user_id < len(user_factors):
        base_vec = user_factors[user_id]
    else:
        base_vec = np.zeros(50)
        
    if not recent_sids: return base_vec
    
    recent_vecs = []
    for sid in recent_sids:
        if sid < len(item_factors):
            recent_vecs.append(item_factors[sid])
            
    if recent_vecs:
        short_term_vec = np.mean(recent_vecs, axis=0)
        return (1 - weight) * base_vec + weight * short_term_vec
    return base_vec

def check_lang31(user_id):
    user_logs = train_df[(train_df['user_id_encoded'] == user_id) & (train_df['target'] == 1)]
    for sid in user_logs['song_id_encoded'].unique():
        if song_info_dict.get(sid, {}).get('language') == 31.0: return True
    return False

def get_recommendations_realtime(user_id, current_song_id, top_k=10, rt_weight=0.3):
    try:
        if current_song_id >= len(cosine_sim): return []
        sim_scores = list(enumerate(cosine_sim[current_song_id]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        candidates = sim_scores[1:61]
    except: return []
    
    recent_sids = get_recent_tracks(user_id)
    dynamic_user_vec = get_realtime_user_vector(user_id, recent_sids, weight=rt_weight)
    
    likes_lang31 = check_lang31(user_id)
    
    final_scores = []
    
    for s_idx, cb_score in candidates:
        try:
            if s_idx >= len(item_factors): cf_score = 0.5
            else:
                cf_score = np.dot(dynamic_user_vec, item_factors[s_idx])
                cf_score = np.clip(cf_score, 0, 1)
        except: cf_score = 0.5
        
        raw_score = 0.3 * cb_score + 0.7 * cf_score
        
        info = song_info_dict.get(s_idx, {})
        artist = str(info.get('artist_name', 'Unknown'))
        genre = str(info.get('genre_ids', ''))
        lang = info.get('language', -1)
        length = info.get('song_length', 0)
        
        notes = []
        if artist == "Various Artists": raw_score *= 0.7; notes.append("Penalty(VA)")
        if '465' in genre: raw_score *= 0.5; notes.append("Penalty(Pop)")
        if artist in high_trust_artists: raw_score *= 1.25; notes.append("HighTrust⭐")
        if lang == 31.0 and likes_lang31: raw_score *= 1.5; notes.append("Lang31🔥")
        if 210000 <= length <= 252000: raw_score *= 1.1; notes.append("GoldTime")
        
        final_scores.append({
            'ID': s_idx, 'Artist': artist, 'Genre': genre, 
            'Language': lang, 'Score': raw_score, 'Tags': ", ".join(notes)
        })
        
    final_scores.sort(key=lambda x: x['Score'], reverse=True)
    
    recs = []
    cnt_465 = 0
    for item in final_scores:
        if len(recs) >= top_k: break
        if '465' in item['Genre']:
            if cnt_465 >= 5: continue
            cnt_465 += 1
        recs.append(item)
        
    return pd.DataFrame(recs)

# ==========================================
# 4. GIAO DIỆN
# ==========================================
st.title("🎧 KKBox Real-time Cloud")
st.caption("Lưu trữ vĩnh viễn trên Firebase Firestore")

with st.sidebar:
    st.header("👤 Hồ sơ User")
    users = [u for u in user_age_dict.keys() if 21 <= user_age_dict[u] <= 30][:50]
    if not users: users = list(user_age_dict.keys())[:50]
    
    selected_user = st.selectbox("Chọn User:", users)
    
    st.divider()
    st.subheader("⚙️ Cấu hình")
    rt_weight = st.slider("Trọng số Real-time:", 0.0, 1.0, 0.3)
    
    st.divider()
    st.subheader("🕒 Lịch sử (Firebase)")
    
    recent_sids = get_recent_tracks(selected_user)
    if recent_sids:
        for sid in recent_sids:
            info = song_info_dict.get(sid, {})
            # HIỂN THỊ NGÔN NGỮ Ở SIDEBAR
            st.text(f"♪ {info.get('artist_name', 'Unknown')} | Lang: {info.get('language', '-')}")
        
        if st.button("Xóa lịch sử trên Cloud"):
            clear_history_cloud(selected_user)
            st.rerun()
    else:
        st.info("Chưa có lịch sử.")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Gợi ý Real-time")
    ulogs = train_df[train_df['user_id_encoded'] == selected_user]
    hids = ulogs['song_id_encoded'].unique() if not ulogs.empty else []
    
    # HIỂN THỊ NGÔN NGỮ TRONG DROPDOWN
    opts = {sid: f"{song_info_dict[sid]['artist_name']} | Lang:{song_info_dict[sid]['language']} (ID:{sid})" 
            for sid in hids if sid in song_info_dict}
    if not opts: opts = {100: "Mẫu: Jason Mraz"}
    
    selected_song_id = st.selectbox("Bài ngữ cảnh gốc:", list(opts.keys()), format_func=lambda x: opts[x])
    
    # HIỂN THỊ CHI TIẾT DƯỚI DROPDOWN
    if selected_song_id in song_info_dict:
        ctx_info = song_info_dict[selected_song_id]
        st.write(f"**Song ID:** `{selected_song_id}` | **Genre:** `{ctx_info['genre_ids']}` | **Lang:** `{ctx_info['language']}`")
    
    if st.button("🔄 Cập nhật Gợi ý", type="primary"):
        with st.spinner("Đang truy vấn Firebase & Tính toán..."):
            recs = get_recommendations_realtime(selected_user, selected_song_id, rt_weight=rt_weight)
            if not recs.empty:
                max_s = recs['Score'].max()
                recs['Match'] = (recs['Score'] / max_s)
                st.dataframe(recs[['ID', 'Artist', 'Genre', 'Language', 'Match', 'Tags']],
                             column_config={
                                 "ID": st.column_config.NumberColumn("Song ID", format="%d"),
                                 "Language": st.column_config.NumberColumn("Lang", format="%.0f"),
                                 "Match": st.column_config.ProgressColumn("Độ hợp", format="%.0f%%")
                             },
                             hide_index=True)
            else:
                st.warning("Không có kết quả.")

with col2:
    st.subheader("🎧 Giả lập Nghe nhạc (Ghi lên Mây)")
    
    # HIỂN THỊ NGÔN NGỮ TRONG DROPDOWN TÌM KIẾM
    all_songs_opts = {sid: f"ID:{sid} | {info['artist_name']} | Lang:{info['language']} | Gen:{info['genre_ids']}" 
                      for sid, info in song_info_dict.items()}
    
    selected_play_song = st.selectbox("Tìm bài hát:", list(all_songs_opts.keys()), format_func=lambda x: all_songs_opts[x])
    repeat_count = st.number_input("Số lần nghe:", 1, 50, 1)
    
    if st.button("▶️ Nghe bài này", type="primary"):
        s_info = song_info_dict.get(selected_play_song, {})
        art_name = s_info.get('artist_name', 'Unknown')
        
        for _ in range(repeat_count):
            log_interaction(selected_user, selected_play_song)
            
        st.toast(f"Đã lưu '{art_name}' lên Firestore! ({repeat_count} lần)", icon="☁️")