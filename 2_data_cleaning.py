import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer

#Chỉ lấy 20000 bản ghi bài hát và 5000 bản ghi người dùng. Cách làm:
#1. Lấy unique song_id và msno từ tập train (2000000 bản ghi ngẫu nhiên)
train_df = pd.read_csv('Data Processing - Python\datasets\\recommendation challenge (Music)\\train.csv\\train.csv').sample(n=2000000, random_state=42).reset_index(drop=True)
#2. Lọc song_df và member_df chỉ giữ lại các bản ghi có song_id và msno trong tập train
valid_song_id = train_df['song_id'].value_counts().head(20000).index
valid_member_id = train_df['msno'].value_counts().head(5000).index
song_df = pd.read_csv('Data Processing - Python\datasets\\recommendation challenge (Music)\songs.csv\songs.csv')
song_df = song_df[song_df['song_id'].isin(valid_song_id)]
#3. Nếu số bản ghi còn lại của song_df > 20000 thì lấy ngẫu nhiên 20000 bản ghi
if len(song_df)>20000:
    song_df = song_df.sample(n=20000, random_state=42).reset_index(drop=True)
    
member_df = pd.read_csv('Data Processing - Python\datasets\\recommendation challenge (Music)\members.csv\members.csv')
member_df = member_df[member_df['msno'].isin(valid_member_id)]
#4. Nếu số bản ghi còn lại của member_df > 5000 thì lấy ngẫu nhiên 5000 bản ghi
if len(member_df)>5000:
    member_df = member_df.sample(n=5000, random_state=42).reset_index(drop=True)

#I. Xử lý dữ liệu cá nhân người dùng và dữ liệu bài hát
#2.2.1. Xử lý missing values
song_df = song_df[['song_id', 'artist_name', 'composer', 'song_length', 'language', 'genre_ids']]
song_df['composer'] = song_df['composer'].fillna('Unknown')
song_df['genre_ids'] = song_df['genre_ids'].fillna('Unknown')

#2.2.2. Xử lý dữ liệu trùng lặp
song_df = song_df.drop_duplicates('song_id')
member_df = member_df.drop_duplicates('msno')

#2.2.3. Xử lý outliers
bd_columns = pd.to_numeric(member_df['bd'], errors='coerce')
repl = int(bd_columns[(bd_columns>=15) & (bd_columns<=65)].mean())
member_df['bd'] = member_df['bd'].apply(lambda x: x if 15<=x<=65 else repl)

language_columns = pd.to_numeric(song_df['language'], errors='coerce')
repl2 = int(language_columns[language_columns>=0].mean())
song_df['language'] = song_df['language'].apply(lambda x: x if x>=0 else repl2)
song_df = song_df[(song_df['song_length']>=60000) & (song_df['song_length']<=360000)].reset_index(drop=True)

#.2.2.4. Chuẩn hóa dữ liệu
#Chuẩn hóa tuổi về khoảng [0,1]
tmp_bd = member_df['bd'].clip(lower=15, upper=65)
member_df['bd'] = (tmp_bd-15)/(65-15)
#Chuẩn hóa giới tính về dạng số
gender_map = {'male': 1, 'female': 0}
member_df['gender'] = member_df['gender'].map(gender_map)
member_df['city'] = member_df['city'].fillna(0)

#Xử lý missing giới tính bằng KNN, sử dụng city và bd làm features để tính khoảng cách
imputer_data = member_df[['gender','city','bd']].copy()
print("City unique: ", len(imputer_data['city'].unique()))
print("Bd unique: ", len(imputer_data['bd'].unique()))
knn_imputer = KNNImputer(n_neighbors=30)
imputed_data = knn_imputer.fit_transform(imputer_data)
imputed_gender = imputed_data[:,2]
member_df['gender'] = imputed_gender
member_df['gender'] = member_df['gender'].apply(lambda x: 0 if x<0.5 else 1)
print(member_df['gender'].value_counts())

#2.2.5. Vectorize
mlb = MultiLabelBinarizer()
encoded_genres = mlb.fit_transform(song_df['genre_ids'].str.split('|'))
genre_df = pd.DataFrame(encoded_genres, columns=mlb.classes_)
svd = TruncatedSVD(n_components=10, random_state=42)
genre_compressed = svd.fit_transform(genre_df)

col_names = [f'genre_{i+1}' for i in range(10)]
genre_df_svd = pd.DataFrame(genre_compressed, columns=col_names).reset_index(drop=True)

#Sử dụng TF-IDF để vectorize text_soup
song_df['text_soup'] = song_df['artist_name'] + " " + song_df['composer']
tfidf = TfidfVectorizer(stop_words='english', min_df=1, max_features=100)
text_matrix = tfidf.fit_transform(song_df['text_soup']).toarray()
text_matrix = pd.DataFrame(text_matrix, columns=tfidf.get_feature_names_out()).reset_index(drop=True)

#Sử dụng OneHotEncoder để mã hóa ngôn ngữ
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')
language_encoded = ohe.fit_transform(song_df[['language']]).reset_index(drop=True)
# -> 10 features

#Sử dụng MinMaxScaler để chuẩn hóa độ dài bài hát về [0,1]
scaler = MinMaxScaler()
song_len = scaler.fit_transform(song_df[['song_length']])
song_len = pd.DataFrame(song_len, columns=['song_length_scaled']).reset_index(drop=True)

#Sử dung LabelEncoder để mã hóa song_id
le = LabelEncoder()
song_ids = le.fit_transform(song_df['song_id'])
song_ids = pd.DataFrame(song_ids, columns=['song_id_encoded']).reset_index(drop=True)

le_user = LabelEncoder()
le_user.fit(member_df['msno'])
member_df['user_id_encoded'] = le_user.transform(member_df['msno'])

#Kết hợp tất cả các features đã xử lý
final_song_df = pd.concat([song_ids, text_matrix, song_len, genre_df_svd, language_encoded], axis=1).reset_index(drop=True)
print(final_song_df.info())

#Mapping dữ liệu gốc song_id với final_song_df
song_mapping = song_df[['song_id','artist_name','composer','song_length','language','genre_ids']].copy()
song_mapping['song_id_encoded'] = song_ids['song_id_encoded']
song_maping = song_mapping[['song_id_encoded','song_id','artist_name','composer','song_length','language','genre_ids']]
song_mapping = song_mapping.drop_duplicates('song_id_encoded')

#II. Xử lý dữ liệu lịch sử nghe nhạc của người dùng

#2.2.1. Lấy các dòng có song_id và msno tồn tại trong song_df và member_df
train_df = train_df[train_df['song_id'].isin(song_df['song_id'])]
train_df = train_df[train_df['msno'].isin(member_df['msno'])]


#2.2.2. Giữ lại thông tin cần thiết để xây dựng mô hình CF
train_df['song_id_encoded'] = le.transform(train_df['song_id'])
train_df['user_id_encoded'] = le_user.transform(train_df['msno'])
final_train_df = train_df[['user_id_encoded', 'song_id_encoded', 'target']].reset_index(drop=True)

#III. Lưu file csv
final_song_df.to_csv('Data Processing - Python\\final_datasets\processed_songs.csv', index=False)
final_train_df.to_csv('Data Processing - Python\\final_datasets\processed_train.csv', index=False)
song_mapping.to_csv('Data Processing - Python\\final_datasets\song_mapping.csv', index=False)
member_df.to_csv('Data Processing - Python\\final_datasets\members_mapping.csv', index=False)