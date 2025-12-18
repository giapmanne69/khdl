import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#Lấy dữ liệu
train_df = pd.read_csv("Data Processing - Python\\final_datasets\processed_train.csv")
song_df = pd.read_csv("Data Processing - Python\\final_datasets\song_mapping.csv")
member_df = pd.read_csv("Data Processing - Python\\final_datasets\members_mapping.csv")
df_merged = train_df.merge(song_df[['song_id_encoded','artist_name','genre_ids','language','song_length']], on='song_id_encoded', how='left')
df_merged = df_merged.merge(member_df[['user_id_encoded','gender','bd']], on='user_id_encoded', how='left')
sns.set_style("whitegrid")

#1. Ta sẽ sử dụng Stacked Bar Chart để tìm hiểu top 10 về độ phổ biến của thể loại, nghệ sĩ và ngôn ngữ
def plot_stacked_bar(data, column, title, top_k=10):
    top_items = df_merged[column].value_counts().head(top_k).index
    df_filtered = data[data[column].isin(top_items)]
    
    cross_tab_count = pd.crosstab(df_filtered[column], df_filtered['target'])
    cross_tab_pct = pd.crosstab(df_filtered[column], df_filtered['target'], normalize='index') #Chuẩn hóa về khoảng [0,1]
    
    cross_tab_count['total'] = cross_tab_count.sum(axis=1)
    cross_tab_count = cross_tab_count.sort_values('total', ascending=True)
    cross_tab_count = cross_tab_count.drop('total', axis=1)
    
    cross_tab_pct = cross_tab_pct.sort_values(by=1, ascending=True)
    
    # Vẽ biểu đồ dựa trên SỐ LƯỢNG (Count)
    ax = cross_tab_count.plot(kind='barh', stacked=True, figsize=(12, 7), color=['#d62728', '#2ca02c'])
    
    plt.title(title, fontsize=14)
    plt.xlabel('Số lượt tương tác (Độ phổ biến)', fontsize=12) # Nhãn trục thể hiện số lượng
    plt.ylabel(column, fontsize=12)
    plt.legend(title='Trạng thái', labels=['Không nghe lại (0)', 'Nghe lại (1)'], loc='lower right')
    
    # Hiển thị % lên biểu đồ (Annotation)
    for n, x in enumerate(cross_tab_count.index):
        # Lấy giá trị đếm để xác định vị trí chữ
        count_0 = cross_tab_count.loc[x, 0]
        count_1 = cross_tab_count.loc[x, 1]
        
        # Lấy giá trị % để ghi nội dung chữ
        pct_0 = cross_tab_pct.loc[x, 0]
        pct_1 = cross_tab_pct.loc[x, 1]
        
        # Ghi % lên phần màu đỏ (0)
        if count_0 > 0:
            plt.text(count_0/2, n, f"{pct_0*100:.0f}%", 
                     ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
        # Ghi % lên phần màu xanh (1) 
        # Vị trí x = độ dài thanh đỏ + 1/2 độ dài thanh xanh
        if count_1 > 0:
            plt.text(count_0 + count_1/2, n, f"{pct_1*100:.0f}%", 
                     ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.show()
    
# Gọi hàm vẽ cho Nghệ sĩ
plot_stacked_bar(df_merged, 'artist_name', 'Tỷ lệ Nghe lại của Top 10 Nghệ sĩ phổ biến nhất')
# Gọi hàm vẽ cho Thể loại
plot_stacked_bar(df_merged, 'genre_ids', 'Tỷ lệ Nghe lại của Top 10 Thể loại phổ biến nhất')
# Gọi hàm vẽ cho Ngôn ngữ
plot_stacked_bar(df_merged, 'language', 'Tỷ lệ Nghe lại của Top 10 Ngôn ngữ phổ biến nhất')

#2. Sử dụng histogram để vẽ phẩn bổ độ dài bài hát
song_len_minutes = df_merged['song_length']/60000
song_len_minutes = song_len_minutes[(song_len_minutes>1) & (song_len_minutes<6)]

sns.histplot(song_len_minutes, bins=30, kde=True, color='purple')
plt.title('Histogram: Phân bổ độ dài bài hát (phút)', fontsize=14)
plt.xlabel('Độ dài (Phút)', fontsize=12)
plt.ylabel('Số lượng bài hát', fontsize=12)
plt.axvline(song_len_minutes.mean(), color='red', linestyle='--', label=f'Trung bình: {song_len_minutes.mean():.2f} phút')
plt.legend()
plt.show()

#3. Sử dụng heatmap để xem quan hệ giữa ngôn ngữ và thể loại. 
top_lang = df_merged['language'].value_counts().head(10).index
top_gen = df_merged['genre_ids'].value_counts().head(10).index

df_subset = df_merged[df_merged['language'].isin(top_lang) & df_merged['genre_ids'].isin(top_gen)]

# Tạo bảng Pivot: Đếm số lượng bài hát tại mỗi cặp (Ngôn ngữ, Thể loại)
pivot_table = pd.pivot_table(df_subset, index='genre_ids', columns='language', values='target', aggfunc='count', fill_value=0)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Heatmap: Phân bổ số lượng bài hát theo (Thể loại vs Ngôn ngữ)', fontsize=14)
plt.ylabel('Thể loại (Genre ID)')
plt.xlabel('Ngôn ngữ (Language ID)')
plt.show()

#4. Xem tỉ lệ nghe lại của từng giới tính.
plt.figure(figsize=(8, 5))
# Tính tỷ lệ nghe lại trung bình theo giới tính
gender_target = df_merged.groupby('gender')['target'].mean().reset_index()

sns.barplot(data=gender_target, x='gender', y='target', palette=['#ff9999', '#66b3ff'])
plt.axhline(0.5, color='gray', linestyle='--', label='Trung bình chung (0.5)')

plt.title('Tỷ lệ nghe lại (Target Mean) theo Giới tính', fontsize=14)
plt.ylabel('Xác suất nghe lại', fontsize=12)
plt.ylim(0, 0.7) # Giới hạn trục y để nhìn rõ sự chênh lệch
for index, row in gender_target.iterrows():
    plt.text(index, row.target + 0.01, f"{row.target:.2f}", ha='center', fontweight='bold')
plt.show()

#5. Nam và nữ có gu âm nhạc khác nhau nhiều không?
df_top_gen = df_merged[df_merged['genre_ids'].isin(top_gen)]
plt.figure(figsize=(12, 6))
# Vẽ biểu đồ so sánh số lượng nghe của Nam và Nữ trên từng thể loại
sns.countplot(data=df_top_gen, x='genre_ids', hue='gender', 
              palette=['#ff9999', '#66b3ff'], order=top_gen)

plt.title('Sở thích Thể loại nhạc: Nam vs Nữ', fontsize=14)
plt.xlabel('Thể loại (Genre ID)', fontsize=12)
plt.ylabel('Số lượt nghe', fontsize=12)
plt.legend(title='Giới tính')
plt.show()

#6. Phân tích mối tương quan giữa độ tuổi và thể loại.
# 1.1 Khôi phục tuổi thật (Reverse Scaling)
# Công thức (dựa trên Giai đoạn 2): Scaled = (Raw - 15) / 50  => Raw = Scaled * 50 + 15
df_merged['real_age'] = df_merged['bd'] * 50 + 15
df_merged['real_age'] = df_merged['real_age'].round().astype(int)

# 1.2 Phân nhóm tuổi (Binning)
def age_group(age):
    if age <= 20: return '1. Teen (15-20)'
    elif age <= 30: return '2. Young Adult (21-30)'
    elif age <= 45: return '3. Adult (31-45)'
    else: return '4. Senior (>45)'

df_merged['age_group'] = df_merged['real_age'].apply(age_group)

# 1.3 Biểu đồ Tỷ lệ Nghe lại theo Nhóm tuổi
plt.figure(figsize=(10, 6))
age_target = df_merged.groupby('age_group')['target'].mean().reset_index()
sns.barplot(data=age_target, x='age_group', y='target', palette='Blues_d')
plt.axhline(0.5, color='red', linestyle='--', label='Trung bình (0.5)')
plt.title('Tỷ lệ nghe lại (Target Mean) theo Nhóm tuổi', fontsize=14)
plt.ylabel('Xác suất nghe lại')
plt.ylim(0, 0.8) # Để nhìn rõ sự chênh lệch
for i, v in enumerate(age_target['target']):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()

# 1.4 Heatmap: Nhóm tuổi nào thích Thể loại nào?
# Lấy Top 5 thể loại phổ biến nhất
top_genres = df_merged['genre_ids'].value_counts().head(5).index
df_age_genre = df_merged[df_merged['genre_ids'].isin(top_genres)]

# Tạo bảng pivot đếm số lượng
age_genre_pivot = pd.pivot_table(df_age_genre, values='target', index='genre_ids', columns='age_group', aggfunc='count')

plt.figure(figsize=(10, 6))
sns.heatmap(age_genre_pivot, annot=True, fmt='.0f', cmap="YlGnBu")
plt.title('Số lượng nghe theo (Thể loại vs Nhóm tuổi)', fontsize=14)
plt.ylabel('Thể loại (Genre ID)')
plt.show()

"""
1. Top 10 nghệ sĩ phổ biến nhất đều có độ nghe lại tốt, đặc biệt là Jason Mraz. Tôi muốn cộng điểm cho các nghệ sĩ có tỉ lệ nghe lại trên 65%.
Trong 10 nghệ sĩ, Various Artists có số lượng nghe nhiều nhất và vượt trội. Do đó, đây là 1 gold feature.

2. Thể loại 465 đang độc tôn quá mạnh, nhưng tỉ lệ nghe lại không quá vượt trội. Mô hình cần giới hạn mức gợi ý cho thể loại này.

3. Các thể loại khác cũng cho thấy sự cân bằng hoặc chênh lệch không đáng kể. Vậy nên ta có thể coi thể loại là 1 gold feature, 
nhưng cần tinh chỉnh.

4. Về ngôn ngữ, ta có 4 ngôn ngữ chiếm hầu hết số bài là : 3, 52, 31, 17. Trong đó, tỉ lệ nghe lại của 31 là cao nhất (57%) 
nhưng độ phổ biến lại thấp trong số 4 ngôn ngữ trên. Vậy nên, việc gợi ý cho ngôn ngữ này là rất phù họp nếu người nghe 
đang nghe bài hát có ngôn ngữ số 31; còn lại thì hầu hết đều nằm trong top đầu.

5. Các bài hát có độ dài từ 3,5-4,2 phút có số lượng nghe lại nhiều nhất. Do đó, ta có thể cộng điểm cho các bài hát có
độ dài trong khoảng này.

6. Nam và nữ có tỉ lệ nghe lại là ngang nhau. Trong việc so sánh lượt nghe các thể loại giữa 2 giới, nam đều chiếm áp đảo.
Điều này có lý do là nam nghe nhiều hơn. Vậy nên không sử dụng giới tính để đánh giá và gợi ý được.

7. Tỉ lệ người từ 21-30 nghe lại nhạc có thể loại 465 là rất cao. Vậy nên cũng cần giới hạn số bài thể loại 465 được gợi ý cho tuổi này.

CHIẾN THUẬT ĐƯA RA CHO GIAI ĐOẠN XÂY MÔ HÌNH:
1. Giảm trọng số của Various Artists xuống 70%, của 465 xuống 50%.
2. Tạo danh sách High-Trust Artists. Nếu gợi ý có nghệ sĩ đạt tỉ lệ nghe lại trên 65%, ta sẽ cộng điểm thưởng để ưu tiên đẩy lên đầu danh sách.
3. Rerank và giới hạn số lượng bài hát được gợi ý thành 5/10 bài với thể loại 465.
4. Nếu phát hiện User từng nghe nhạc ngôn ngữ 31 -> Lập tức gợi ý nhạc có ngôn ngữ 31.
"""