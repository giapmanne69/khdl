import kagglehub
import shutil

path = kagglehub.dataset_download("arjunvankani/recommendation-system-challenge-music-suggestion")
print(path)

source_folder = "C:\Users\DELL\.cache\kagglehub\datasets\\arjunvankani\\recommendation-system-challenge-music-suggestion\\versions\\1"
destination_folder = "E:\Project\TEST KHDL\Data Processing - Python\datasets"
shutil.move(source_folder, destination_folder) #Đoạn này thật ra đang chưa đưa được dữ liệu về đây, nên em kéo thủ công ạ.