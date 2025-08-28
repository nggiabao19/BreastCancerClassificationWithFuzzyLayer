import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import io
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_image_from_bytes(image_bytes, shape, channels=3):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if channels == 3:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        elif channels == 1:
            if image.mode != 'L':
                image = image.convert('L')
        
        image = image.resize(shape)
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        if img_array.shape[-1] != channels:
            return None
            
        return img_array
    except Exception as e:
        print(f"Lỗi khi load ảnh: {e}")
        return None

class CsvImageDataset:
    '''
    Lớp đọc dataset từ HDFS dựa vào file CSV và thực hiện tiền xử lí, tăng cường dữ liệu cho bài toán phân loại
    CSV phải chứa các cột: study_id, image_id (không đuôi) và breast_birads (label)
    '''
    def __init__(self, spark, dataset_path, csv_path, image_shape=(128, 128), channels=3):
        # spark can be None for local mode
        self.spark = spark
        self.dataset_path = dataset_path
        self.csv_path = csv_path
        self.image_shape = image_shape
        self.channels = channels

    def load_images_and_labels(self):
        '''Đọc ảnh và nhãn. Tự động dùng Spark nếu có, ngược lại đọc local.'''
        if self.spark is not None:
            # Đọc CSV từ HDFS bằng Spark
            df = self.spark.read.csv(self.csv_path, header=True).collect()

            # Tạo dictionary cho nhãn
            label_dict = {}
            for row in df:
                study_id = row['study_id']
                image_id = row['image_id']
                breast_birads = row['breast_birads']
                birads_num = int(re.search(r'\d+', breast_birads).group())
                # Chuyển đổi nhãn: 3 -> 0, 4 -> 1, otherwise -> 2
                label = 0 if birads_num == 3 else 1 if birads_num == 4 else 2
                label_dict[(study_id, image_id)] = label

            # Tải ảnh từ HDFS qua Spark
            images_rdd = self.spark.sparkContext.binaryFiles(self.dataset_path)
            images_list = images_rdd.collect()

            images = []
            labels = []
            for filename, content in images_list:
                basename = os.path.basename(filename)
                parts = basename.split('_')
                if len(parts) < 2:
                    continue
                study_id = parts[0]
                image_id = '_'.join(parts[1:]).split('.')[0]
                key = (study_id, image_id)
                if key in label_dict:
                    label = label_dict[key]
                    image_array = load_image_from_bytes(content, self.image_shape, self.channels)
                    if image_array is not None:
                        images.append(image_array)
                        labels.append(label)

            images = np.array(images)
            labels = np.array(labels, dtype=np.int32)
            return images, labels

        # Local mode: đọc CSV bằng pandas và ảnh từ filesystem
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        df = pd.read_csv(self.csv_path)
        required_cols = {"study_id", "image_id", "breast_birads"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV missing required columns: {required_cols}")

        label_dict = {}
        for _, row in df.iterrows():
            study_id = str(row["study_id"]).strip()
            image_id = str(row["image_id"]).strip()
            breast_birads = str(row["breast_birads"]) if not pd.isna(row["breast_birads"]) else ""
            match = re.search(r"\d+", breast_birads)
            if match is None:
                continue
            birads_num = int(match.group())
            label = 0 if birads_num == 3 else 1 if birads_num == 4 else 2
            label_dict[(study_id, image_id)] = label

        image_file_patterns = [
            os.path.join(self.dataset_path, "**", "*.png"),
            os.path.join(self.dataset_path, "**", "*.jpg"),
            os.path.join(self.dataset_path, "**", "*.jpeg"),
            os.path.join(self.dataset_path, "**", "*.bmp"),
        ]
        file_list = []
        for pattern in image_file_patterns:
            file_list.extend(glob.glob(pattern, recursive=True))

        images = []
        labels = []
        for filepath in file_list:
            basename = os.path.basename(filepath)
            parts = basename.split('_')
            if len(parts) < 2:
                continue
            study_id = parts[0]
            image_id = '_'.join(parts[1:]).split('.')[0]
            key = (study_id, image_id)
            if key in label_dict:
                label = label_dict[key]
                try:
                    image = Image.open(filepath)
                    if self.channels == 3 and image.mode != 'RGB':
                        image = image.convert('RGB')
                    elif self.channels == 1 and image.mode != 'L':
                        image = image.convert('L')
                    image = image.resize(self.image_shape)
                    img_array = np.array(image, dtype=np.float32) / 255.0
                    if img_array.shape[-1] != self.channels:
                        continue
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Lỗi khi đọc ảnh local {filepath}: {e}")

        images = np.array(images)
        labels = np.array(labels, dtype=np.int32)
        return images, labels

    def apply_data_augmentation(self, images, labels):
        '''Cân bằng và tăng cường dữ liệu'''
        counter = Counter(labels)
        max_count = max(counter.values())

        aug_images, aug_labels = [], []
        for img, lb in zip(images, labels):
            aug_images.append(img)
            aug_labels.append(lb)

            aug_images.append(tf.image.adjust_contrast(img, 2.0).numpy())
            aug_labels.append(lb)

            aug_images.append(tf.image.adjust_brightness(img, 0.3).numpy())
            aug_labels.append(lb)

        return np.array(aug_images), np.array(aug_labels)

    def plot_distribution(self, labels, title="Phân phối labels"):
        '''Vẽ biểu đồ phân phối labels'''
        cnt = Counter(labels)
        cats = list(cnt.keys())
        vals = list(cnt.values())

        plt.figure(figsize=(8, 6))
        bars = plt.bar(cats, vals)
        plt.ylabel('Số lượng')
        plt.title(title)
        for bar, v in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width()/2, v + 2, str(v), ha='center')
        plt.show()

def prepare_data_splits(images, labels, test_size=0.3, val_size=0.5, random_state=42):
    print("\nChuẩn bị chia dữ liệu...")
    
    images, labels = shuffle(images, labels, random_state=random_state)
    
    train_img, test_img, train_label, test_label = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    val_img, test_img_final, val_label, test_label_final = train_test_split(
        test_img, test_label, test_size=val_size, stratify=test_label, random_state=random_state
    )
    
    print(f"Đã chuẩn bị chia dữ liệu:")
    print(f"  Huấn luyện: {len(train_img)} mẫu")
    print(f"  Xác thực: {len(val_img)} mẫu")  
    print(f"  Kiểm tra: {len(test_img_final)} mẫu")
    
    return (train_img, train_label, val_img, val_label, test_img_final, test_label_final)