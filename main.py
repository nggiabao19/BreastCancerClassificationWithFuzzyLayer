# File dành cho tập VinDr
import os
import time
import importlib
from pyspark.sql import SparkSession
from pyspark import SparkConf
from data_loader_mvindr import CsvImageDataset, prepare_data_splits
from src.visualize import plot_training_history, evaluate_and_visualize

def main():
    dataset_path = "hdfs://172.20.201.155:9000/user/ubuntu/dataset/MVINDR-BREAST-MAMMO-VN-V1/images"
    csv_path = "hdfs://172.20.201.155:9000/user/ubuntu/dataset/MVINDR-BREAST-MAMMO-VN-V1/manifest.csv"
    model_name = "resnet50v2_fuzzyFC"  # Tên mô hình (Tương ứng với tên file trong /src)
    app_name = "SparkBreastCancerClassifier"

    # Initialize Spark session
    conf = SparkConf()
    conf.set("spark.hadoop.fs.defaultFS", "hdfs://172.20.201.155:9000")

    spark = SparkSession.builder \
        .appName(f"{model_name.capitalize()} Training") \
        .master("spark://172.20.201.154:7077") \
        .config("spark.driver.memory", "4g") \
        .config(conf=conf) \
        .getOrCreate()

    sc = spark.sparkContext
    sc.addPyFile("data_loader.py")

    try:
        hdfs_models_dir = "hdfs://172.20.201.155:9000/user/ubuntu/result/giabao/models"
        hdfs_plots_dir = "hdfs://172.20.201.155:9000/user/ubuntu/result/giabao/plots"

        fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
        fs.mkdirs(sc._jvm.org.apache.hadoop.fs.Path(hdfs_models_dir))
        fs.mkdirs(sc._jvm.org.apache.hadoop.fs.Path(hdfs_plots_dir))

        # Khởi tạo CsvImageDataset
        data_loader = CsvImageDataset(sc, dataset_path, csv_path, image_shape=(128, 128), channels=3)

        # Tải ảnh và nhãn từ HDFS
        images, labels = data_loader.load_images_and_labels()
        if len(images) == 0:
            raise ValueError("No images loaded from the dataset. Check the dataset path and CSV file.")

        # Cân bằng và tăng cường dữ liệu
        images, labels = data_loader.balance_and_augment(images, labels)

        # Chia dữ liệu
        (train_img, train_label, val_img, val_label, test_img, test_label) = prepare_data_splits(
            images, labels
        )

        # Dynamic import of train function
        model_module = importlib.import_module(f"src.models.{model_name}")
        train_func = getattr(model_module, f"train_{model_name}")

        # Train and evaluate model
        print(f"[INFO] Starting {model_name} model training...")
        start_time = time.time()

        model, history = train_func(train_img, train_label, val_img, val_label)

        duration = time.time() - start_time
        print(f"[INFO] Training completed in {duration:.2f} seconds.")

        # Visualize and evaluate
        plot_training_history(history, model_name, hdfs_plots_dir, spark)
        evaluate_and_visualize(model, test_img, test_label, model_name, save_dir=hdfs_plots_dir, spark=spark)

        # Save model
        local_model_path = f"/tmp/{model_name}_model.h5"
        model.save(local_model_path)
        print(f"[INFO] Model saved locally to {local_model_path}")

        hdfs_model_path = f"{hdfs_models_dir}/{model_name}_model.h5"
        hadoop_path = sc._jvm.org.apache.hadoop.fs.Path(hdfs_model_path)
        fs.copyFromLocalFile(False, True, sc._jvm.org.apache.hadoop.fs.Path(local_model_path), hadoop_path)
        print(f"[INFO] Model copied to HDFS at {hdfs_model_path}")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise

    finally:
        print("[INFO] Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    main()