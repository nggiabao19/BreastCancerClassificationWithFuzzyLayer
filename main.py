# File dành cho chạy local
import os
import time
import importlib
from data_loader_mvindr import CsvImageDataset, prepare_data_splits
from src.visualize import plot_training_history, evaluate_and_visualize

def main():
    # Đường dẫn local (chỉnh lại theo thư mục của bạn)
    dataset_path = os.path.join("data", "images")
    csv_path = os.path.join("data", "manifest.csv")
    model_name = "resnet50v2_fuzzyFC"  # Tên mô hình (Tương ứng với tên file trong /src)

    os.makedirs("results", exist_ok=True)

    try:
        # Khởi tạo CsvImageDataset ở chế độ local (spark=None)
        data_loader = CsvImageDataset(None, dataset_path, csv_path, image_shape=(128, 128), channels=3)

        # Tải ảnh và nhãn từ local
        images, labels = data_loader.load_images_and_labels()
        if len(images) == 0:
            raise ValueError("No images loaded from the dataset. Check the dataset path and CSV file.")

        # Tăng cường dữ liệu
        images, labels = data_loader.apply_data_augmentation(images, labels)

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

        # Visualize and evaluate (save locally)
        plot_training_history(history, model_name, save_dir='results', spark=None)
        evaluate_and_visualize(model, test_img, test_label, model_name, save_dir='results', spark=None)

        # Save model locally
        local_model_path = os.path.join('results', f"{model_name}_model.h5")
        model.save(local_model_path)
        print(f"[INFO] Model saved to {local_model_path}")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise

if __name__ == "__main__":
    main()