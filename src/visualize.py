import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from pyspark.sql import SparkSession

def save_to_hdfs(hdfs_path, content, spark):
    """Save file to HDFS using Hadoop FileSystem API."""
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        hdfs_path_obj = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(hdfs_path)
        fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
        
        # Create parent directories if they don't exist
        parent_dir = hdfs_path_obj.getParent()
        if parent_dir and not fs.exists(parent_dir):
            fs.mkdirs(parent_dir)
        
        # Delete existing file if present (automatic overwrite)
        if fs.exists(hdfs_path_obj):
            print(f"File exists at {hdfs_path}, overwriting...")
            fs.delete(hdfs_path_obj, False)
        
        # Create output stream
        writer = fs.create(hdfs_path_obj, True)  # True for overwrite
        
        try:
            if isinstance(content, str):
                # Write string content directly to HDFS
                writer.write(content.encode('utf-8'))
            elif isinstance(content, bytes):
                # Write bytes content directly to HDFS
                writer.write(content)
            else:
                # Assume content is a local file path
                if os.path.exists(content):
                    with open(content, 'rb') as local_file:
                        buffer_size = 4096
                        while True:
                            buffer = local_file.read(buffer_size)
                            if not buffer:
                                break
                            writer.write(buffer)
                    # Verify file size
                    copied_status = fs.getFileStatus(hdfs_path_obj)
                    local_size = os.path.getsize(content)
                    hdfs_size = copied_status.getLen()
                    if local_size != hdfs_size:
                        print(f"Warning: File size mismatch - Local: {local_size}, HDFS: {hdfs_size}")
                        raise Exception("File copy verification failed")
                    # Clean up temporary file
                    try:
                        os.remove(content)
                    except OSError as e:
                        print(f"Warning: Could not remove temporary file {content}: {e}")
                else:
                    raise FileNotFoundError(f"Local file not found: {content}")
        
        finally:
            # Ensure the output stream is closed
            writer.close()
        
        print(f"File saved successfully to HDFS: {hdfs_path}")
                
    except Exception as e:
        print(f"Error saving to HDFS {hdfs_path}: {e}")
        raise

def verify_png_file(file_path):
    """Verify if PNG file is valid."""
    try:
        with open(file_path, 'rb') as f:
            png_signature = f.read(8)
            expected_signature = b'\x89PNG\r\n\x1a\n'
            if png_signature != expected_signature:
                print(f"Invalid PNG signature in {file_path}")
                return False
        return True
    except Exception as e:
        print(f"Error verifying PNG file {file_path}: {e}")
        return False

def plot_training_history(history, model_name, save_dir='hdfs://172.20.201.155:9000/user/ubuntu/result/giabao/plots', spark=None):
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    
    try:
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        
        epochs = range(1, len(train_loss) + 1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        plt.title(f'{model_name} Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs("/tmp", exist_ok=True)
        tmp_loss_path = f"/tmp/{model_name}_loss_plot.png"
        
        plt.savefig(tmp_loss_path, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        if verify_png_file(tmp_loss_path):
            print(f"PNG file verified: {tmp_loss_path}")
            save_to_hdfs(f"{save_dir}/{model_name}_loss_plot_original.png", tmp_loss_path, spark)
        else:
            raise Exception(f"Generated PNG file is invalid: {tmp_loss_path}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        plt.title(f'{model_name} Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        tmp_acc_path = f"/tmp/{model_name}_accuracy_plot.png"
        
        plt.savefig(tmp_acc_path, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        if verify_png_file(tmp_acc_path):
            print(f"PNG file verified: {tmp_acc_path}")
            save_to_hdfs(f"{save_dir}/{model_name}_accuracy_plot_original.png", tmp_acc_path, spark)
        else:
            raise Exception(f"Generated PNG file is invalid: {tmp_acc_path}")
        
        print(f"Training history plots saved successfully for {model_name}")
        
    except Exception as e:
        print(f"Error plotting training history for {model_name}: {e}")
        raise

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir='hdfs://172.20.201.155:9000/user/ubuntu/result/giabao/plots', spark=None):
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        class_names = ['Benign', 'Malignant', 'Normal']
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(label='Count')

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, fontsize=10)
        plt.yticks(tick_marks, class_names, fontsize=10)

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs("/tmp", exist_ok=True)
        tmp_cm_path = f"/tmp/{model_name}_confusion_matrix.png"
        
        plt.savefig(tmp_cm_path, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        if verify_png_file(tmp_cm_path):
            print(f"PNG file verified: {tmp_cm_path}")
            save_to_hdfs(f"{save_dir}/{model_name}_confusion_matrix_original.png", tmp_cm_path, spark)
        else:
            raise Exception(f"Generated PNG file is invalid: {tmp_cm_path}")
        
        print(f"Confusion matrix saved successfully for {model_name}")
        
    except Exception as e:
        print(f"Error plotting confusion matrix for {model_name}: {e}")
        raise

def generate_classification_report(y_true, y_pred, model_name, save_dir='hdfs://172.20.201.155:9000/user/ubuntu/result/giabao/plots', spark=None):
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    
    try:
        class_names = ['Benign', 'Malignant', 'Normal']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False, digits=4)
        
        report_content = f"""
{model_name} Classification Report
{'='*50}

{report}

Model: {model_name}
Total Samples: {len(y_true)}
Generated on: {spark.sql("SELECT current_timestamp()").collect()[0][0]}
"""
        
        save_to_hdfs(f"{save_dir}/{model_name}_classification_report.txt", report_content, spark)
        print(f"Classification report saved successfully for {model_name}")
        
        return report
        
    except Exception as e:
        print(f"Error generating classification report for {model_name}: {e}")
        raise

def evaluate_and_visualize(model, X_test, y_test, model_name, save_dir='hdfs://172.20.201.155:9000/user/ubuntu/result/giabao/plots', spark=None):
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    
    try:
        print(f"Starting evaluation and visualization for {model_name}...")
        
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        if len(y_pred_classes) != len(y_test):
            raise ValueError(f"Prediction length ({len(y_pred_classes)}) doesn't match test length ({len(y_test)})")
        
        plot_confusion_matrix(y_test, y_pred_classes, model_name, save_dir, spark)
        
        report = generate_classification_report(y_test, y_pred_classes, model_name, save_dir, spark)
        print(f"\n{model_name} Classification Report:\n{report}")
        
        print(f"Evaluation and visualization completed successfully for {model_name}")
        
    except Exception as e:
        print(f"Error in evaluate_and_visualize for {model_name}: {e}")
        raise