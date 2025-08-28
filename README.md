# Breast Cancer Classification with Fuzzy Layer  

## 📌 Introduction  
This project focuses on **classifying mammogram images** into three categories:  
- **Benign**  
- **Malignant**  
- **Normal**  

The goal is to support **early detection of breast cancer** by combining **Deep Learning (ResNet50V2/Xception/MammoViT)** with **Fuzzy Logic layers** to improve interpretability and potentially enhance performance.  

---

## 🗂️ Dataset  
- **Source**: VinDr-Mammo
- **Number of images**: ~1000+ mammogram samples  
- **Format**: Grayscale images  
- **Preprocessing steps**:  
  - Resize images to a fixed resolution  
  - Data augmentation (rotation, flipping, contrast adjustment)  
  - Split into train/validation/test sets  

---

## 🏗️ Model Architecture  
- **Baseline**: ResNet50V2 / Xception / MammoViT
- **Variants**:  
  - **Fuzzy Fully Connected Layer**  
  - **Fuzzy Pooling Layer**  
  - **Hybrid Model (Fuzzy Fully Connected Layer + Fuzzy Pooling Layer)**  

---

## ⚙️ Installation & Usage  

### Requirements  
- Python >= 3.8  
- TensorFlow / Keras  
- NumPy, Pandas, scikit-learn  
- Matplotlib  

### Installation  
```bash
git clone https://github.com/nggiabao19/BreastCancerClassificationWithFuzzyLayer.git
cd BreastCancerClassificationWithFuzzyLayer
pip install -r requirements.txt
```
## ⚒️ Training
- Before training, take a look in line 12 at `main.py`
- Replace value of "model_name" with correct model name (it's file name in src/models)
- After all, run:
`python main.py`
## 📊 Results
- Best Accuracy (Xception Hybrid): 94.79%
  
| Class     | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| Benign    |  95.67    | 97.36  | 96.51    |
| Malignant |  90.59    | 95.06  | 92.77    |
| Normal    |  95.63    | 91.62  | 93.58    |

## 📚 References
- [VinDr-Mammo](https://vindr.ai/datasets/mammo)
- [Fuzzy Pooling](https://arxiv.org/abs/2202.08372)
- [FP-CNN: Fuzzy pooling-based convolutional neural network for lung ultrasound image classification with explainable AI](https://www.sciencedirect.com/science/article/pii/S0010482523008727)

## 🐧 Author
- Bao Nguyen (me)
- Contact: nggiabao19@gmail.com


