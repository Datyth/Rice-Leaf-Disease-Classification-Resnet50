# Rice-Leaf-Disease-Classification-Resnet50
This project implements a ResNet50 model from scratch to classify various diseases in rice plant leaves. This model can help farmers and agricultural researchers quickly identify common rice diseases through image analysis.

Give me a star if you like it :D
## 1. Project Structure
Rice-Leaf-Disease-Classification-Resnet50/ 

  ├── CNN_model.py     # Source code implementing the ResNet50 model
  
  ├── README.md        # Documentation
  
  └── .gitignore       # Git ignore configuration

## 2. Installation
```bash
# Clone repository
git clone https://github.com/Datyth/Rice-Leaf-Disease-Classification-Resnet50.git
cd Rice-Leaf-Disease-Classification-Resnet50

# Install required libraries
pip install torch torchvision
pip install matplotlib
pip install numpy
pip install scikit-learn
```
## 3. Dataset
The dataset used in this project is the Rice Leaf Disease dataset, downloaded from kaggle, which includes images of rice leaves under various conditions, both healthy and diseased. The dataset has been augmented to create a more diverse and rich collection of images.

The dataset includes more than 10 000 samples with 10 different classes:
- bacterial_leaf_blight
- brown_spot
- healthy
- leaf_blast
- leaf_scald
- narrow_brown_spot
- neck_blast
- rice_hispa
- sheath_blight
- tungro

Dataset: "Rice Leaf Diseases Detection". Kaggle, 2023.  [https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection/data](https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection/data)

This is some pictures from data training set:
<td align="center">
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/55b604fd-576b-4036-897d-1abc3d2b27f4" alt="healthy" width="300"><br>
      <em>Figure 1.: healthy leaf.</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/43e40268-cc38-41a4-bf90-b7ad795197a5" alt="Bacterial leaf blight" width="300"><br>
      <em>Figure 2.: Bacterial leaf blight.</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/98b7c3fe-e127-4229-8e56-084d7cdf3d99" alt="Brow spot" width="300"><br>
      <em>Figure 3.: Brow spot leaf.</em>
    </td>
  </tr>
</table>
</td>

!! Please check all the folder name after dowloading to make sure they are correct to run model!
## 4. Model detail
Architecture: ResNet50 (from scratch)\
Framwork: PyTorch\
Input size: 224x224x3\
Output: 10 classes\
Training parameters:
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 16
- Epochs: 10
  <td align="center"> 
  <img src="https://github.com/user-attachments/assets/fcf75532-8a5d-4de4-9e84-c8f6acee68e6" alt="healthy" width="1200"> <br>
  <em>ResNet50 architecture.</em>
</td>

## 5. Validation
Validation results indicate that the model achieves an accuracy exceeding 87% within the first 10 epochs.

<td align="center"> 
  <img src="https://github.com/user-attachments/assets/a7e5ae3f-fe98-48e2-ae1b-09d4caceea62" alt="healthy" width="1200"> <br>
  <em>Loss and Accuracy in training and testing with the first 10 epochs.</em>
</td>

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
GitHub: Datyth\
Email: tophatdat.160305@gmail.com
