# ProViDNet: Leveraging Self-Supervised Learning for Medical Image Segmentation
<img src="./images/ProViDNet.png" width="500px"></img>


## Overview
ProViDNet introduces a novel approach for medical image segmentation by leveraging self-supervised learning, specifically utilizing the DINO (Distillation with No Label) v2 architecture. This model significantly improves prostate cancer segmentation accuracy across various MRI modalities and 3D Transrectal Ultrasound (TRUS) datasets.

## Installation

### Prerequisites

The project requires the following dependencies:
- Python 3.11.3
- PyTorch 2.1.0
- MONAI 1.4.dev2405
- cv2 4.9.0
- UMAP-learn 0.5.5
- Matplotlib 3.7.2
- SimpleITK 2.3.0
- pandas 2.1.1
- numpy 1.26.0

### Setup
To set up the project environment:

```bash
git clone https://github.com/yourGitHub/providnet.git
cd providnet
pip install -r requirements.txt
```



## Usage
To train the model, adjust your configurations in `config.yaml`, then run:

python train.py --ModelName ProViDNet --pretrained_weights <path_to_pretrained_weights> --save_directory ./MODEL/




# For model evaluation:
python evaluate.py --pretrained_weights <path_to_trained_model> --save_pred 1 --save_heatmap 1



## Data Preparation
Your dataset should be organized into directories as specified in `config.yaml`. The script expects MRI and TRUS images along with their corresponding labels.

## Configuration
Modify `config.yaml` to fit the paths and parameters for your specific dataset and training preferences.

## Contributing
We welcome contributions and suggestions. Please submit pull requests or open an issue if you have feedback or proposals.

## License
Specify your project's license here.

## Citation
If you use ProViDNet in your research, please cite our paper:


@article{yourCitation,
    title={ProViDNet: Leveraging Self-Supervised Learning for Medical Image Segmentation},
    author={Authors},
    journal={Journal Name},
    year={Year}
}
