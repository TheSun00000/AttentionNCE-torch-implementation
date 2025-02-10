# AttentionNCE: Contrastive Learning with Instance Attention

This repository contains a minimalist implementation of AttentionNCE: Contrastive Learning with Instance Attention. [[openreview paper]](https://openreview.net/forum?id=FfHGAAoSVJ)

## Abstract:
Contrastive learning has found extensive applications in computer vision, natural language processing, and information retrieval, significantly advancing the fron- tier of self-supervised learning. However, the limited availability of labels poses challenges in contrastive learning, as the positive and negative samples can be noisy, adversely affecting model training. To address this, we introduce instance- wise attention into the variational lower bound of contrastive loss, and proposing the AttentionNCE loss accordingly. AttentioNCE incorporates two key compo- nents that enhance contrastive learning performance: First, it replaces instance- level contrast with attention-based sample prototype contrast, helping to mitigate noise disturbances. Second, it introduces a flexible hard sample mining mecha- nism, guiding the model to focus on high-quality, informative samples. Theoret- ically, we demonstrate that optimizing AttentionNCE is equivalent to optimizing the variational lower bound of contrastive loss, offering a worst-case guarantee for maximum likelihood estimation under noisy conditions. Empirically, we apply AttentionNCE to popular contrastive learning frameworks and validate its effec- tiveness.

## Usage

### Training the Model
Run the training script with the following command:

```bash
python train.py [OPTIONS]
```

### Command-Line Arguments
The training script supports the following arguments:

| Argument          | Type   | Default Value | Description |
|------------------|--------|---------------|-------------|
| `--epochs`       | `int`  | 400           | Number of epochs for training. |
| `--batch_size`   | `int`  | 256           | Batch size for training. |
| `--lr`           | `float`| 1e-3          | Learning rate. |
| `--save_dir`     | `str`  | "checkpoints" | Directory to save model checkpoints. |
| `--n_eval_epochs` | `int` | 5             | Frequency of evaluation (in epochs). |
| `--n_save_epochs` | `int` | 20            | Frequency of checkpoint saving (in epochs). |

### Example Usage
Train the model with default parameters:

```bash
python train.py
```

Train with a custom learning rate and batch size:

```bash
python train.py --lr 5e-4 --batch_size 128
```

### Checkpoints
Model checkpoints are organized as follows:

|Batch Size | Positive Views | Checkpoint Link |
|-----------|----------------|-----------------|
|64         | 4              | [Download](https://drive.google.com/file/d/1Nx4U5U73kDX6PdmoCJ0ZfXkKGaMEonG6/view?usp=sharing)   |
|128        | 4              | [Download](https://drive.google.com/file/d/1Nx4U5U73kDX6PdmoCJ0ZfXkKGaMEonG6/view?usp=sharing)   |
|256        | 4              | [Download](https://drive.google.com/file/d/1Nx4U5U73kDX6PdmoCJ0ZfXkKGaMEonG6/view?usp=sharing)   |
|256        | 1              | [Download](https://drive.google.com/file/d/1Nx4U5U73kDX6PdmoCJ0ZfXkKGaMEonG6/view?usp=sharing)   |
|256        | 2              | [Download](https://drive.google.com/file/d/1Nx4U5U73kDX6PdmoCJ0ZfXkKGaMEonG6/view?usp=sharing)   |
|256        | 6              | [Download](https://drive.google.com/file/d/1Nx4U5U73kDX6PdmoCJ0ZfXkKGaMEonG6/view?usp=sharing)   |

## Linear Evaluation

A linear evaluation script is provided to train a classifier using a pretrained model.

### Running the Evaluation
Execute the script with:

```bash
python linear_eval.py [OPTIONS]
```

### Command-Line Arguments
The evaluation script supports the following arguments:

| Argument        | Type   | Default Value           | Description |
|----------------|--------|-------------------------|-------------|
| `--model_path` | `str`  | "checkpoint_epoch_5.pth" | Path to pretrained model. |
| `--batch_size` | `int`  | 512                     | Batch size for evaluation. |
| `--epochs`     | `int`  | 100                     | Number of epochs for training. |
| `--log_dir`    | `str`  | "./logs"               | Directory to save log file. |

### Example Usage
Run linear evaluation with default settings:

```bash
python linear_eval.py
```

Specify a different model and batch size:

```bash
python linear_eval.py --model_path checkpoint_epoch_10.pth --batch_size 256
```

## License
This project is licensed under the MIT License.

