# Weight Initialization Effects on Neural Networks

This project reproduces the research on weight initialization strategies for CNNs and transformers, comparing Xavier, He, LeCun, and Random initialization methods using ResNet-18 (on CIFAR-10) and DistilBERT (on IMDB).

## Key Findings

- **CNN Performance**: He initialization achieved the highest accuracy (86.1%) and fastest convergence (6 epochs).
- **Transformer Performance**: Xavier initialization reduced gradient explosion by 41% compared to random.
- **LeCun Performance**: LeCun initialization underperformed despite theoretical advantages.

## Setup and Installation

### Requirements

```
pytorch>=1.10.0
torchvision>=0.11.0
transformers>=4.21.0
datasets>=2.4.0
matplotlib>=3.5.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.64.0
numpy>=1.22.0
```

You can install all required dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Information

- **CIFAR-10**: 50,000 training images across 10 classes
- **IMDB**: 25,000 movie reviews for sentiment analysis (binary)

## Running the Experiments

### Run All Experiments

To run all experiments (all models with all initialization methods):

```bash
python main.py
```

### Run Specific Experiments

To run a specific model with a specific initialization method:

```bash
# Train ResNet-18 with He initialization
python main.py --model resnet --init he

# Train DistilBERT with Xavier initialization
python main.py --model distilbert --init xavier
```

### Evaluate Results

To evaluate existing results without training:

```bash
python main.py --evaluate
```

## Project Structure

```
project_root/
├── main.py                        # Orchestrates experiments
├── analysis-ipynb.py              # Analysis and visualization
├── README.md
├── configs/
│   └── config.yaml                # Configuration parameters
├── init_methods/
│   └── initialization.py         # Weight initialization functions
├── models/
│   ├── train_resnet.py           # Trains ResNet-18 on CIFAR-10
│   └── train_distilbert.py       # Fine-tunes DistilBERT on IMDB
├── utils/
│   ├── evaluate.py               # Evaluates model performance
│   └── utils.py                  # Data loading, saving, etc.

```

## Configuration

The `config.yaml` file contains all hyperparameters and settings:

- Batch sizes (64 for ResNet, 8 for DistilBERT)
- Learning rates (0.001 for ResNet, 5e-5 for DistilBERT)
- Training epochs (10 for ResNet, 3 for DistilBERT)
- Optimizers (Adam for ResNet, AdamW for DistilBERT)

## Customization

You can customize training parameters by editing the `config.yaml` file. For example:

```yaml
cnn:
  batch_size: 128  # change batch size
  learning_rate: 0.0005  # change learning rate
```

## Implementation Details

### Initialization Methods

- **Xavier**: Normalizes initialization for linear layers
- **He**: Scales variance for ReLU activations
- **LeCun**: Optimized for tanh activations
- **Random**: Baseline with weights sampled from N(0, 0.01)

### Metrics Tracked

- Accuracy (training and validation)
- Loss convergence
- Gradient norm for stability analysis
- Convergence speed (epochs to reach 95% of final accuracy)

## Results Visualization

After running experiments, visualizations will be generated in the `results/plots/` directory:

- Training/validation accuracy curves
- Loss curves
- Gradient norm plots
- Comparison bar charts

## Extending the Project

To test additional initialization methods:

1. Add the method to `initialization.py`
2. Update the `init_methods` list in `config.yaml`
3. Run the experiments

## Citation

If you use this code in your research, please cite the original paper:

```
@article{mansour2025weightinitialization,
  title={Effects of Weight Initialization Strategies on CNN and Transformer Performance},
  author={Mansour, Jason},
  year={2025}
}
```
