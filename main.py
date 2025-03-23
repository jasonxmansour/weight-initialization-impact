import os
import json
import argparse
import logging
from datetime import datetime

from utils import load_config, set_seed, setup_directories
from train_resnet import train_resnet
from train_distilbert import train_distilbert
from evaluate import evaluate_all_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def save_metrics(metrics, model_type, init_method, config):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dict of metrics
        model_type: 'resnet' or 'distilbert'
        init_method: Initialization method name
        config: Configuration dictionary
    """
    if model_type == 'resnet':
        metrics_dir = os.path.join(config['cnn']['results_dir'], 'metrics')
    else:  # distilbert
        metrics_dir = os.path.join(config['transformer']['results_dir'], 'metrics')
    
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{model_type}_{init_method}_metrics.json")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {metrics_path}")

def run_experiment(args, config):
    """
    Run the specified experiment.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
    """
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Create necessary directories
    setup_directories(config)
    
    # Determine which experiments to run
    run_models = []
    if args.model == 'all':
        run_models = ['resnet', 'distilbert']
    else:
        run_models = [args.model]
    
    run_inits = []
    if args.init == 'all':
        run_inits = config['init_methods']
    else:
        run_inits = [args.init]
    
    # Run experiments
    for model in run_models:
        for init_method in run_inits:
            logger.info(f"Running {model} with {init_method} initialization")
            
            if model == 'resnet':
                metrics = train_resnet(init_method, config)
                save_metrics(metrics, 'resnet', init_method, config)
            elif model == 'distilbert':
                metrics = train_distilbert(init_method, config)
                save_metrics(metrics, 'distilbert', init_method, config)
    
    # Evaluate results if all experiments have been run
    if args.evaluate or (args.model == 'all' and args.init == 'all'):
        logger.info("Running evaluation on all results")
        evaluate_all_results(args.config)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run weight initialization experiments')
    parser.add_argument('--model', type=str, choices=['resnet', 'distilbert', 'all'],
                        default='all', help='Model to train')
    parser.add_argument('--init', type=str, 
                        choices=['xavier', 'he', 'lecun', 'random', 'all'],
                        default='all', help='Weight initialization method')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate results without training')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.evaluate:
        evaluate_all_results(args.config)
    else:
        run_experiment(args, config)

if __name__ == "__main__":
    main()
