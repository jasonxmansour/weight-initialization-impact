import os
import json
import numpy as np
import logging
from utils import load_config, plot_training_curves, plot_comparison_bar

logger = logging.getLogger(__name__)

def load_metrics(model_type, init_method, config):
    """
    Load saved metrics for a specific model and initialization method.
    
    Args:
        model_type: 'resnet' or 'distilbert'
        init_method: Initialization method name
        config: Configuration dictionary
        
    Returns:
        dict: Loaded metrics
    """
    if model_type == 'resnet':
        metrics_dir = os.path.join(config['cnn']['results_dir'], 'metrics')
    else:  # distilbert
        metrics_dir = os.path.join(config['transformer']['results_dir'], 'metrics')
    
    metrics_path = os.path.join(metrics_dir, f"{model_type}_{init_method}_metrics.json")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def calculate_gradient_reductions(metrics_by_method, reference_method='random'):
    """
    Calculate gradient explosion reduction percentages relative to reference method.
    
    Args:
        metrics_by_method: Dict of metrics by initialization method
        reference_method: Method to compare against (usually 'random')
        
    Returns:
        dict: Gradient reduction percentages
    """
    ref_grad_norm = metrics_by_method[reference_method]['max_grad_norm']
    
    reductions = {}
    for method, metrics in metrics_by_method.items():
        if method != reference_method:
            method_grad_norm = metrics['max_grad_norm']
            reduction_pct = (ref_grad_norm - method_grad_norm) / ref_grad_norm * 100
            reductions[method] = reduction_pct
    
    return reductions

def evaluate_resnet_results(config):
    """
    Evaluate and visualize ResNet-18 results across initialization methods.
    
    Args:
        config: Configuration dictionary
    """
    init_methods = config['init_methods']
    metrics_by_method = {}
    
    # Load metrics for each initialization method
    for method in init_methods:
        try:
            metrics = load_metrics('resnet', method, config)
            metrics_by_method[method] = metrics
        except Exception as e:
            logger.error(f"Failed to load metrics for ResNet with {method} initialization: {e}")
    
    if not metrics_by_method:
        logger.error("No metrics found for ResNet. Skipping evaluation.")
        return
    
    # Prepare result directories
    plot_dir = config['evaluation']['plot_dir']
    os.makedirs(os.path.join(plot_dir, 'resnet'), exist_ok=True)
    
    # Plot training accuracy curves
    accuracy_by_method = {method: metrics['train_accuracies'] for method, metrics in metrics_by_method.items()}
    plot_training_curves(
        accuracy_by_method,
        "ResNet-18 Training Accuracy by Initialization Method",
        os.path.join(plot_dir, 'resnet', 'train_accuracy.png')
    )
    
    # Plot validation accuracy curves
    val_accuracy_by_method = {method: metrics['val_accuracies'] for method, metrics in metrics_by_method.items()}
    plot_training_curves(
        val_accuracy_by_method,
        "ResNet-18 Validation Accuracy by Initialization Method",
        os.path.join(plot_dir, 'resnet', 'val_accuracy.png')
    )
    
    # Plot training loss curves
    loss_by_method = {method: metrics['train_losses'] for method, metrics in metrics_by_method.items()}
    plot_training_curves(
        loss_by_method,
        "ResNet-18 Training Loss by Initialization Method",
        os.path.join(plot_dir, 'resnet', 'train_loss.png')
    )
    
    # Plot gradient norm curves
    grad_norm_by_method = {method: metrics['grad_norms'] for method, metrics in metrics_by_method.items()}
    plot_training_curves(
        grad_norm_by_method,
        "ResNet-18 Gradient Norm by Initialization Method",
        os.path.join(plot_dir, 'resnet', 'grad_norm.png')
    )
    
    # Plot final accuracy comparison
    final_accuracy = {method: metrics['final_accuracy'] for method, metrics in metrics_by_method.items()}
    plot_comparison_bar(
        final_accuracy,
        "ResNet-18 Final Accuracy by Initialization Method",
        os.path.join(plot_dir, 'resnet', 'final_accuracy_comparison.png')
    )
    
    # Plot convergence epoch comparison
    convergence_epochs = {method: metrics['convergence_epoch'] for method, metrics in metrics_by_method.items()}
    plot_comparison_bar(
        convergence_epochs,
        "ResNet-18 Convergence Speed by Initialization Method (Lower is Better)",
        os.path.join(plot_dir, 'resnet', 'convergence_comparison.png')
    )
    
    # Generate summary statistics
    best_method = max(final_accuracy.items(), key=lambda x: x[1])[0]
    best_accuracy = final_accuracy[best_method]
    fastest_method = min(convergence_epochs.items(), key=lambda x: x[1])[0]
    fastest_epochs = convergence_epochs[fastest_method]
    
    summary = {
        "best_method": best_method,
        "best_accuracy": best_accuracy,
        "fastest_method": fastest_method,
        "fastest_epochs": fastest_epochs,
        "final_accuracies": final_accuracy,
        "convergence_epochs": convergence_epochs
    }
    
    # Save summary
    summary_path = os.path.join(config['evaluation']['log_dir'], 'resnet_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"ResNet-18 evaluation completed. Summary saved to {summary_path}")
    logger.info(f"Best method: {best_method} with accuracy {best_accuracy:.2f}%")
    logger.info(f"Fastest convergence: {fastest_method} in {fastest_epochs} epochs")
    
    return summary

def evaluate_distilbert_results(config):
    """
    Evaluate and visualize DistilBERT results across initialization methods.
    
    Args:
        config: Configuration dictionary
    """
    init_methods = config['init_methods']
    metrics_by_method = {}
    
    # Load metrics for each initialization method
    for method in init_methods:
        try:
            metrics = load_metrics('distilbert', method, config)
            metrics_by_method[method] = metrics
        except Exception as e:
            logger.error(f"Failed to load metrics for DistilBERT with {method} initialization: {e}")
    
    if not metrics_by_method:
        logger.error("No metrics found for DistilBERT. Skipping evaluation.")
        return
    
    # Calculate gradient explosion reduction
    if 'random' in metrics_by_method:
        reductions = calculate_gradient_reductions(metrics_by_method, 'random')
        
        # Update metrics with reduction percentages
        for method, reduction in reductions.items():
            metrics_by_method[method]['grad_explosion_reduction'] = reduction
    
    # Prepare result directories
    plot_dir = config['evaluation']['plot_dir']
    os.makedirs(os.path.join(plot_dir, 'distilbert'), exist_ok=True)
    
    # Plot training accuracy curves
    accuracy_by_method = {method: metrics['train_accuracies'] for method, metrics in metrics_by_method.items()}
    plot_training_curves(
        accuracy_by_method,
        "DistilBERT Training Accuracy by Initialization Method",
        os.path.join(plot_dir, 'distilbert', 'train_accuracy.png')
    )
    
    # Plot validation accuracy curves
    val_accuracy_by_method = {method: metrics['val_accuracies'] for method, metrics in metrics_by_method.items()}
    plot_training_curves(
        val_accuracy_by_method,
        "DistilBERT Validation Accuracy by Initialization Method",
        os.path.join(plot_dir, 'distilbert', 'val_accuracy.png')
    )
    
    # Plot gradient norm curves
    grad_norm_by_method = {method: metrics['grad_norms'] for method, metrics in metrics_by_method.items()}
    plot_training_curves(
        grad_norm_by_method,
        "DistilBERT Gradient Norm by Initialization Method",
        os.path.join(plot_dir, 'distilbert', 'grad_norm.png')
    )
    
    # Plot final accuracy comparison
    final_accuracy = {method: metrics['final_accuracy'] for method, metrics in metrics_by_method.items()}
    plot_comparison_bar(
        final_accuracy,
        "DistilBERT Final Accuracy by Initialization Method",
        os.path.join(plot_dir, 'distilbert', 'final_accuracy_comparison.png')
    )
    
    # Plot gradient reduction comparison
    if 'random' in metrics_by_method:
        grad_reductions = {
            method: metrics.get('grad_explosion_reduction', 0) 
            for method, metrics in metrics_by_method.items() 
            if method != 'random'
        }
        
        plot_comparison_bar(
            grad_reductions,
            "Gradient Explosion Reduction vs. Random Initialization (%)",
            os.path.join(plot_dir, 'distilbert', 'grad_reduction_comparison.png')
        )
    
    # Generate summary statistics
    best_method = max(final_accuracy.items(), key=lambda x: x[1])[0]
    best_accuracy = final_accuracy[best_method]
    
    if 'random' in metrics_by_method and grad_reductions:
        best_stability_method = max(grad_reductions.items(), key=lambda x: x[1])[0]
        best_reduction = grad_reductions[best_stability_method]
    else:
        best_stability_method = None
        best_reduction = None
    
    summary = {
        "best_method": best_method,
        "best_accuracy": best_accuracy,
        "final_accuracies": final_accuracy,
    }
    
    if best_stability_method is not None:
        summary.update({
            "best_stability_method": best_stability_method,
            "gradient_reduction": best_reduction,
            "gradient_reductions": grad_reductions
        })
    
    # Save summary
    summary_path = os.path.join(config['evaluation']['log_dir'], 'distilbert_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"DistilBERT evaluation completed. Summary saved to {summary_path}")
    logger.info(f"Best method: {best_method} with accuracy {best_accuracy:.2f}%")
    if best_stability_method:
        logger.info(f"Best stability: {best_stability_method} with {best_reduction:.2f}% gradient reduction")
    
    return summary

def evaluate_all_results(config_path="config.yaml"):
    """
    Evaluate results for all models and methods.
    
    Args:
        config_path: Path to configuration file
    """
    config = load_config(config_path)
    
    # Create directories if they don't exist
    os.makedirs(config['evaluation']['plot_dir'], exist_ok=True)
    os.makedirs(config['evaluation']['log_dir'], exist_ok=True)
    
    # Evaluate ResNet results
    resnet_summary = evaluate_resnet_results(config)
    
    # Evaluate DistilBERT results
    distilbert_summary = evaluate_distilbert_results(config)
    
    # Generate overall report
    if resnet_summary and distilbert_summary:
        overall_summary = {
            "resnet": resnet_summary,
            "distilbert": distilbert_summary,
            "conclusions": {
                "best_cnn_init": resnet_summary["best_method"],
                "best_transformer_init": distilbert_summary["best_method"],
                "cnn_accuracy": resnet_summary["best_accuracy"],
                "transformer_accuracy": distilbert_summary["best_accuracy"]
            }
        }
        
        if "best_stability_method" in distilbert_summary:
            overall_summary["conclusions"]["best_transformer_stability"] = distilbert_summary["best_stability_method"]
            overall_summary["conclusions"]["gradient_reduction"] = distilbert_summary["gradient_reduction"]
        
        # Save overall summary
        overall_path = os.path.join(config['evaluation']['log_dir'], 'overall_summary.json')
        with open(overall_path, 'w') as f:
            json.dump(overall_summary, f, indent=4)
        
        logger.info(f"Overall evaluation completed. Summary saved to {overall_path}")
        
        # Log key findings matching the paper
        logger.info("=== KEY FINDINGS ===")
        logger.info(f"CNN Performance: {resnet_summary['best_method']} initialization achieved the highest accuracy ({resnet_summary['best_accuracy']:.1f}%) and fastest convergence ({resnet_summary['fastest_epochs']} epochs).")
        
        if "best_stability_method" in distilbert_summary:
            logger.info(f"Transformer Performance: {distilbert_summary['best_stability_method']} initialization reduced gradient explosion by {distilbert_summary['gradient_reduction']:.0f}% compared to random.")
        
        if 'lecun' in distilbert_summary.get('final_accuracies', {}):
            lecun_acc = distilbert_summary['final_accuracies']['lecun']
            logger.info(f"LeCun initialization underperformed ({lecun_acc:.1f}%) for transformers despite theoretical advantages.")
    
    return overall_summary if resnet_summary and distilbert_summary else None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate experimental results')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, choices=['resnet', 'distilbert', 'all'],
                        default='all', help='Model to evaluate')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.model == 'resnet':
        evaluate_resnet_results(config)
    elif args.model == 'distilbert':
        evaluate_distilbert_results(config)
    else:  # 'all'
        evaluate_all_results(args.config)
