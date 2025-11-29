"""
Script to train all ML models for Parkinson's disease detection
Run this script to train models before using the web application
"""
from ml_models import train_all_models

if __name__ == '__main__':
    print("Starting model training...")
    print("=" * 50)
    
    metrics = train_all_models()
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("\nModel Performance Summary:")
    print("-" * 50)
    
    for model_name, metric in metrics.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:    {metric['accuracy']:.4f} ({metric['accuracy']*100:.2f}%)")
        print(f"  Sensitivity: {metric['sensitivity']:.4f} ({metric['sensitivity']*100:.2f}%)")
        print(f"  Specificity: {metric['specificity']:.4f} ({metric['specificity']*100:.2f}%)")
        print(f"  Precision:   {metric['precision']:.4f} ({metric['precision']*100:.2f}%)")
        print(f"  F1-Score:    {metric['f1_score']:.4f} ({metric['f1_score']*100:.2f}%)")
    
    print("\n" + "=" * 50)
    print("Models saved in 'models/' directory")
    print("Metrics saved in 'model_metrics.json'")

