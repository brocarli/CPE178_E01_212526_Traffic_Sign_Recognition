import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, top_k_accuracy_score
import seaborn as sns


class ModelEvaluator:

    def __init__(self, num_classes=43, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names
        print(f"Model Evaluator initialized")
        print(f"  Number of classes: {num_classes}")

    def get_predictions(self, model, X_test):
        y_pred_proba = model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred, y_pred_proba

    def compute_top_k_accuracy(self, y_true, y_pred_proba, k=5):
        top_k_acc = top_k_accuracy_score(y_true, y_pred_proba, k=k)
        print(f"\nTop-{k} Accuracy: {top_k_acc:.4f} ({top_k_acc*100:.2f}%)")
        return top_k_acc

    def compute_per_class_accuracy(self, y_true, y_pred):
        per_class_acc = {}
        for cls in range(self.num_classes):
            mask = y_true == cls
            if mask.sum() == 0:
                per_class_acc[cls] = None
                continue
            per_class_acc[cls] = accuracy_score(y_true[mask], y_pred[mask])
        return per_class_acc

    def print_classification_report(self, y_true, y_pred):
        target_names = None
        if self.class_names:
            target_names = [self.class_names[i] for i in range(self.num_classes)
                            if i in np.unique(y_true)]

        report = classification_report(y_true, y_pred,
                                       target_names=target_names)
        print("\nClassification Report:")
        print("-" * 60)
        print(report)
        return report

    def plot_confusion_matrix(self, y_true, y_pred,
                              save_path='results/confusion_matrix.png',
                              normalize=False):
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=(self.num_classes <= 20), fmt=fmt,
                    cmap='Blues', cbar=True,
                    xticklabels=self.class_names if self.class_names else 'auto',
                    yticklabels=self.class_names if self.class_names else 'auto')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()

        return cm

    def plot_per_class_accuracy(self, y_true, y_pred,
                                save_path='results/per_class_accuracy.png'):
        per_class_acc = self.compute_per_class_accuracy(y_true, y_pred)

        classes = [c for c, acc in per_class_acc.items() if acc is not None]
        accuracies = [per_class_acc[c] for c in classes]

        plt.figure(figsize=(16, 6))
        colors = ['green' if acc >= 0.9 else 'orange' if acc >= 0.7 else 'red'
                  for acc in accuracies]
        plt.bar(classes, accuracies, color=colors)
        plt.axhline(y=np.mean(accuracies), color='blue', linestyle='--',
                    label=f'Mean: {np.mean(accuracies):.2f}')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Per-class accuracy plot saved to {save_path}")
        plt.close()

        print("\nPer-Class Accuracy Summary:")
        print(f"  Best class:  Class {classes[np.argmax(accuracies)]} "
              f"({max(accuracies)*100:.1f}%)")
        print(f"  Worst class: Class {classes[np.argmin(accuracies)]} "
              f"({min(accuracies)*100:.1f}%)")
        print(f"  Mean accuracy: {np.mean(accuracies)*100:.2f}%")
        print(f"  Classes >= 90% accuracy: "
              f"{sum(1 for a in accuracies if a >= 0.9)}/{len(accuracies)}")

        return per_class_acc

    def run_full_evaluation(self, model, X_test, y_test,
                            results_dir='results'):
        print("\n" + "=" * 60)
        print("FULL MODEL EVALUATION")
        print("=" * 60)

        print("\nGenerating predictions...")
        y_pred, y_pred_proba = self.get_predictions(model, X_test)

        overall_acc = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

        top5_acc = self.compute_top_k_accuracy(y_test, y_pred_proba, k=5)

        self.print_classification_report(y_test, y_pred)

        self.plot_confusion_matrix(
            y_test, y_pred,
            save_path=f'{results_dir}/confusion_matrix.png',
            normalize=False
        )
        self.plot_confusion_matrix(
            y_test, y_pred,
            save_path=f'{results_dir}/confusion_matrix_normalized.png',
            normalize=True
        )

        per_class_acc = self.plot_per_class_accuracy(
            y_test, y_pred,
            save_path=f'{results_dir}/per_class_accuracy.png'
        )

        eval_results = {
            'overall_accuracy': float(overall_acc),
            'top_5_accuracy': float(top5_acc),
            'per_class_accuracy': {str(k): (float(v) if v is not None else None)
                                   for k, v in per_class_acc.items()}
        }

        print("\n" + "=" * 60)
        print("✓ Evaluation complete!")
        print("=" * 60)

        return eval_results


if __name__ == "__main__":

    print("=" * 50)
    print("Traffic Sign Recognition - Model Evaluation (Sprint 2)")
    print("=" * 50)
    print()

    import os
    os.makedirs('results', exist_ok=True)

    print("Creating sample predictions for demonstration...")
    num_samples = 200
    num_classes = 43

    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred_proba = np.random.dirichlet(np.ones(num_classes), size=num_samples)

    evaluator = ModelEvaluator(num_classes=num_classes)

    print("\nRunning evaluation on sample data...")
    y_pred = np.argmax(y_pred_proba, axis=1)

    evaluator.print_classification_report(y_true, y_pred)
    evaluator.compute_top_k_accuracy(y_true, y_pred_proba, k=5)
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.plot_per_class_accuracy(y_true, y_pred)

    print("\n✓ Evaluation demo complete!")
