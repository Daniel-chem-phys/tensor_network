import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(chain, current_best_loss,actpos, dir_sweep, filename):
    data = {
        'chain': chain,
        'best_loss': current_best_loss,
        'actpos': actpos,
        'dir_sweep': dir_sweep
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Checkpoint saved! Loss: {current_best_loss:.4f} | Sito: {actpos} | Dir: {dir_sweep}")

def load_checkpoint(filename="best_mps_model.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # If new format (dictionary)
        if isinstance(data, dict):
            print("📦 Checkpoint format detected.")
            return data['chain'], data['best_loss'], data['actpos'], data['dir_sweep']
        
        # If old format formato (list), returns the list and inf loss
        elif isinstance(data, list):
            print("⚠️ Vecchio formato (solo lista) rilevato. Reset best_loss.")
            return data, float('inf')
            

def plot_train_results(losses, errvec, window=50, save_path='training_results.png'):
    """
    Plots the evolution of Loss and Error Norm during training.
    """
    plt.figure(figsize=(12, 5))

    # --- Subplot 1: Cross-entropy Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(losses, color='skyblue', alpha=0.4, label='Instant loss')
    if len(losses) >= window:
        smooth_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), smooth_loss, 'navy', lw=2, label=f'Mobile mean ({window})')
    plt.xlabel('Step')
    plt.ylabel('Cross-entropy Loss')
    plt.title('Loss evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Subplot 2: Error ---
    plt.subplot(1, 2, 2)
    plt.plot(errvec, color='lightcoral', alpha=0.4, label='Instant error')
    if len(errvec) >= window:
        smooth_err = np.convolve(errvec, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(errvec)), smooth_err, 'darkred', lw=2, label=f'Mobile average ({window})')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Critical threshold (1.0)')
    plt.xlabel('Step')
    plt.ylabel('Error Norm')
    plt.title('Error evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
