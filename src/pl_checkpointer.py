import torch
import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback
import os
from pathlib import Path


class TorchModelSaver(Callback):
    """
    Saves the full PyTorch model (not Lightning checkpoint) whenever validation improves.
    Works like ModelCheckpoint but saves torch models directly.
    """
    
    def __init__(
        self,
        dirpath=None,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=False,
    ):
        super().__init__()
        self.dirpath = dirpath or "saved_models"
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_model_score = None
        self.best_model_path = None
        self.saved_models = []  # Track saved models for top_k management
        
        # Create directory if it doesn't exist
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)
        
        # Determine comparison function
        self.mode_dict = {
            "min": lambda x, y: x < y,
            "max": lambda x, y: x > y,
        }
        self.compare = self.mode_dict[mode]
    
    def _format_checkpoint_name(self, metrics, epoch):
        """Format the filename with metrics and epoch."""
        filename = self.filename
        filename = filename.replace("{epoch:02d}", f"{epoch:02d}")
        filename = filename.replace("{epoch}", str(epoch))
        
        # Replace metric placeholders
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Handle different format specifications
                if f"{{{key}:.4f}}" in filename:
                    filename = filename.replace(f"{{{key}:.4f}}", f"{value:.4f}")
                elif f"{{{key}:.2f}}" in filename:
                    filename = filename.replace(f"{{{key}:.2f}}", f"{value:.2f}")
                elif f"{{{key}}}" in filename:
                    filename = filename.replace(f"{{{key}}}", str(value))
        
        return filename + ".pt"
    
    def _save_model(self, trainer, filepath):
        """Save the actual PyTorch model."""
        model = trainer.lightning_module.model if hasattr(trainer.lightning_module, 'model') else trainer.lightning_module
        
        # Save comprehensive checkpoint
        save_dict = {
            'model': model,
            'state_dict': model.state_dict(),
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
        }
        
        # Add hyperparameters if available
        if hasattr(model, 'hparams'):
            save_dict['hparams'] = model.hparams
        if hasattr(model, 'config'):
            save_dict['config'] = model.config
        print(f'saved_model {trainer.current_epoch}')
        print(' ')
        # trainer.save_checkpoint(filepath+'.ckpt')
        torch.save(model, filepath)
        
        if self.verbose:
            print(f"Saved model to {filepath}")
    
    def _should_save(self, current_score):
        """Determine if current model should be saved based on metric."""
        if self.best_model_score is None:
            return True
        
        if self.mode == "min":
            return current_score < self.best_model_score
        else:  # mode == "max"
            return current_score > self.best_model_score
    
    def _remove_old_models(self):
        """Remove old models if we exceed save_top_k."""
        if self.save_top_k > 0 and len(self.saved_models) > self.save_top_k:
            # Sort by score
            self.saved_models.sort(key=lambda x: x[1], reverse=(self.mode == "max"))
            
            # Remove worst models
            while len(self.saved_models) > self.save_top_k:
                path_to_remove, _ = self.saved_models.pop()
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
                    # os.remove(path_to_remove+'.ckpt')
                    if self.verbose:
                        print(f"Removed old model: {path_to_remove}")
    
    def on_validation_end(self, trainer, pl_module):
        """Called when validation ends."""
        # Get current metrics
        metrics = trainer.callback_metrics

        if trainer.sanity_checking:
            return
        
        if self.monitor not in metrics:
            if self.verbose:
                print(f"Warning: {self.monitor} not found in metrics")
            return
        
        current_score = metrics[self.monitor].item()
        print(f'\n\n\ncurrent score: {current_score} - best score {self.best_model_score}\n\n\n')
        # Check if we should save
        if self._should_save(current_score):
            # Format filename
            filename = self._format_checkpoint_name(metrics, trainer.current_epoch)
            filepath = os.path.join(self.dirpath, filename)
            
            # Save model
            self._save_model(trainer, filepath)
            
            # Update best score and path
            self.best_model_score = current_score
            self.best_model_path = filepath
            
            # Track saved models
            self.saved_models.append((filepath, current_score))
            
            # Remove old models if needed
            self._remove_old_models()
    
    def on_train_end(self, trainer, pl_module):
        """Save last model if requested."""
        if self.save_last:
            filepath = os.path.join(self.dirpath, "last.pt")
            self._save_model(trainer, filepath)
            if self.verbose:
                print(f"Saved last model to {filepath}")