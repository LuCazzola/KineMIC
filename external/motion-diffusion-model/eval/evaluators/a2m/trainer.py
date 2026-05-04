import torch
import logging
from os.path import join as pjoin
from contextlib import nullcontext
from utils import dist_util
from torch.optim import SGD
from train.train_platforms import WandBPlatform, NoPlatform

class TrainerA2M():

    def __init__(self, model, data, num_epochs=80, train_platform=None, save_dir='.', log_file='training.log'):

        self.model = model
        self.data = data
        self.log_interval = 10
        self.eval_interval = 1
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.train_platform = train_platform
        
        # Setup logging
        self.setup_logging(log_file)
        
        self.steps_per_epoch = -(len(self.data['train'].dataset) // -self.data['train'].batch_size)
        self.total_iterations = self.num_epochs * self.steps_per_epoch

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.optimizer = SGD(
            model.parameters(),
            lr=1e-1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.total_iterations,
            eta_min=0
        )

        transforms = self.data['train'].dataset.m_dataset.transforms
        self.data['train'].dataset.set_transforms([
            transforms.to_unit_length,
            #lambda motion, m_length: self.data['train'].dataset.normalize_motion(motion)
        ])

        transforms = self.data['val'].dataset.m_dataset.transforms
        self.data['val'].dataset.set_transforms([
            transforms.to_unit_length,
            #lambda motion, m_length: self.data['train'].dataset.normalize_motion(motion)
        ])

        self.iteration = 0

    def setup_logging(self, log_file):
        """Setup logging to file and console"""
        # Create logger
        self.logger = logging.getLogger('TrainerSTGCN')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create formatter (just the message, no timestamp/logger name)
        formatter = logging.Formatter('%(message)s')
        
        # File handler, with filemode='w' to clear the file on startup
        file_handler = logging.FileHandler(pjoin(self.save_dir, log_file), mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def loss_compute(self, logits, gt):
        loss = self.criterion(logits, gt)
        loss_dict = {'loss': loss.item()}
        if self.model.training:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
        return loss_dict

    def loop(self, epoch, mode='train'):
        
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        accuracy = []
        with torch.no_grad() if mode == 'val' else nullcontext():
            for step, (motion, cond) in enumerate(self.data[mode]):
                # Prepare motion and conditioning
                motion = motion.to(dist_util.dev())
                cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                gt = cond['y']['action'].squeeze()
                lengths = cond['y']['lengths']
                # Forward pass
                features, logits = self.model(motion, lengths)
                # loss
                loss_dict = self.loss_compute(logits, gt) if mode == 'train' else {}
                accuracy.append(self.model.compute_topk_accuracy(logits, gt, k=1))
                # Logging
                if mode == 'train':
                    if step % self.log_interval == 0:
                        train_acc = sum(accuracy)/len(accuracy)
                        current_lr = self.optimizer.param_groups[0]['lr']
                        accuracy = []
                        self.logger.info(f"[{mode}] (epoch={epoch}, it={self.iteration}) - train_acc: {train_acc:.4f} - lr: {current_lr:.6f} - {loss_dict}")
                        if self.train_platform :
                            self.train_platform.report_scalar(name='train_acc', value=train_acc, iteration=self.iteration, group_name='Metrics')
                            self.train_platform.report_scalar(name='learning_rate', value=current_lr, iteration=self.iteration, group_name='Training')
                            for k, v in loss_dict.items():
                                self.train_platform.report_scalar(name=f'train_{k}', value=v, iteration=self.iteration, group_name='Loss')
                    # increment every training step
                    self.iteration += 1

        if mode == 'val':
            self.logger.info("======== Validation ========")
            val_acc = sum(accuracy)/len(accuracy)
            self.logger.info(f"[{mode}] (epoch={epoch}) - acc: {val_acc:.4f}")
            if self.train_platform:
                self.train_platform.report_scalar(name='val_acc', value=val_acc, iteration=self.iteration, group_name='Metrics')
            self.logger.info("============================")
            return val_acc
        

    def train(self):
        best_val_acc, best_val_epoch = 0.0, 0
        for epoch in range(self.num_epochs):
            self.logger.info(f"========= Epoch {epoch}/{self.num_epochs} =========")
            self.loop(epoch, mode='train')
            if epoch % self.eval_interval == 0 or epoch == self.num_epochs - 1:
                val_acc = self.loop(epoch, mode='val')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    torch.save(self.model.state_dict(), pjoin(self.save_dir, 'model_best.pth'))
                    self.logger.info(f"New best model saved with val_acc: {best_val_acc:.4f}")
        self.logger.info(f"Training completed. Best validation accuracy [{best_val_acc:.4f}] at epoch [{best_val_epoch}].")

    def evaluate(self):
        self.model.eval()
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for _, (motion, cond) in enumerate(self.data['val']):
                # Prepare motion and conditioning
                motion = motion.to(dist_util.dev())
                cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val 
                            for key, val in cond['y'].items()}
                gt = cond['y']['action'].squeeze()
                lengths = cond['y']['lengths']
                
                # Forward pass
                _, logits = self.model(motion, lengths)
                
                # Store logits and targets for accuracy computation
                all_logits.append(logits)
                all_targets.append(gt)
        
        # Concatenate all predictions and targets
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute global top-k accuracies
        top1_acc = self.compute_topk_accuracy(all_logits, all_targets, k=1)
        top2_acc = self.compute_topk_accuracy(all_logits, all_targets, k=2)
        top3_acc = self.compute_topk_accuracy(all_logits, all_targets, k=3)
        
        # Compute class-wise top-1 accuracy
        classwise_acc = self.compute_classwise_accuracy(all_logits, all_targets)
        
        # Print results
        print(f"Global Top-1 Accuracy: {top1_acc:.4f}")
        print(f"Global Top-2 Accuracy: {top2_acc:.4f}")
        print(f"Global Top-3 Accuracy: {top3_acc:.4f}")
        print(f"Average Class-wise Top-1 Accuracy: {classwise_acc['mean']:.4f}")
        
        # Print per-class accuracies
        print("\nPer-class Top-1 Accuracies:")
        for class_idx, acc in classwise_acc['per_class'].items():
            print(f"Class {self.data['train'].dataset.m_dataset.get_compact_class_id(class_idx, reverse=True)}: {acc:.4f}")
        
        return {
            'top1_accuracy': top1_acc,
            'top2_accuracy': top2_acc,
            'top3_accuracy': top3_acc,
            'classwise_accuracy': classwise_acc
        }

    def compute_topk_accuracy(self, logits, targets, k):
        """Compute top-k accuracy from pre-softmax logits"""
        with torch.no_grad():
            # Get top-k predictions from logits (pre-softmax values)
            _, pred_topk = torch.topk(logits, k, dim=1, largest=True, sorted=True)
            # Check if true labels are in top-k predictions
            targets_expanded = targets.view(-1, 1).expand(-1, k)
            correct = pred_topk.eq(targets_expanded).sum(dim=1)
            # Calculate accuracy
            accuracy = (correct > 0).float().mean().item()
            return accuracy

    def compute_classwise_accuracy(self, logits, targets):
        """Compute per-class top-1 accuracy from pre-softmax logits"""
        with torch.no_grad():
            # Get predictions from logits (highest logit value)
            _, pred = torch.max(logits, dim=1)
            
            # Get unique classes
            unique_classes = torch.unique(targets)
            
            per_class_acc = {}
            class_accuracies = []
            
            for class_idx in unique_classes:
                # Get indices for this class
                class_mask = targets == class_idx
                class_preds = pred[class_mask]
                class_targets = targets[class_mask]
                
                # Compute accuracy for this class
                if len(class_targets) > 0:
                    class_acc = (class_preds == class_targets).float().mean().item()
                    per_class_acc[class_idx.item()] = class_acc
                    class_accuracies.append(class_acc)
                else:
                    per_class_acc[class_idx.item()] = 0.0
                    class_accuracies.append(0.0)
            
            # Compute mean class-wise accuracy
            mean_classwise_acc = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0
            
            return {
                'mean': mean_classwise_acc,
                'per_class': per_class_acc
            }