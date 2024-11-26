# -*- coding: utf-8 -*-
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
import logging

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device, logger, is_vec=False, patience=5,
                 scheduler_type="constant", model_save_path="model_weights/best_model.pth", **scheduler_params):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.logger = logger
        self.is_vec = is_vec
        self.model.to(device)
        self.model_save_path = model_save_path
        if scheduler_type == "constant":
            self.scheduler = None
        elif scheduler_type == "linear":
            self.scheduler = lr_scheduler.LinearLR(optimizer, **scheduler_params)
        elif scheduler_type == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        else:
            raise ValueError("Unsupported scheduler type. Choose from 'constant', 'linear', 'cosine'.")
   
    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path))

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_labels = []
        all_outputs = []
        for inputs, labels in self.train_loader:
            if self.is_vec:
                inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
            else:
                inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.float().detach())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().detach().numpy())
        if self.scheduler:
            self.scheduler.step()
        avg_loss = running_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_outputs)
        return avg_loss, auc

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                if self.is_vec:
                    inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
                else:
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float().detach())
                running_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().detach().numpy())

        avg_loss = running_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_outputs)
        return avg_loss, auc

    def test(self, test_loader, model_path):
        self.load_model(model_path)
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                if self.is_vec:
                    inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
                else:
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float().detach())
                running_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().detach().numpy())

        avg_loss = running_loss / len(test_loader) 
        auc = roc_auc_score(all_labels, all_outputs)
        print(f'Test Loss: {avg_loss:.4f}, Test AUC: {auc:.4f}')
        self.logger.info(f'Test Loss: {avg_loss:.4f}, Test AUC: {auc:.4f}')
        return avg_loss, auc

    def predict(self, input_data):
        self.model.eval()
        pre_result = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                if self.is_vec:
                    inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
                else:
                    inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions = outputs.cpu().numpy().tolist()
                pre_result.extend(predictions)
        return pre_result

    def train(self, epochs):
        best_val_auc = 0

        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}')
            train_loss, train_auc = self.train_epoch()
            val_loss, val_auc = self.validate_epoch()
            print(
                f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}')
            self.logger.info(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}')
            
            if val_auc > best_val_auc:
                torch.save(self.model.state_dict(), self.model_save_path)
                best_val_auc = val_auc


    def train_early_stop(self, epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        

        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}')
            train_loss, train_auc = self.train_epoch()
            val_loss, val_auc = self.validate_epoch()
            print(
                f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}')
            self.logger.info(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print('Early stopping triggered.')
                self.logger.info('Early stopping triggered.')
                break


