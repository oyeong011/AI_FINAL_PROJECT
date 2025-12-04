"""
Improved BERT Fine-tuning Module for Text Classification
========================================================

This module provides a clean, modular implementation of BERT fine-tuning
with best practices including type hints, proper error handling, and
configuration management.

Author: AI Assistant
Date: 2025-12-04
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BertConfig:
    """Configuration class for BERT fine-tuning hyperparameters."""

    # Model configuration
    model_name: str = 'bert-base-uncased'
    num_classes: int = 2
    bert_hidden_size: int = 768
    hidden_size: int = 256
    dropout_rate: float = 0.3
    dropout_rate_2: float = 0.2

    # Training configuration
    max_seq_length: int = 256
    batch_size: int = 16
    num_epochs: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Data configuration
    data_path: str = './data/IMDB Dataset Train.csv'
    test_size: float = 0.2
    random_state: int = 42

    # Early stopping configuration
    early_stopping_patience: int = 3
    early_stopping_delta: float = 0.001

    # Model saving
    model_save_path: str = './finetuned_bert.pth'

    # Device configuration
    device: Optional[str] = None

    def __post_init__(self):
        """Post initialization to set device."""
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

    def display(self) -> None:
        """Display configuration parameters."""
        logger.info("Configuration:")
        for key, value in self.__dict__.items():
            logger.info(f"  {key}: {value}")


class ImdbDataset(Dataset):
    """PyTorch Dataset for IMDB text classification.

    Args:
        texts: List of text samples
        labels: List of labels
        tokenizer: BERT tokenizer
        max_seq_length: Maximum sequence length for padding/truncation
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_seq_length: int
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing input_ids, attention_mask, and label tensors
        """
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(nn.Module):
    """BERT-based text classifier with custom head.

    Args:
        config: Configuration object containing model parameters
    """

    def __init__(self, config: BertConfig):
        super(BertClassifier, self).__init__()
        self.config = config

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(config.model_name)

        # Custom classification head
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(config.bert_hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(config.dropout_rate_2)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

        logger.info(f"Initialized BertClassifier with {config.model_name}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_length)
            attention_mask: Attention mask tensor of shape (batch_size, seq_length)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Get BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Classification head
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)

        return logits

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """

    def __init__(
        self,
        patience: int = 3,
        delta: float = 0.001,
        mode: str = 'max'
    ):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if model improved, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered")
            return False


def download_imdb_dataset(data_dir: str = './data') -> str:
    """Download IMDB dataset from GitHub.

    Args:
        data_dir: Directory to save the dataset

    Returns:
        Path to the downloaded dataset

    Raises:
        RuntimeError: If download fails
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    file_path = data_path / 'IMDB Dataset Train.csv'

    if file_path.exists():
        logger.info(f"Dataset already exists at {file_path}")
        return str(file_path)

    logger.info("Downloading IMDB dataset...")
    url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"

    import urllib.request
    try:
        urllib.request.urlretrieve(url, file_path)
        logger.info(f"Dataset downloaded successfully to {file_path}")
        return str(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")


def load_imdb_data(file_path: str) -> Tuple[List[str], List[int]]:
    """Load IMDB dataset from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (texts, labels)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    try:
        df = pd.read_csv(file_path)

        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must contain 'review' and 'sentiment' columns")

        texts = df['review'].tolist()
        labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]

        logger.info(f"Loaded {len(texts)} samples from {file_path}")
        return texts, labels

    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    max_grad_norm: float = 1.0
) -> float:
    """Train model for one epoch.

    Args:
        model: PyTorch model
        data_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # Update weights
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str
) -> Tuple[float, str, np.ndarray]:
    """Evaluate model on validation/test data.

    Args:
        model: PyTorch model
        data_loader: Evaluation data loader
        device: Device to evaluate on

    Returns:
        Tuple of (accuracy, classification_report, confusion_matrix)
    """
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(actual_labels, predictions)
    report = classification_report(actual_labels, predictions)
    conf_matrix = confusion_matrix(actual_labels, predictions)

    return accuracy, report, conf_matrix


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: BertConfig,
    early_stopping: Optional[EarlyStopping] = None
) -> Dict[str, List[float]]:
    """Complete training loop with validation and early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration object
        early_stopping: Early stopping object (optional)

    Returns:
        Dictionary containing training history
    """
    history = {
        'train_loss': [],
        'val_accuracy': []
    }

    best_accuracy = 0.0

    for epoch in range(config.num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config.device, config.max_grad_norm
        )
        logger.info(f"Average Training Loss: {train_loss:.4f}")

        # Evaluate
        accuracy, report, conf_matrix = evaluate(model, val_loader, config.device)
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info(f"\n{report}")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(accuracy)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), config.model_save_path)
            logger.info(f"Model saved with accuracy: {accuracy:.4f}")

        # Early stopping
        if early_stopping is not None:
            early_stopping(accuracy)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered, ending training")
                break

    logger.info(f"\n{'='*50}")
    logger.info(f"Best Validation Accuracy: {best_accuracy:.4f}")
    logger.info(f"{'='*50}")

    return history


def main():
    """Main execution function."""
    # Initialize configuration
    config = BertConfig()
    config.display()

    # Download and load data
    try:
        data_path = download_imdb_dataset()
        texts, labels = load_imdb_data(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=config.test_size,
        random_state=config.random_state
    )

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # Create datasets
    train_dataset = ImdbDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    val_dataset = ImdbDataset(val_texts, val_labels, tokenizer, config.max_seq_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Initialize model
    model = BertClassifier(config).to(config.device)
    logger.info(f"Total trainable parameters: {model.get_num_parameters():,}")

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(config.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        delta=config.early_stopping_delta,
        mode='max'
    )

    # Train model
    try:
        history = train(
            model, train_loader, val_loader,
            optimizer, scheduler, config, early_stopping
        )
        logger.info("Training completed successfully!")
        return history
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
