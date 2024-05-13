# pip install transformers[torch] #RUN PIP command first
import os
import pandas as pd
import random
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaForSequenceClassification
torch.manual_seed(36)

from google.colab import drive

drive.mount('/content/drive')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_data(directory):
    """
    Load data from parquet files in the specified directory.

    Args:
    - directory (str): The directory path where the parquet files are located.

    Returns:
    - pd.DataFrame: Concatenated DataFrame containing data from all parquet files.
    """
    data_files = os.listdir(directory)
    dataframes = []

    for file in data_files:
        if file.endswith('.parquet'):
            file_path = os.path.join(directory, file)
            print(f"Loading file {file_path}")
            dataframes.append(pd.read_parquet(file_path, columns=['Text', 'NACE']))

    df = pd.concat(dataframes)
    return df

def downsample_data(df, time_median):
    """
    Downsample the data based on the median number of samples per NACE category.

    Args:
    - df (pd.DataFrame): Input DataFrame containing the data.
    - time_median (int): Median time value used for downsampling.

    Returns:
    - pd.DataFrame: Downsampled DataFrame.
    """
    resampled_dfs = []

    median_samples = time_median * int(df['NACE'].value_counts().median())

    def downsample_group(group):
        if len(group) > median_samples:
            return resample(group, replace=False, n_samples=median_samples, random_state=36)
        return group

    # Apply the downsample function to each group
    resampled_df = df.groupby('NACE').apply(downsample_group).reset_index(drop=True)

    print(resampled_df['NACE'].value_counts())
    print(resampled_df)

    return resampled_df

"""#Data loading"""

def split_data(df):
    """
    Split the dataset into training, validation, and testing sets.

    Args:
    - df (pd.DataFrame): Input DataFrame containing the dataset.

    Returns:
    - tuple: Four data arrays: train_texts, test_texts, val_texts, train_labels, test_labels, val_labels.
    """
    x = df['Text']
    y = df['NACE']

    train_texts, test_texts, train_labels, test_labels = train_test_split(x, y, test_size=0.3, random_state=36)
    test_texts, val_texts, test_labels, val_labels = train_test_split(test_texts, test_labels, test_size=0.5, random_state=36)

    # If the size of the training set is one more than a multiple of the batch size,
    # the last row will be dropped to prevent an error during training the last batch.
    if len(train_texts) % 8 == 1:
        train_texts = train_texts[:-1]
        train_labels = train_labels[:-1]
        print("Dropped one row because of rest after division")

    return x, y, train_texts, test_texts, val_texts, train_labels, test_labels, val_labels

def calculate_num_labels(df):
    """
    Calculate the number of unique labels in the 'NACE' column of the DataFrame.

    Args:
    - df (pd.DataFrame): Input DataFrame containing the dataset.

    Returns:
    - int: Number of unique labels.
    """
    num_labels = len(df['NACE'].unique())
    return num_labels

def tokenize_texts(texts, tokenizer, truncation=True, padding=True):
    """
    Tokenize a list of texts using the provided tokenizer.

    Args:
    - texts (list of str): List of texts to tokenize.
    - tokenizer: Tokenizer object from Hugging Face's transformers library.
    - truncation (bool): Whether to truncate the texts to the maximum length.
    - padding (bool): Whether to pad the sequences to the maximum length.

    Returns:
    - dict: Tokenized encodings.
    """
    encodings = tokenizer(list(texts), truncation=truncation, padding=padding)
    return encodings

def encode_labels(labels, label2id):
    """
    Encode a list of labels using a label-to-id mapping.

    Args:
    - labels (list of str): List of labels to encode.
    - label2id (dict): Mapping from labels to ids.

    Returns:
    - list of int: Encoded labels.
    """
    labels_ids = [label2id[label] for label in labels]
    return labels_ids

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_dataset(encodings, labels):
    """
    Create a PyTorch Dataset object.

    Args:
    - encodings (dict): Encodings of the dataset.
    - labels (list): Labels of the dataset.

    Returns:
    - Dataset: PyTorch Dataset object.
    """
    dataset = Dataset(encodings, labels)
    return dataset

"""# New model training and evaluation"""

class CustomRobertaConfig(AutoConfig):
    """
    Custom configuration class for a RoBERTa-based model with additional features.

    Args:
    - out_features (int): Number of output features for the custom classification head.
    - **kwargs: Additional keyword arguments for the base configuration.

    Attributes:
    - out_features (int): Number of output features for the custom classification head.

    Methods:
    - to_dict(): Serialize the configuration instance to a Python dictionary.
    - from_dict(config_dict): Instantiate a configuration from a Python dictionary.
    """
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def to_dict(self):
        """
        Serialize this instance to a Python dictionary.
        This implementation adds 'out_features' to the dictionary.
        """
        config_dict = super().to_dict()
        config_dict['out_features'] = self.out_features
        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiate a configuration from a Python dictionary of parameters.
        This implementation loads 'out_features' from the dictionary.
        """
        out_features = config_dict.pop('out_features', None)
        return super().from_dict(config_dict, out_features=out_features)

class CustomRobertaClassificationHead(torch.nn.Module):
    """
    Custom classification head for a RoBERTa-based model with additional layers.

    Args:
    - config: Instance of CustomRobertaConfig containing model configuration.

    Attributes:
    - dense (torch.nn.Linear): Dense layer for feature transformation.
    - dropout (torch.nn.Dropout): Dropout layer for regularization.
    - batchnorm1 (torch.nn.BatchNorm1d): Batch normalization layer.
    - out_proj (torch.nn.Linear): Output layer for classification.

    Methods:
    - forward(features, **kwargs): Forward pass through the classification head.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.out_features)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.batchnorm1 = torch.nn.BatchNorm1d(config.out_features)
        self.out_proj = torch.nn.Linear(config.out_features, config.num_labels)

    def forward(self, features, **kwargs):
        """
        Forward pass through the classification head.

        Args:
        - features (torch.Tensor): Input features from the RoBERTa model.
        - **kwargs: Additional keyword arguments.

        Returns:
        - torch.Tensor: Output logits for sequence classification.
        """
        x = features[:, 0, :]  # Extract the first token's embeddings (CLS token)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.batchnorm1(x)
        x = nn.GELU()(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    Custom RoBERTa model for sequence classification with a custom classification head.

    Args:
    - config: Instance of CustomRobertaConfig containing model configuration.

    Attributes:
    - classifier (CustomRobertaClassificationHead): Custom classification head.

    Methods:
    - forward(**kwargs): Forward pass through the model.
    """
    def __init__(self, config):
        super().__init__(config)
        self.classifier = CustomRobertaClassificationHead(config)

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for a classification task.

    Args:
    - eval_pred (tuple): Tuple containing predictions and labels.
        - predictions (numpy.ndarray): Predicted probabilities or logits.
        - labels (numpy.ndarray): True labels (one-hot encoded or class indices).

    Returns:
    - dict: Dictionary containing computed metrics.
        - "accuracy" (float): Accuracy of the predictions.
        - "f1" (float): Weighted F1-score of the predictions.
    """
    predictions, labels = eval_pred

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions.argmax(axis=1))

    # Calculate F1-score
    f1 = f1_score(labels, predictions.argmax(axis=1), average='weighted')

    # You can calculate other metrics like precision, recall, F1-score, etc.

    return {"accuracy": accuracy, "f1": f1}

def create_trainer(model, train_dataset, val_dataset, output_dir, logging_dir, num_train_epochs, freeze_bottom_layers=False):
    """
    Train the provided model using the specified datasets and configuration.

    Args:
    - model: The model to train.
    - train_dataset: Dataset object for training data.
    - val_dataset: Dataset object for validation data.
    - output_dir (str): Directory to save the model checkpoints.
    - logging_dir (str): Directory for storing training logs.
    - num_train_epochs (int): Number of training epochs.
    - freeze_bottom_layers (bool): Whether to freeze the parameters of the bottom layers.

    Returns:
    - Trainer: Trainer object for training the model.
    """
    # Ensure directories exist
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' not found.")
    if not os.path.isdir(logging_dir):
        raise FileNotFoundError(f"Logging directory '{logging_dir}' not found.")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        save_strategy="epoch"
    )

    # Freeze bottom layers if specified
    if freeze_bottom_layers:
        for name, param in model.named_parameters():
            if name.startswith('roberta.encoder.layer') and int(name.split('.')[3]) < 5:
                param.requires_grad = False

    # Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    return trainer

"""#Evaluation"""

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for a classification task.

    Args:
    - eval_pred (tuple): Tuple containing predictions and labels.
        - predictions (numpy.ndarray): Predicted probabilities or logits.
        - labels (numpy.ndarray): True labels (one-hot encoded or class indices).

    Returns:
    - dict: Dictionary containing computed metrics.
        - "accuracy" (float): Accuracy of the predictions.
        - "f1" (float): Weighted F1-score of the predictions.
    """
    predictions, labels = eval_pred

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions.argmax(axis=1))

    # Calculate F1-score
    f1 = f1_score(labels, predictions.argmax(axis=1), average='weighted')

    # You can calculate other metrics like precision, recall, F1-score, etc.

    return {"accuracy": accuracy, "f1": f1}

import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaForSequenceClassification

class CustomRobertaConfig(AutoConfig):
    """
    Custom configuration class for a RoBERTa-based model with additional features.

    Args:
    - out_features (int): Number of output features for the custom classification head.
    - **kwargs: Additional keyword arguments for the base configuration.

    Attributes:
    - out_features (int): Number of output features for the custom classification head.

    Methods:
    - to_dict(): Serialize the configuration instance to a Python dictionary.
    - from_dict(config_dict): Instantiate a configuration from a Python dictionary.
    """
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def to_dict(self):
        """
        Serialize this instance to a Python dictionary.
        This implementation adds 'out_features' to the dictionary.
        """
        config_dict = super().to_dict()
        config_dict['out_features'] = self.out_features
        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiate a configuration from a Python dictionary of parameters.
        This implementation loads 'out_features' from the dictionary.
        """
        out_features = config_dict.pop('out_features', None)
        return super().from_dict(config_dict, out_features=out_features)

class CustomRobertaClassificationHead(torch.nn.Module):
    """
    Custom classification head for a RoBERTa-based model with additional layers.

    Args:
    - config: Instance of CustomRobertaConfig containing model configuration.

    Attributes:
    - dense (torch.nn.Linear): Dense layer for feature transformation.
    - dropout (torch.nn.Dropout): Dropout layer for regularization.
    - batchnorm1 (torch.nn.BatchNorm1d): Batch normalization layer.
    - out_proj (torch.nn.Linear): Output layer for classification.

    Methods:
    - forward(features, **kwargs): Forward pass through the classification head.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.out_features)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.batchnorm1 = torch.nn.BatchNorm1d(config.out_features)
        self.out_proj = torch.nn.Linear(config.out_features, config.num_labels)

    def forward(self, features, **kwargs):
        """
        Forward pass through the classification head.

        Args:
        - features (torch.Tensor): Input features from the RoBERTa model.
        - **kwargs: Additional keyword arguments.

        Returns:
        - torch.Tensor: Output logits for sequence classification.
        """
        x = features[:, 0, :]  # Extract the first token's embeddings (CLS token)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.batchnorm1(x)
        x = nn.GELU()(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    Custom RoBERTa model for sequence classification with a custom classification head.

    Args:
    - config: Instance of CustomRobertaConfig containing model configuration.

    Attributes:
    - classifier (CustomRobertaClassificationHead): Custom classification head.

    Methods:
    - forward(**kwargs): Forward pass through the model.
    """
    def __init__(self, config):
        super().__init__(config)
        self.classifier = CustomRobertaClassificationHead(config)

def trainer_for_evaluation(model_dir, output_dir, val_dataset, compute_metrics_func, freeze_bottom_layers=False):
    """
    Reload a trained RoBERTa model from a checkpoint and prepare it for evaluation.

    Args:
    - model_dir (str): Directory path for the existing model.
    - val_dataset: Dataset object for validation data.
    - compute_metrics_func (function): Function for computing evaluation metrics.
    - freeze_bottom_layers (bool): Whether to freeze the parameters of the bottom layers.

    Returns:
    - Trainer: Trainer object for evaluation.
    """
    # Reload the trained model
    user_input = input("MAKE SURE YOU CHECK THE DIRECTORIES BEFORE RUNNING! Continue (yes/no)?")
    if user_input != 'yes':
        raise ValueError('User did not respond with \'yes\'.')

    # Load the configuration
    config = CustomRobertaConfig.from_pretrained(model_dir)
    # Load the model
    model = CustomRobertaForSequenceClassification.from_pretrained(model_dir, config=config)
    print(model)

    # Freeze bottom layers if specified
    if freeze_bottom_layers:
        for name, param in model.named_parameters():
            if name.startswith('roberta.encoder.layer') and int(name.split('.')[3]) < 5:
                param.requires_grad = False

    # Initialize Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=output_dir),
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_func

    )

    return trainer

"""##Evaluation on nr of digits"""

def calculate_metrics(original_nace_codes, pred_nace_codes, level):
  original_nace_level = [code[:level] for code in original_nace_codes]
  pred_nace_level = [code[:level] for code in pred_nace_codes]

  # Calculate accuracy
  accuracy = accuracy_score(original_nace_level, pred_nace_level)

  # Calculate F1-score
  f1 = f1_score(original_nace_level, pred_nace_level, average='weighted')

  return accuracy, f1

def correct_nace_codes(nace_labels):
    """
    Corrects NACE codes by adding leading zeros to labels with 4 digits.

    Args:
    - nace_labels (list of str): List of NACE codes to be corrected.

    Returns:
    - list of str: List of corrected NACE codes.
    """
    corrected_labels = []
    for label in nace_labels:
        if len(label) == 4 and label.isdigit():
            label = "0" + label
        corrected_labels.append(label)
    return corrected_labels

def evaluate_nace_code_classification(labels_corrected, pred_labels_corrected):
    """
    Evaluate NACE code classification performance at different levels of classification.

    Args:
    - val_labels_corrected (list of str): List of corrected original NACE codes.
    - val_pred_labels_corrected (list of str): List of corrected predicted NACE codes.

    Returns:
    - dict: Dictionary containing accuracy and F1-score for each level of classification.
    """
    evaluation_results = {}
    for level in range(1, 6):
        accuracy, f1 = calculate_metrics(labels_corrected, pred_labels_corrected, level)
        evaluation_results[f"{level} digits"] = {"Accuracy": accuracy, "F1-score": f1}
    return evaluation_results

"""#Main"""

def main():
    """
    Main function to execute the entire pipeline of loading, preprocessing, training, and evaluating the model.
    """
    # Directory containing the data
    directory = '/content/drive/...'

    # Median time value for downsampling
    time_median = 1

    # Load the data
    df = load_data(directory)

    # Downsample the data
    df_downsampled = downsample_data(df, time_median)

    # Split the dataset into training, validation, and testing sets
    x, y, train_texts, test_texts, val_texts, train_labels, test_labels, val_labels = split_data(df)

    # Calculate the number of unique labels
    num_labels = calculate_num_labels(df)

    # Tokenize the texts
    tokenizer = AutoTokenizer.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
    train_encodings = tokenize_texts(train_texts, tokenizer)
    val_encodings = tokenize_texts(val_texts, tokenizer)
    test_encodings = tokenize_texts(test_texts, tokenizer)

    # Encode the labels
    label2id = {label: id for id, label in enumerate(sorted(set(y)))}
    inverse_label_map = {id: label for label, id in label2id.items()}

    train_labels_ids = encode_labels(train_labels, label2id)
    val_labels_ids = encode_labels(val_labels, label2id)
    test_labels_ids = encode_labels(test_labels, label2id)

    # Create datasets
    train_dataset = create_dataset(train_encodings, train_labels_ids)
    val_dataset = create_dataset(val_encodings, val_labels_ids)
    test_dataset = create_dataset(test_encodings, test_labels_ids)

    # Load the configuration
    config = CustomRobertaConfig.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base')
    # Modify the configuration as needed
    config.num_labels = num_labels  # Change the number of labels if necessary
    config.out_features = 4096  # Modify the out_features as desired

    # Load the model
    model = CustomRobertaForSequenceClassification.from_pretrained('DTAI-KULeuven/robbert-2023-dutch-base', config=config)
    print(model)

    # Specify output and logging directories
    output_dir = '/content/drive/...'
    logging_dir = '/content/drive/...'

    # Ensure user confirmation before continuing
    user_input = input("MAKE SURE YOU CHECK THE DIRECTORIES BEFORE RUNNING! Continue (yes/no)?")
    if user_input != 'yes':
        raise ValueError('User did not respond with \'yes\'.')

    # Check if required directory exists
    if not os.path.isdir('/content/drive/...'):
        raise FileNotFoundError('Folder not found')

    # Create trainer for training
    trainer = create_trainer(model, train_dataset, val_dataset, output_dir, logging_dir, num_train_epochs=1, freeze_bottom_layers=True)

    # Train the model
    trainer.train()

    # Specify model checkpoint directory
    model_dir = '/content/drive/...'

    # Create trainer for evaluation
    trainer = trainer_for_evaluation(model_dir, output_dir, val_dataset, compute_metrics, freeze_bottom_layers=True)

    # Make predictions on validation dataset
    val_predictions = trainer.predict(val_dataset)
    val_pred_label_ids = val_predictions.predictions.argmax(axis=1).tolist()
    val_pred_labels = [inverse_label_map[label_id] for label_id in val_pred_label_ids]

    # Correct predicted NACE codes if necessary
    val_pred_labels_corrected = correct_nace_codes(val_pred_labels)
    val_labels_corrected = correct_nace_codes(val_labels.to_list())

    # Evaluate NACE code classification
    val_evaluation_results = evaluate_nace_code_classification(val_labels_corrected, val_pred_labels_corrected)
    for level, metrics in val_evaluation_results.items():
        print(f"{level}: Accuracy = {metrics['Accuracy']}; F1 = {metrics['F1-score']}")

    # Make predictions on test dataset
    test_predictions = trainer.predict(test_dataset)
    test_pred_label_ids = test_predictions.predictions.argmax(axis=1).tolist()
    test_pred_labels = [inverse_label_map[label_id] for label_id in test_pred_label_ids]

    # Correct predicted NACE codes if necessary
    test_pred_labels_corrected = correct_nace_codes(test_pred_labels)
    test_labels_corrected = correct_nace_codes(test_labels.to_list())

    # Evaluate NACE code classification
    test_evaluation_results = evaluate_nace_code_classification(test_labels_corrected, test_pred_labels_corrected)
    for level, metrics in test_evaluation_results.items():
        print(f"{level}: Accuracy = {metrics['Accuracy']}; F1 = {metrics['F1-score']}")


if __name__ == "__main__":
    main()