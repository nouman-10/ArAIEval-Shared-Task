import csv

import pandas as pd
from arabert.preprocess import ArabertPreprocessor
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
import torch.nn as nn

from constants import subtask_files, class_weights, id2label, label2id


# This is run for the task 2A and 2B before training
def clean_and_overwrite_jsonl(filepath: str) -> None:
    """Load a JSONL file, replace NaN in the 'text' column with the string "nan", and overwrite the original file.

    Args:
        filepath (str): filepath to read and overwrite
    """
    df = pd.read_json(filepath, lines=True)
    df['text'].fillna('nan', inplace=True)
    df.to_json(filepath, orient='records', lines=True)


def load_hf_dataset(subtask: str) -> DatasetDict:
    """Load the HuggingFace dataset for a given subtask

    Args:
        subtask (str): subtask, from "1A", "1B", "2A", "2B"

    Returns:
        DatasetDict: HuggingFace dataset for the given subtask
    """
    dataset = load_dataset('json', data_files=subtask_files[subtask])
    return dataset

def remove_hashtag(text: str) -> str:
    """Remove hashtags from text

    Args:
        text (str): text to remove hashtags from

    Returns:
        str: text with hashtags removed
    """
    return " ".join([word for word in text.split() if not word.startswith("#")])


def remove_link_and_hashtags(example: dict) -> dict:
    """Remove links and hashtags from text

    Args:
        example (dict): example from dataset

    Returns:
        dict: example with links and hashtags removed
    """
    example["text"] = [row.replace("LINK", "").strip() for row in example['text']]
    example['text'] = [remove_hashtag(row) for row in example['text']]
    return example


def load_arabert_prep(model_name_or_path: str = "aubmindlab/bert-base-arabertv02") -> ArabertPreprocessor:
    """Load the ArabertPreprocessor

    Args:
        model_name_or_path (str, optional): model name or path to use for preprocessing. Defaults to "aubmindlab/bert-base-arabertv02".

    Returns:
        ArabertPreprocessor: ArabertPreprocessor
    """
    arabert_prep = ArabertPreprocessor(
        model_name=model_name_or_path, 
        strip_tashkeel = True,
        strip_tatweel = True,
        insert_white_spaces = True,
        remove_non_digit_repetition = True,
        apply_farasa_segmentation = True,
        map_hindi_numbers_to_arabic = True
    )
    return arabert_prep


def preprocess_arabert(example: dict, arabert_prep: ArabertPreprocessor) -> dict:
    """
    Preprocesses the text using arabert_prep.preprocess
    Args:
        example (dict): example from dataset
        model_name_or_path (str): model name or path to use for preprocessing

    Returns:
        dict: example with preprocessed text
    """
    example['text'] = arabert_prep.preprocess(example['text'])
    return example


def preprocess_dataset(dataset: DatasetDict, arabert_prep: ArabertPreprocessor, remove_hashtags: bool = True):
    """Preprocess the dataset using the ArabertPreprocessor

    Args:
        dataset (DatasetDict): dataset to preprocess
        model_name_or_path (str): model name or path to use for preprocessing
        remove_hashtags (bool, optional): whether to remove hashtags from the dataset. Defaults to True.

    Returns:
        DatasetDict: preprocessed dataset
    """
    if remove_hashtags:
        dataset = dataset.map(remove_link_and_hashtags)

    dataset = dataset.map(lambda example: preprocess_arabert(example, arabert_prep))
    return dataset


def load_pretrained_tokenizer(model_name_or_path: str = "aubmindlab/bert-base-arabertv02") -> PreTrainedTokenizer:
    """Load the pretrained tokenizer
    
    Args:
        model_name_or_path (str, optional): model name or path to use for preprocessing. Defaults to "aubmindlab/bert-base-arabertv02".

    Returns:
        AutoTokenizer: pretrained tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def load_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    """Load the data collator
    
    Args:
        tokenizer (AutoTokenizer): pretrained tokenizer

    Returns:
        DataCollatorWithPadding: data collator
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator


def load_pretrained_model(subtask: str, model_name_or_path: str = "aubmindlab/bert-base-arabertv02") -> AutoModelForSequenceClassification:
    """Load the pretrained model

    Args:
        subtask (str): subtask, from "1A", "1B", "2A", "2B"
        model_name_or_path (str, optional): model name or path to use for preprocessing. Defaults to "aubmindlab/bert-base-arabertv02".

    Returns:    
        AutoModelForSequenceClassification: pretrained model
    """
    additional_kwargs = {}
    if subtask == "1B":
        # For subtask 1B, we need to use a multi-label classification model
        additional_kwargs = {"problem_type": "multi_label_classification"}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(id2label[subtask]),
        id2label=id2label[subtask],
        label2id=label2id[subtask],
        **additional_kwargs
    )
    
    return model

def get_training_args(out_model_name: str, epochs: int = 20, metric: str = "f1", push_to_hub: bool = False) -> TrainingArguments:
    """Get the training arguments for the Trainer
    
    Args:
        out_model_name (str): output model name
        epochs (int, optional): number of epochs. Defaults to 10.
        metric (str, optional): metric to use for early stopping. Defaults to "f1".
        push_to_hub (bool, optional): whether to push the model to the HuggingFace Hub. Defaults to False.

    Returns:
        TrainingArguments: training arguments for the Trainer
    """
    return TrainingArguments(
        output_dir=out_model_name,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=epochs,
        metric_for_best_model=metric,
        push_to_hub=push_to_hub,
    )


def load_metric(metric_name: str):
    """Load the metric
    
    Args:
        metric_name (str): metric name

    Returns:
        evaluate.EvalMetric: metric
    """
    return evaluate.load(metric_name)

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics_multilabel(p):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

def compute_metrics_singlelabel(eval_pred):
    predictions, labels = eval_pred
    f1 = load_metric("f1")
    acc = load_metric("accuracy")

    predictions = np.argmax(predictions, axis=1)
    f1_score = f1.compute(predictions=predictions, references=labels, average='micro')
    acc_score = acc.compute(predictions=predictions, references=labels)
    return {'f1': f1_score['f1'], 'acc': acc_score['accuracy']}

def get_trainer(
        model: AutoModelForSequenceClassification, 
        training_args: TrainingArguments, 
        tokenized_dataset: DatasetDict, 
        tokenizer: AutoTokenizer, 
        data_collator: DataCollatorWithPadding, 
        compute_metrics=compute_metrics_singlelabel,
        class_weights=None
    ):
    """Get the Trainer
    
    Args:
        model (AutoModelForSequenceClassification): pretrained model
        training_args (TrainingArguments): training arguments for the Trainer
        tokenized_dataset (DatasetDict): tokenized dataset
        tokenizer (AutoTokenizer): pretrained tokenizer
        data_collator (DataCollatorWithPadding): data collator
        class_weights (list, optional): class weights. Defaults to None.

    Returns:
        Trainer: Trainer
    """

    return CustomTrainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset["train"],
      eval_dataset=tokenized_dataset["dev"],
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
      class_weights=class_weights
    )


def write_predictions_to_tsv(dataset, predictions, output_file='output.tsv'):
    """
    Write predictions and IDs from a dataset to a TSV file.

    Args:
        dataset (Dataset): The dataset containing the IDs.
        predictions (List[str]): A list of predicted labels.
        output_file (str, optional): The path to the TSV file to write. Defaults to 'output.tsv'.
    """

    # Extract IDs from the dataset
    ids = dataset['id']  # Adjust the split if necessary

    # Prepare data for writing to TSV
    data = list(zip(ids, predictions))

    # Write to TSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'label'])  # Header
        writer.writerows(data)

    print(f"Data written to {output_file}")


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = torch.Tensor(class_weights).to(device) if class_weights else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

