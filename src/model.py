import numpy as np
from sklearn.metrics import f1_score


from constants import class_weights, id2label, label2id
import helper_funcs as f


class SequenceClassifier:
    def __init__(
        self,
        subtask: str,
        model_name_or_path: str,
        remove_links_hashtags: bool = True,
        with_class_weights: bool = True,
    ) -> None:
        
        self.subtask = subtask
        self.model_name_or_path = model_name_or_path
        self.remove_links_hashtags = remove_links_hashtags
        self.with_class_weights = with_class_weights
        
        self.dataset = f.load_hf_dataset(subtask)

        self.tokenizer = f.load_pretrained_tokenizer(model_name_or_path)
        self.data_collator = f.load_data_collator(self.tokenizer)
        self.model = f.load_pretrained_model(subtask, model_name_or_path)

        self.arabert_prep = f.load_arabert_prep(model_name_or_path)

        self.dataset = f.preprocess_dataset(self.dataset, self.arabert_prep, remove_links_hashtags)

        
        if subtask == "1B":
            self.compute_metrics = f.compute_metrics_multilabel
        else:
            self.compute_metrics = f.compute_metrics_singlelabel
        
        self.class_weights = class_weights[subtask] if with_class_weights else None
        
        self.label2id = label2id[subtask]
        self.id2label = id2label[subtask]
    def preprocess_function(self, examples: dict):
        """
        Preprocesses the text using the tokenizer
        
        Args:
            examples (Iterable[Dict[str, Any]]): examples from dataset

        Returns:
            dict: tokenized batch
        """
        tokenized_batch = self.tokenizer(examples['text'], truncation=True)
        tokenized_batch["label"] = [self.label2id[label] for label in examples["label"]]
        return tokenized_batch
    
    def tokenize_dataset(self):
        """
        Tokenizes the dataset
        """
        self.dataset = self.dataset.map(
            self.preprocess_function,
            batched=True
        )

    def generate_out_model_name(self):
        """
        Generates the output model name
        """

        model_name = f"{self.model_name_or_path.split('/')[-1]}"
        model_name += "_no-links-hashtags" if self.remove_links_hashtags else ""
        model_name += "_with-class-weights" if self.with_class_weights else ""
        return model_name

    def train_and_evaluate(self):
        """
        Trains the model
        """
        self.tokenize_dataset()
        training_args = f.get_training_args(f"/scratch/s4992113/arabicai/models/{self.subtask}/{self.generate_out_model_name()}")
        self.trainer = f.get_trainer(self.model, training_args, self.dataset, self.tokenizer, self.data_collator, self.compute_metrics, self.class_weights)
        
        self.trainer.train()
        
        self.evaluate(dataset=self.dataset["dev"], split="dev")
        self.evaluate(dataset=self.dataset["test"], split="test")




    def evaluate(self, dataset, split="dev"):
        predictions = np.argmax(self.trainer.predict(dataset).predictions, axis=1)
        predictions = [self.id2label[id_] for id_ in predictions]

        f.write_predictions_to_tsv(
            dataset,
            predictions,
            f"predictions/{self.subtask}/{split}/{self.generate_out_model_name()}.tsv"
        )
        ground_truth = [self.id2label[row['label']] for row in dataset]

        micro_f1 = f1_score(ground_truth, predictions, average='micro')
        macro_f1 = f1_score(ground_truth, predictions, average='macro')

        print(f"Evaluation on {split} set:")
        print(f"Micro F1: {micro_f1}")
        print(f"Macro F1: {macro_f1}")
        print("\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", "-s", required=True, choices=['1A', '1B', '2A', '2B'],
                        help="The subtask for which we are checking format: 1A, 1B, 2A, 2B", type=str)
    parser.add_argument("--model_name_or_path", "-m", required=True,
                        help="The model name or path to the model", type=str)
    parser.add_argument("--remove_links_hashtags", "-r", action="store_true",
                        help="Whether to remove links and hashtags from the text")
    parser.add_argument("--with_class_weights", "-w", action="store_true",
                        help="Whether to use class weights")

    args = parser.parse_args()

    subtask = args.subtask
    model_name_or_path = args.model_name_or_path
    remove_links_hashtags = args.remove_links_hashtags
    with_class_weights = args.with_class_weights

    classifier = SequenceClassifier(
        subtask,
        model_name_or_path,
        remove_links_hashtags,
        with_class_weights
    )

    classifier.train_and_evaluate()
    
