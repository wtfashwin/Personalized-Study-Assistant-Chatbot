import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from tqdm.auto import tqdm
import collections
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QAFineTuner:
    """
    A class to encapsulate the fine-tuning process for a Question Answering model.
    This class adheres to OOP principles by managing model, tokenizer, data, and training.
    """
    def __init__(self, model_name: str, output_dir: str = "./fine_tuned_model"):
        """
        Initializes the QA Fine-tuner with a pre-trained model and tokenizer.

        Args:
            model_name (str): The name of the pre-trained model from Hugging Face Hub.
            output_dir (str): Directory to save the fine-tuned model and tokenizer.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        logger.info(f"Initialized model: {model_name} and tokenizer.")

        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare_train_features(self, examples):
        """
        Preprocesses training examples for Question Answering.
        This function handles tokenization, mapping answers to token spans,
        and managing long contexts. It's a critical part of QA fine-tuning.

        Args:
            examples (dict): A batch of examples from the dataset.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding: Tokenized features.
        """
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True, 
            return_offsets_mapping=True,    
            padding="max_length",       
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id) # Find the CLS token index

            sequence_ids = tokenized_examples.sequence_ids(i)
            
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1: 
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1: 
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def fine_tune(self, num_train_epochs: int = 1, batch_size: int = 1):
        """
        Performs the fine-tuning process with highly optimized settings for low RAM.

        Args:
            num_train_epochs (int): Number of training epochs. Set to 1 for quick demo.
            batch_size (int): Per device train batch size. Set to 1 for minimal RAM.
        """
        logger.info("Loading a very small subset of SQuAD v2 dataset for extremely low RAM...")
        try:
            raw_datasets = load_dataset("squad_v2", split={"train": "train[:100]", "validation": "validation[:20]"})
            train_dataset = raw_datasets["train"]
            eval_dataset = raw_datasets["validation"]
            logger.info(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} validation examples.")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}. Please ensure 'datasets' library is installed.")
            return

        logger.info("Preprocessing training features...")
        # Apply the preprocessing function to the datasets
        processed_train_dataset = train_dataset.map(
            self._prepare_train_features,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on training dataset",
        )
        logger.info("Preprocessing validation features...")
        processed_eval_dataset = eval_dataset.map(
            self._prepare_train_features,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Running tokenizer on validation dataset",
        )

        logger.info("Setting up TrainingArguments for extremely low RAM...")
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,  
            gradient_accumulation_steps=8,          
            eval_strategy="epoch",                  
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",                  
            save_total_limit=1,                     
            learning_rate=3e-5,                     
            weight_decay=0.01,
            report_to="none",                       
            dataloader_num_workers=0,               
        )

        logger.info("Initializing Trainer...")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_train_dataset,
            eval_dataset=processed_eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator, 
        )

        logger.info("Starting training...")
        try:
            trainer.train()
            logger.info("Training complete!")
        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
            return

        logger.info(f"Saving the fine-tuned model and tokenizer to {self.output_dir}...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Model and tokenizer saved successfully.")

if __name__ == '__main__':
    MODEL_NAME = "deepset/roberta-base-squad2"
    OUTPUT_DIR = "./fine_tuned_model"
    qa_finetuner = QAFineTuner(model_name=MODEL_NAME, output_dir=OUTPUT_DIR)
    qa_finetuner.fine_tune(num_train_epochs=1, batch_size=1)
    logger.info("Fine-tuning script finished execution.")
    logger.info("You can now use the model saved in './fine_tuned_model' for RAG.")
