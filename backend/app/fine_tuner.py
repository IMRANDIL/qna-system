import json
from datasets import Dataset
from transformers import (AutoModelForQuestionAnswering, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer)

class FineTuner:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def prepare_data(self, context):
        # Simulate question-answer pairs from the PDF context
        # This can be done programmatically, but for demo purposes we'll use dummy data
        examples = [{
            'context': context,
            'question': "What is the content about?",
            'answers': {'text': ["Answer from context"], 'answer_start': [0]}
        }]

        return Dataset.from_dict(examples)

    def fine_tune(self, context):
        # Prepare data for fine-tuning
        dataset = self.prepare_data(context)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./models',
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_dir='./logs',
            logging_steps=10,
        )

        # Define Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        # Fine-tune the model
        trainer.train()
        trainer.save_model('./models/finetuned_model')

        return "Model fine-tuned successfully!"

