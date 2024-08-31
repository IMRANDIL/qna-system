import json
from datasets import Dataset
from transformers import (AutoModelForQuestionAnswering, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer)

class FineTuner:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def prepare_data(self, context):
        print("Preparing data...")

        # Define questions and corresponding answers based on the context
        questions_and_answers = [
            {
                'question': "Who is the applicant?",
                'answer': "Ali Imran Adil"
            },
            {
                'question': "What is the subject of the FIR?",
                'answer': "Domestic Abuse, Harassment, and Mental Instability"
            },
            {
                'question': "What are the grievances mentioned in the FIR?",
                'answer': "Harassment and Abuse, False Allegations, Mental Harassment, Mental Instability, Effect on My Son"
            },
            {
                'question': "What evidence does the applicant have for verbal abuse?",
                'answer': "Evidence in chat messages"
            },
            {
                'question': "What impact has the harassment had on the applicant's mental well-being?",
                'answer': "Unable to focus on daily activities and job responsibilities due to constant stress and anxiety"
            },
            {
                'question': "What does the applicant suspect about his wife's mental condition?",
                'answer': "The applicant believes his wife's mental condition is unstable, based on video, audio, and chat evidence"
            },
            {
                'question': "How is the applicant's son affected?",
                'answer': "The applicant's 4-year-old son is being affected emotionally and psychologically due to his mother's behavior"
            }
        ]

        # Separate lists for the dataset
        contexts = []
        questions = []
        answers_text = []
        answers_start = []

        for qa in questions_and_answers:
            question = qa['question']
            answer = qa['answer']
            start_index = context.find(answer)
            if start_index == -1:
                print(f"Warning: Answer not found in context for question: '{question}'")
                continue  # Skip if the answer is not found in the context

            contexts.append(context)
            questions.append(question)
            answers_text.append([answer])
            answers_start.append([start_index])

        # Create dictionary in the format expected by Dataset.from_dict
        data = {
            'context': contexts,
            'question': questions,
            'answers': {
                'text': answers_text,
                'answer_start': answers_start
            }
        }

        return Dataset.from_dict(data)

    def fine_tune(self, context):
        print("Preparing dataset for fine-tuning...")
        dataset = self.prepare_data(context)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./models',
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_dir='./logs',
            logging_steps=10,
            remove_unused_columns=False
        )

        print("Initializing Trainer...")
        # Define Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        print("Starting training...")
        # Fine-tune the model
        trainer.train()
        
        print("Saving the model...")
        trainer.save_model('./models/finetuned_model')

        return "Model fine-tuned successfully!"
