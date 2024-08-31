import json
from datasets import Dataset
from transformers import (AutoModelForQuestionAnswering, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer,
                          default_data_collator)

class FineTuner:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def prepare_data(self, context):
        print("Preparing data...")

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

        data = []
        for qa in questions_and_answers:
            question = qa['question']
            answer = qa['answer']
            start_index = context.find(answer)
            if start_index == -1:
                print(f"Warning: Answer not found in context for question: '{question}'")
                continue

            data.append({
                'context': context,
                'question': question,
                'answer': {
                    'text': [answer],
                    'answer_start': [start_index]
                }
            })

        dataset = Dataset.from_list(data)

        def preprocess_function(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = self.tokenizer(
                questions,
                examples["context"],
                max_length=384,
                truncation="only_second",
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            answers = examples["answer"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                answer = answers[i]
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label it (0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs

        return dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    def fine_tune(self, context):
        print("Preparing dataset for fine-tuning...")
        dataset = self.prepare_data(context)
        
        training_args = TrainingArguments(
            output_dir='./models',
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_dir='./logs',
            logging_steps=10,
            remove_unused_columns=False,
            learning_rate=5e-5,
        )

        print("Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
        )

        print("Starting training...")
        trainer.train()
        
        print("Saving the model...")
        trainer.save_model('./models/finetuned_model')

        return "Model fine-tuned successfully!"