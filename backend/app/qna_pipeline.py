from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

class QnAPipeline:
    def __init__(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained('./models/finetuned_model')
        self.tokenizer = AutoTokenizer.from_pretrained('./models/finetuned_model')
        self.qna_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)

    def get_answer(self, question, context):
        result = self.qna_pipeline({'question': question, 'context': context})
        print(result)
        return result  # Return the full result dictionary