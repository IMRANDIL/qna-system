from fastapi import FastAPI, Form
from qna_pipeline import QnAPipeline
from pdf_processor import PDFProcessor
from fine_tuner import FineTuner

app = FastAPI()
# pdf_processor = PDFProcessor(pdf_path='app/data/samplepdf.pdf')
# qna_pipeline = QnAPipeline()
# fine_tuner = FineTuner()

# @app.post("/ask/")
# async def ask_question(question: str = Form(...)):
#     context = pdf_processor.extract_text()
#     answer = qna_pipeline.get_answer(question, context)
#     return {"answer": answer}
pdf_processor = PDFProcessor(pdf_path='data/samplepdf.pdf')
@app.post("/fine-tune/")
async def fine_tune_model():
    
    # qna_pipeline = QnAPipeline()
    fine_tuner = FineTuner()
    context = pdf_processor.extract_text()
    message = fine_tuner.fine_tune(context)
    return {"message": message}
