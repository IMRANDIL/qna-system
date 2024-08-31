from fastapi import FastAPI, Form
from app.qna_pipeline import QnAPipeline
from app.pdf_processor import PDFProcessor
from app.fine_tuner import FineTuner

app = FastAPI()
pdf_processor = PDFProcessor(pdf_path='app/data/your_pdf_file.pdf')
qna_pipeline = QnAPipeline()
fine_tuner = FineTuner()

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    context = pdf_processor.extract_text()
    answer = qna_pipeline.get_answer(question, context)
    return {"answer": answer}

@app.post("/fine-tune/")
async def fine_tune_model():
    context = pdf_processor.extract_text()
    fine_tuner.fine_tune(context)
    return {"message": "Model fine-tuned successfully"}
