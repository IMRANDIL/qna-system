from fastapi import FastAPI, Form, HTTPException
from qna_pipeline import QnAPipeline
from pdf_processor import PDFProcessor
from fine_tuner import FineTuner
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_processor = PDFProcessor(pdf_path='data/samplepdf.pdf')
# qna_pipeline = QnAPipeline()
fine_tuner = FineTuner()

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        context = pdf_processor.extract_text()
        answer = fine_tuner.answer_question(context, question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-tune/")
async def fine_tune_model():
    try:
        context = pdf_processor.extract_text()
        message = fine_tuner.fine_tune(context)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}