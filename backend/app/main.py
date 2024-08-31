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
fine_tuner = FineTuner()
qna_pipeline = None  # We'll initialize this after fine-tuning

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    qna_pipeline = QnAPipeline()
    try:
        if qna_pipeline is None:
            raise HTTPException(status_code=400, detail="Model not fine-tuned yet. Please call /fine-tune/ first.")
        
        context = pdf_processor.extract_text()
        answer = qna_pipeline.get_answer(question, context)
        print(answer)
        return {"answer": answer['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-tune/")
async def fine_tune_model():
    # global qna_pipeline
    try:
        context = pdf_processor.extract_text()
        message = fine_tuner.fine_tune(context)
        
        # After fine-tuning, initialize the QnAPipeline with the new model
        # qna_pipeline = QnAPipeline()
        
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}