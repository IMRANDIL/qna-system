# Q&A System

This is a simple Question and Answer system built with FastAPI, React, and Hugging Face's Transformers library. It uses a pre-trained language model to generate answers to user questions.

## Installation

To install the necessary dependencies, run the following commands:

```bash
pip install -r backend/app/requirements.txt
npm install
```
## Fix for os Error

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

## Usage

To start the backend server, navigate to the `backend` directory and run:

```bash
uvicorn app.main:app --reload
```

To start the frontend, navigate to the `frontend` directory and run:

```bash
npm run dev
```

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request. We are open to suggestions and improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
