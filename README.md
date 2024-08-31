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

## Some Glimpse 

![Screenshot 2024-08-31 104823](https://github.com/user-attachments/assets/6b464362-5875-440c-bbc2-cc59edffe90b)


![Screenshot 2024-08-31 104844](https://github.com/user-attachments/assets/4367fb5a-6dec-4b04-a2c6-1ac0733d2717)

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request. We are open to suggestions and improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
