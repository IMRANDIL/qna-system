import { useState } from 'react';
import QuestionForm from './components/QuestionForm';
import AnswerDisplay from './components/AnswerDisplay';

interface AppProps {}

const App: React.FC<AppProps> = () => {
    const [answer, setAnswer] = useState<string>('');

    const handleQuestionSubmit = async (question: string) => {
        const formData = new FormData();
        formData.append('question', question);

        const response = await fetch('http://localhost:8000/ask/', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        setAnswer(data.answer);
    };

    return (
        <div>
            <h1>Q&A System</h1>
            <QuestionForm onQuestionSubmit={handleQuestionSubmit} />
            <AnswerDisplay answer={answer} />
        </div>
    );
}

export default App;
