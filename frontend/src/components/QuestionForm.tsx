import { useState } from 'react';

interface QuestionFormProps {
    onQuestionSubmit: (question: string) => void;
}

function QuestionForm({ onQuestionSubmit }: QuestionFormProps) {
    const [question, setQuestion] = useState('');

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        onQuestionSubmit(question);
    };

    return (
        <form onSubmit={handleSubmit}>
            <div>
                <label>Question: </label>
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    required
                />
            </div>
            <button type="submit">Ask</button>
        </form>
    );
}

export default QuestionForm;
