interface AnswerDisplayProps {
    answer: string;
}

function AnswerDisplay({ answer }: AnswerDisplayProps) {
    return (
        <div>
            <h3>Answer:</h3>
            <p>{answer}</p>
        </div>
    );
}

export default AnswerDisplay;
