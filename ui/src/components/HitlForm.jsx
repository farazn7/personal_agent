import { useState } from 'react'
import './HitlForm.css'

export default function HitlForm({ questions, onSubmit, onSkip }) {
  const [answers, setAnswers] = useState(() =>
    Object.fromEntries(questions.map(q => [q, '']))
  )

  const allAnswered = questions.every(q => answers[q]?.trim())

  const handleSubmit = () => {
    const filled = Object.fromEntries(
      Object.entries(answers).filter(([, v]) => v.trim())
    )
    onSubmit(filled)
  }

  return (
    <div className="hitl-container">
      <div className="hitl-card">
        <div className="hitl-title-row">
          <span className="hitl-icon">🤔</span>
          <span className="hitl-title">A few details before I write</span>
        </div>
        <p className="hitl-desc">
          I found some context online, but need a couple of specifics from you
          to avoid making things up.
        </p>

        <div className="hitl-questions">
          {questions.map((q, i) => (
            <div key={i} className="hitl-q">
              <label className="hitl-label">
                <span className="hitl-q-num">{i + 1}.</span> {q}
              </label>
              <textarea
                className="hitl-input"
                rows={2}
                placeholder="Your answer…"
                value={answers[q] || ''}
                onChange={e => setAnswers(prev => ({ ...prev, [q]: e.target.value }))}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey && allAnswered) {
                    e.preventDefault()
                    handleSubmit()
                  }
                }}
              />
            </div>
          ))}
        </div>

        <div className="hitl-actions">
          <button
            className="hitl-btn hitl-btn-primary"
            onClick={handleSubmit}
            disabled={!allAnswered}
          >
            ✅ Submit &amp; generate
          </button>
          <button
            className="hitl-btn hitl-btn-ghost"
            onClick={onSkip}
          >
            ⏭ Skip — write with what you have
          </button>
        </div>
      </div>
    </div>
  )
}
