import { useEffect, useRef } from 'react'
import './ChatWindow.css'

export default function ChatWindow({ messages, streaming, currentAgent }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming, currentAgent])

  if (messages.length === 0 && !streaming) {
    return (
      <div className="chat-window chat-empty">
        <div className="empty-state">
          <div className="empty-icon">⚡</div>
          <h2 className="empty-title">Personal Assistant</h2>
          <p className="empty-sub">Ask anything, search your profile, or generate a LinkedIn post.</p>
          <div className="empty-examples">
            <div className="example-chip">What skills do I have?</div>
            <div className="example-chip">Write a LinkedIn post about my latest project</div>
            <div className="example-chip">What happened in AI this week?</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-window">
      <div className="messages-list">
        {messages.map((msg, i) => (
          <Message key={msg.id ?? i} msg={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}


// ── Individual message ────────────────────────────────────────────────────

function Message({ msg }) {

  if (msg.role === 'user') {
    return (
      <div className="msg-row msg-user">
        <div className="msg-bubble msg-bubble-user">
          <MessageContent content={msg.content} />
        </div>
      </div>
    )
  }

  // Loading — show live pipeline log
  if (msg.type === 'loading') {
    return (
      <div className="msg-row msg-assistant">
        <div className="msg-avatar">AI</div>
        <div className="msg-bubble msg-bubble-assistant pipeline-bubble">
          <PipelineLog steps={msg.agentSteps || []} live />
        </div>
      </div>
    )
  }

  if (msg.type === 'error') {
    return (
      <div className="msg-row msg-assistant">
        <div className="msg-avatar">AI</div>
        <div className="msg-bubble msg-bubble-error">
          <span>⚠️ </span>
          <MessageContent content={msg.content} />
        </div>
      </div>
    )
  }

  if (msg.type === 'linkedin') {
    return (
      <div className="msg-row msg-assistant">
        <div className="msg-avatar li-avatar">in</div>
        <div className="msg-bubble msg-bubble-linkedin">
          {msg.agentSteps?.length > 0 && (
            <PipelineLog steps={msg.agentSteps} collapsible />
          )}
          <div className="linkedin-header">
            <span className="linkedin-badge">LinkedIn Post</span>
            <CopyButton text={msg.content} />
          </div>
          <div className="linkedin-post">{msg.content}</div>
        </div>
      </div>
    )
  }

  if (msg.type === 'hitl_answers') {
    return (
      <div className="msg-row msg-user">
        <div className="msg-bubble msg-bubble-user hitl-answers-bubble">
          <div className="hitl-answers-label">Your answers</div>
          <MessageContent content={msg.content} />
        </div>
      </div>
    )
  }

  // Default chat
  return (
    <div className="msg-row msg-assistant">
      <div className="msg-avatar">AI</div>
      <div className="msg-bubble msg-bubble-assistant">
        {msg.agentSteps?.length > 1 && (
          <PipelineLog steps={msg.agentSteps} collapsible />
        )}
        <MessageContent content={msg.content} />
      </div>
    </div>
  )
}


// ── Pipeline log component ────────────────────────────────────────────────
// Shows the sequence of agent steps with their detail lines.
// "live" = currently running (shows a pulsing dot on the last item)
// "collapsible" = show a toggle to expand/collapse

import { useState } from 'react'

const AGENT_ICONS = {
  Memory:          '🧠',
  Router:          '🔀',
  Researcher:      '🔍',
  Clarifier:       '🤔',
  Generator:       '✍️',
  Evaluator:       '🔎',
  'Style Matcher': '🎨',
  Chatbot:         '💬',
  'Database Search': '🗃️',
}

function PipelineLog({ steps, live = false, collapsible = false }) {
  const [open, setOpen] = useState(false)

  if (!steps || steps.length === 0) {
    if (!live) return null
    // No steps yet but live — show spinner
    return (
      <div className="pipeline-log pipeline-live">
        <div className="pipeline-spinner" />
        <span className="pipeline-waiting">Starting…</span>
      </div>
    )
  }

  if (collapsible && !open) {
    return (
      <button className="pipeline-toggle" onClick={() => setOpen(true)}>
        ▶ Pipeline · {steps.length} steps
      </button>
    )
  }

  return (
    <div className={`pipeline-log ${live ? 'pipeline-live' : ''}`}>
      {collapsible && (
        <button className="pipeline-toggle pipeline-toggle-open" onClick={() => setOpen(false)}>
          ▼ Pipeline · {steps.length} steps
        </button>
      )}
      {steps.map((step, i) => {
        const isLast = i === steps.length - 1
        const icon   = AGENT_ICONS[step.agent] || '•'
        return (
          <div key={i} className={`pipeline-step ${isLast && live ? 'pipeline-step-active' : ''}`}>
            <div className="pipeline-step-left">
              <span className="pipeline-icon">{icon}</span>
              <div className="pipeline-line" />
            </div>
            <div className="pipeline-step-body">
              <span className="pipeline-agent">{step.agent}</span>
              {step.detail && (
                <span className="pipeline-detail">{step.detail}</span>
              )}
              {isLast && live && <div className="pipeline-pulse" />}
            </div>
          </div>
        )
      })}
    </div>
  )
}


// ── Markdown content renderer ─────────────────────────────────────────────

function MessageContent({ content }) {
  if (!content) return null

  const lines      = content.split('\n')
  const rendered   = []
  let inCodeBlock  = false
  let codeLines    = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (line.startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true
        codeLines   = []
      } else {
        rendered.push(
          <pre key={i} className="code-block">
            <code>{codeLines.join('\n')}</code>
          </pre>
        )
        inCodeBlock = false
        codeLines   = []
      }
      continue
    }

    if (inCodeBlock) { codeLines.push(line); continue }

    rendered.push(
      <p key={i} className="msg-para">
        <InlineMarkdown text={line} />
      </p>
    )
  }

  return <div className="msg-content">{rendered}</div>
}

function InlineMarkdown({ text }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**'))
          return <strong key={i}>{part.slice(2, -2)}</strong>
        if (part.startsWith('`') && part.endsWith('`'))
          return <code key={i} className="inline-code">{part.slice(1, -1)}</code>
        return part
      })}
    </>
  )
}

function CopyButton({ text }) {
  const copy = () => navigator.clipboard.writeText(text).catch(() => {})
  return (
    <button className="copy-btn" onClick={copy} title="Copy">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
      </svg>
      Copy
    </button>
  )
}
