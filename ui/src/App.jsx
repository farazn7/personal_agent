import { useState, useEffect, useRef, useCallback } from 'react'
import Sidebar from './components/Sidebar.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import HitlForm from './components/HitlForm.jsx'
import './App.css'

const API = ''  // proxied by Vite to http://localhost:8000

const AGENT_LABELS = {
  Memory:          '🧠 Loading your memory…',
  Router:          '🔀 Routing…',
  Researcher:      '🔍 Researching the topic…',
  Clarifier:       '🤔 Checking what info is needed…',
  Generator:       '✍️  Writing your draft…',
  Evaluator:       '🔎 Reviewing for accuracy…',
  'Style Matcher': '🎨 Matching your voice…',
  Chatbot:         '💬 Thinking…',
  'Database Search': '🗃️  Searching your profile…',
}

export default function App() {
  const [threadId, setThreadId]         = useState(() => crypto.randomUUID())
  const [messages, setMessages]         = useState([])
  const [streaming, setStreaming]       = useState(false)
  const [currentAgent, setCurrentAgent] = useState('')
  const [hitlPending, setHitlPending]   = useState(false)
  const [hitlQuestions, setHitlQuestions] = useState([])
  const [hitlOriginal, setHitlOriginal] = useState('')
  const [ltmFacts, setLtmFacts]         = useState([])
  const [showLtm, setShowLtm]           = useState(false)
  const [stats, setStats]               = useState({})
  const [ollamaOk, setOllamaOk]         = useState(true)

  // ── Health check on mount ────────────────────────────────────
  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => r.json())
      .then(d => setOllamaOk(d.ollama))
      .catch(() => setOllamaOk(false))

    fetch(`${API}/api/stats`)
      .then(r => r.json())
      .then(setStats)
      .catch(() => {})
  }, [])

  // ── Load LTM facts ───────────────────────────────────────────
  const loadLtm = useCallback(() => {
    fetch(`${API}/api/ltm`)
      .then(r => r.json())
      .then(d => setLtmFacts(d.facts || []))
      .catch(() => {})
  }, [])

  useEffect(() => { if (showLtm) loadLtm() }, [showLtm, loadLtm])

  // ── Stream handler ───────────────────────────────────────────
  const streamResponse = useCallback(async (url, body, userMsg) => {
    setStreaming(true)
    setCurrentAgent('')

    // Add user message immediately
    if (userMsg) {
      setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    }

    // Add a placeholder assistant message that we'll fill in as we stream
    const assistantId = Date.now()
    setMessages(prev => [...prev, {
      id: assistantId, role: 'assistant', type: 'loading', content: ''
    }])

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      // Capture thread_id from header if server assigned one
      const newThreadId = res.headers.get('X-Thread-Id')
      if (newThreadId) setThreadId(newThreadId)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })

        const lines = buf.split('\n')
        buf = lines.pop()  // keep incomplete line in buffer

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const evt = JSON.parse(line.slice(6))
            handleSseEvent(evt, assistantId)
          } catch { /* incomplete JSON, skip */ }
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, type: 'error', content: `Connection error: ${err.message}` }
          : m
      ))
    } finally {
      setStreaming(false)
      setCurrentAgent('')
      // Refresh LTM after each turn (facts may have been extracted)
      if (showLtm) loadLtm()
    }
  }, [showLtm, loadLtm])

  const handleSseEvent = useCallback((evt, assistantId) => {
    switch (evt.type) {
      case 'agent': {
        setCurrentAgent(evt.agent)
        setMessages(prev => prev.map(m =>
          m.id === assistantId && m.type === 'loading'
            ? {
                ...m,
                agentSteps: [
                  ...(m.agentSteps || []),
                  { agent: evt.agent, detail: evt.detail || '' },
                ],
              }
            : m
        ))
        break
      }

      case 'chat':
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? { ...m, type: 'chat', content: evt.content, agentSteps: m.agentSteps || [] }
            : m
        ))
        break

      case 'linkedin':
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? { ...m, type: 'linkedin', content: evt.content, agentSteps: m.agentSteps || [] }
            : m
        ))
        break

      case 'hitl':
        setMessages(prev => prev.filter(m => m.id !== assistantId))
        setHitlQuestions(evt.questions || [])
        setHitlPending(true)
        // Server sends thread_id in hitl event — keep it in sync
        if (evt.thread_id) setThreadId(evt.thread_id)
        break

      case 'error':
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? { ...m, type: 'error', content: evt.content }
            : m
        ))
        break

      case 'done':
        // Stream complete — nothing extra to do
        break
    }
  }, [])

  // ── Send chat message ────────────────────────────────────────
  const sendMessage = useCallback((text) => {
    if (!text.trim() || streaming || hitlPending) return
    setHitlOriginal(text)  // track last sent message so HITL resume knows what to send
    streamResponse(
      `${API}/api/chat`,
      { message: text, thread_id: threadId },
      text
    )
  }, [streaming, hitlPending, threadId, streamResponse])

  // ── Submit HITL answers ──────────────────────────────────────
  const submitHitl = useCallback((answers) => {
    setHitlPending(false)
    setHitlQuestions([])

    // Show answers as a user message
    const summary = Object.entries(answers)
      .map(([q, a]) => `**${q}**\n${a}`)
      .join('\n\n')
    setMessages(prev => [...prev, {
      role: 'user', content: summary, type: 'hitl_answers'
    }])

    streamResponse(
      `${API}/api/hitl`,
      { answers, thread_id: threadId },
      null
    )
  }, [hitlOriginal, threadId, streamResponse])

  const skipHitl = useCallback(() => {
    setHitlPending(false)
    setHitlQuestions([])
    streamResponse(
      `${API}/api/hitl`,
      { answers: {}, thread_id: threadId },
      null
    )
  }, [hitlOriginal, threadId, streamResponse])

  // ── New session ──────────────────────────────────────────────
  const newSession = useCallback(() => {
    setThreadId(crypto.randomUUID())
    setMessages([])
    setHitlPending(false)
    setHitlQuestions([])
    setHitlOriginal('')
    setCurrentAgent('')
  }, [])

  // ── Load past session ─────────────────────────────────────────
  const loadSession = useCallback((tid) => {
    fetch(`${API}/api/session-messages?thread_id=${encodeURIComponent(tid)}`)
      .then(r => r.json())
      .then(data => {
        setThreadId(tid)
        setMessages((data.messages || []).map((m, i) => ({
          id: i,
          role: m.type === 'human' ? 'user' : 'assistant',
          type: m.type === 'human' ? undefined : 'chat',
          content: m.content,
        })))
        setHitlPending(false)
        setHitlQuestions([])
        setCurrentAgent('')
      })
      .catch(() => { setThreadId(tid); setMessages([]) })
  }, [])

  // ── Delete LTM fact ──────────────────────────────────────────
  const deleteFact = useCallback((key) => {
    fetch(`${API}/api/ltm/${encodeURIComponent(key)}`, { method: 'DELETE' })
      .then(() => loadLtm())
      .catch(() => {})
  }, [loadLtm])

  return (
    <div className="app-shell">
      <Sidebar
        threadId={threadId}
        onNewSession={newSession}
        onLoadSession={loadSession}
        showLtm={showLtm}
        onToggleLtm={() => setShowLtm(v => !v)}
        ltmFacts={ltmFacts}
        onDeleteFact={deleteFact}
        stats={stats}
        ollamaOk={ollamaOk}
      />
      <main className="main-area">
        {!ollamaOk && (
          <div className="banner banner-error">
            ⚠️ Ollama is not reachable — run <code>ollama serve</code> in a terminal
          </div>
        )}
        <ChatWindow
          messages={messages}
          streaming={streaming}
          currentAgent={currentAgent}
          onSend={sendMessage}
        />
        {hitlPending && (
          <HitlForm
            questions={hitlQuestions}
            onSubmit={submitHitl}
            onSkip={skipHitl}
          />
        )}
        <InputBar
          onSend={sendMessage}
          disabled={streaming || hitlPending}
        />
      </main>
    </div>
  )
}

// ── Input bar (inline — simple enough to keep here) ──────────────────────
function InputBar({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  const submit = () => {
    const text = value.trim()
    if (!text || disabled) return
    onSend(text)
    setValue('')
    textareaRef.current?.focus()
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [value])

  return (
    <div className="input-bar">
      <textarea
        ref={textareaRef}
        className="input-textarea"
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={
          disabled
            ? 'Waiting for response...'
            : 'Message… (Enter to send, Shift+Enter for new line)'
        }
        disabled={disabled}
        rows={1}
      />
      <button
        className="send-btn"
        onClick={submit}
        disabled={disabled || !value.trim()}
        title="Send (Enter)"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 2L11 13"/><path d="M22 2L15 22 11 13 2 9l20-7z"/>
        </svg>
      </button>
    </div>
  )
}
