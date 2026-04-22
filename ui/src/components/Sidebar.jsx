import { useState } from 'react'
import './Sidebar.css'

const CONF_ICON = { 1: '○', 2: '◐', 3: '●' }
const CAT_COLORS = {
  skill: '#7c6af7', achievement: '#22c55e', project: '#0ea5e9',
  event: '#f59e0b', role: '#ec4899', education: '#8b5cf6',
  preference: '#14b8a6', identity: '#f97316',
}

export default function Sidebar({
  threadId, onNewSession, onLoadSession,
  showLtm, onToggleLtm, ltmFacts, onDeleteFact,
  stats, ollamaOk,
}) {
  const [ltmFilter, setLtmFilter]       = useState('')
  const [sessions, setSessions]         = useState([])
  const [showSessions, setShowSessions] = useState(false)

  const loadSessions = () => {
    fetch('/api/sessions')
      .then(r => r.json())
      .then(setSessions)
      .catch(() => {})
  }

  const filteredFacts = ltmFacts.filter(f =>
    !ltmFilter || f.fact.toLowerCase().includes(ltmFilter.toLowerCase())
      || f.category.toLowerCase().includes(ltmFilter.toLowerCase())
  )

  const chromaFacts    = stats?.chroma?.personal_facts    ?? '–'
  const chromaPosts    = stats?.chroma?.linkedin_examples ?? '–'
  const ltmCount       = ltmFacts.length

  return (
    <aside className="sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <span className="sidebar-logo-icon">⚡</span>
          <span className="sidebar-logo-text">Personal AI</span>
        </div>
        <div className={`status-dot ${ollamaOk ? 'ok' : 'err'}`} title={ollamaOk ? 'Ollama running' : 'Ollama offline'} />
      </div>

      {/* Session */}
      <div className="sidebar-section">
        <div className="section-label">Session</div>
        <div className="session-id">
          <span className="session-id-text" title={threadId}>{threadId.slice(0, 8)}…</span>
        </div>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button className="btn-outline" onClick={onNewSession} style={{ flex: 1 }}>
            + New
          </button>
          <button
            className="btn-outline"
            style={{ flex: 1 }}
            onClick={() => { setShowSessions(v => !v); if (!showSessions) loadSessions() }}
          >
            History
          </button>
        </div>

        {showSessions && (
          <div className="session-history">
            {sessions.length === 0 ? (
              <p className="ltm-empty">No past sessions found.</p>
            ) : (
              sessions.map(s => (
                <button
                  key={s.thread_id}
                  className={`session-item ${s.thread_id === threadId ? 'session-item-active' : ''}`}
                  onClick={() => { onLoadSession(s.thread_id); setShowSessions(false) }}
                  title={s.thread_id}
                >
                  <span className="session-item-id">{s.thread_id.slice(0, 12)}…</span>
                  <span className="session-item-label">Load</span>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="sidebar-section">
        <div className="section-label">Knowledge base</div>
        <div className="stats-row">
          <StatPill label="Documents" value={chromaFacts} color="#0ea5e9" />
          <StatPill label="Posts" value={chromaPosts} color="#7c6af7" />
          <StatPill label="LTM" value={ltmCount} color="#22c55e" />
        </div>
      </div>

      {/* LTM viewer */}
      <div className="sidebar-section ltm-section">
        <button className="ltm-toggle" onClick={onToggleLtm}>
          <span>🧠 Long-term memory</span>
          <span className="chevron">{showLtm ? '▲' : '▼'}</span>
        </button>

        {showLtm && (
          <div className="ltm-panel">
            {ltmFacts.length === 0 ? (
              <p className="ltm-empty">
                No facts yet — extracted automatically as you chat.
              </p>
            ) : (
              <>
                <input
                  className="ltm-search"
                  placeholder="Filter facts…"
                  value={ltmFilter}
                  onChange={e => setLtmFilter(e.target.value)}
                />
                <div className="ltm-list">
                  {filteredFacts.map(f => (
                    <div key={f.key} className="ltm-fact">
                      <div className="ltm-fact-meta">
                        <span
                          className="ltm-cat"
                          style={{ background: (CAT_COLORS[f.category] || '#666') + '22',
                                   color: CAT_COLORS[f.category] || '#aaa' }}
                        >
                          {f.category}
                        </span>
                        <span className="ltm-conf" title={`Confidence ${f.confidence}/3`}>
                          {CONF_ICON[f.confidence] || '○'}
                        </span>
                      </div>
                      <div className="ltm-fact-text">{f.fact}</div>
                      <button
                        className="ltm-delete"
                        onClick={() => onDeleteFact(f.key)}
                        title="Delete fact"
                      >×</button>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <span>Ollama · LangGraph · ChromaDB · PostgreSQL</span>
      </div>
    </aside>
  )
}

function StatPill({ label, value, color }) {
  return (
    <div className="stat-pill">
      <span className="stat-value" style={{ color }}>{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  )
}
