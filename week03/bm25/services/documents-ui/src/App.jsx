import { useState, useEffect, useCallback } from 'react'
import './App.css'

// ── API ──────────────────────────────────────────────────────────

async function apiFetch(url, options) {
  const r = await fetch(url, options)
  if (!r.ok) {
    const body = await r.text()
    let detail = body
    try { detail = JSON.parse(body).detail ?? body } catch (_) {}
    throw new Error(detail || `HTTP ${r.status}`)
  }
  return r.json()
}

// Read-only. The corpus is built in dataset/ and loaded into Postgres by the
// loader service — nothing is uploaded or deleted from here any more.
const api = {
  documents: () => apiFetch('/api/documents'),
  chunks: id => apiFetch(`/api/documents/${id}/chunks`),
}

// ── Sub-components ───────────────────────────────────────────────

function ThemeToggle() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light')

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('theme', theme)
  }, [theme])

  return (
    <button
      className="theme-btn"
      title={theme === 'light' ? 'Switch to dark' : 'Switch to light'}
      onClick={() => setTheme(t => (t === 'light' ? 'dark' : 'light'))}
    >
      {theme === 'light' ? (
        <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
          <path d="M11.5 8.6A5 5 0 015.4 2.5a5 5 0 106.1 6.1z" fill="currentColor"/>
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <circle cx="7" cy="7" r="3" fill="currentColor"/>
          <path d="M7 .5v1.8M7 11.7v1.8M.5 7h1.8M11.7 7h1.8M2.4 2.4l1.3 1.3M10.3 10.3l1.3 1.3M11.6 2.4l-1.3 1.3M3.7 10.3l-1.3 1.3"
                stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
        </svg>
      )}
    </button>
  )
}

function Spinner({ size = 14 }) {
  return <span className="spinner" style={{ width: size, height: size }} />
}

function DocItem({ doc, selected, onSelect }) {
  return (
    <div
      className={`doc-item ${selected ? 'active' : ''}`}
      onClick={() => onSelect(doc)}
    >
      <div className="doc-icon">
          <svg width="14" height="16" viewBox="0 0 14 16" fill="none">
            <path d="M2 0h7l5 5v11H2V0z" fill="none" stroke="currentColor" strokeWidth="1.2"/>
            <path d="M9 0v5h5" fill="none" stroke="currentColor" strokeWidth="1.2"/>
            <path d="M4 8h6M4 11h4" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.6"/>
          </svg>
      </div>
      <div className="doc-body">
        <div className="doc-name" title={doc.filename}>
          {doc.filename.replace(/\.pdf$/i, '')}
        </div>
        <div className="doc-sub">
          <span className="badge">{doc.chunk_count} chunks</span>
          <span className="doc-date">{new Date(doc.loaded_at).toLocaleDateString()}</span>
        </div>
      </div>
    </div>
  )
}

function EmbeddingPanel({ embedding }) {
  const [open, setOpen] = useState(false)
  if (!Array.isArray(embedding) || embedding.length === 0) {
    return (
      <div className="embed-panel">
        <div className="embed-head muted">
          <span>No embedding</span>
        </div>
      </div>
    )
  }
  const dim = embedding.length
  const preview = embedding.slice(0, 6)
  const fmt = n => n.toFixed(4)
  const min = Math.min(...embedding)
  const max = Math.max(...embedding)

  return (
    <div className={`embed-panel ${open ? 'open' : ''}`}>
      <button className="embed-head" onClick={() => setOpen(o => !o)}>
        <span className="embed-caret">{open ? '▾' : '▸'}</span>
        <span className="embed-label">embedding</span>
        <span className="embed-dim">{dim}d</span>
        {!open && (
          <span className="embed-preview">
            [{preview.map(fmt).join(', ')}, …]
          </span>
        )}
        <span className="embed-range">
          min {fmt(min)} · max {fmt(max)}
        </span>
      </button>
      {open && (
        <div className="embed-grid">
          {embedding.map((v, i) => (
            <div key={i} className="embed-cell" title={`dim ${i}: ${v}`}>
              <span className="embed-idx">{i}</span>
              <span
                className="embed-val"
                style={{
                  color: v >= 0 ? 'var(--accent)' : 'var(--err)',
                  opacity: 0.35 + Math.min(Math.abs(v), 1) * 0.65,
                }}
              >
                {fmt(v)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ChunkId({ id }) {
  const [copied, setCopied] = useState(false)

  function copy() {
    navigator.clipboard?.writeText(String(id))
    setCopied(true)
    setTimeout(() => setCopied(false), 1200)
  }

  return (
    <button
      className={`chunk-id ${copied ? 'copied' : ''}`}
      onClick={copy}
      title="Copy chunk id"
    >
      <span className="chunk-id-key">id</span>
      {copied ? 'copied' : id}
    </button>
  )
}

function ChunkCard({ chunk }) {
  const headings = Array.isArray(chunk.headings) ? chunk.headings : []
  const pages = Array.isArray(chunk.page_numbers) ? chunk.page_numbers : []

  return (
    <div className="chunk-card">
      <div className="chunk-meta">
        <span className="chunk-num">#{chunk.chunk_index + 1}</span>
        <ChunkId id={chunk.id} />
        {headings.length > 0 && (
          <span className="chunk-path">
            {headings.map((h, i) => (
              <span key={i}>
                {i > 0 && <span className="crumb-sep">›</span>}
                <span>{h}</span>
              </span>
            ))}
          </span>
        )}
        {pages.length > 0 && (
          <span className="page-tag">p.{pages.join(', ')}</span>
        )}
      </div>
      <p className="chunk-text">{chunk.text}</p>
      <EmbeddingPanel embedding={chunk.embedding} />
    </div>
  )
}

// ── App ──────────────────────────────────────────────────────────

export default function App() {
  const [docs, setDocs] = useState([])
  const [selected, setSelected] = useState(null)
  const [chunks, setChunks] = useState([])
  const [loadingDocs, setLoadingDocs] = useState(true)
  const [loadingChunks, setLoadingChunks] = useState(false)
  const [toast, setToast] = useState(null)

  const showToast = useCallback((msg, type = 'error') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }, [])

  // The corpus is frozen — loaded once at boot, never changed from here. One
  // fetch, no polling.
  const loadDocs = useCallback(async () => {
    try {
      setDocs(await api.documents())
    } catch (e) {
      showToast(e.message)
    } finally {
      setLoadingDocs(false)
    }
  }, [showToast])

  useEffect(() => { loadDocs() }, [loadDocs])

  const selectDoc = useCallback(async doc => {
    setSelected(doc)
    setChunks([])
    setLoadingChunks(true)
    try {
      setChunks(await api.chunks(doc.id))
    } catch (e) {
      showToast(e.message)
    } finally {
      setLoadingChunks(false)
    }
  }, [showToast])

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="brand">
            <div className="brand-dot" />
            <span>RAG Manager</span>
          </div>
          <span className="doc-tally">{docs.length}</span>
          <ThemeToggle />
        </div>

        <div className="doc-list-header">Documents</div>
        <div className="doc-list">
          {loadingDocs ? (
            <div className="state-msg"><Spinner /> Loading…</div>
          ) : docs.length === 0 ? (
            <div className="state-msg">
              No documents. Build the dataset, then restart: <code>cd dataset &amp;&amp; docker compose run --rm build</code>
            </div>
          ) : (
            docs.map(doc => (
              <DocItem
                key={doc.id}
                doc={doc}
                selected={selected?.id === doc.id}
                onSelect={selectDoc}
              />
            ))
          )}
        </div>
      </aside>

      <main className="main">
        {!selected ? (
          <div className="empty-state">
            <div className="empty-icon">
              <svg width="48" height="56" viewBox="0 0 48 56" fill="none">
                <path d="M4 0h28l16 16v40H4V0z" fill="none" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M32 0v16h16" fill="none" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M12 24h24M12 32h18M12 40h20" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <p className="empty-title">Select a document</p>
            <p className="empty-sub">Choose one from the sidebar</p>
          </div>
        ) : (
          <>
            <div className="main-header">
              <div className="main-doc-info">
                <h2 className="main-doc-name">{selected.filename}</h2>
                <span className="main-doc-sub">
                  {loadingChunks ? 'Loading…' : `${chunks.length} chunks`}
                </span>
              </div>
            </div>

            <div className="chunks-scroll">
              {loadingChunks ? (
                <div className="state-msg"><Spinner /> Loading chunks…</div>
              ) : (
                chunks.map(chunk => <ChunkCard key={chunk.id} chunk={chunk} />)
              )}
            </div>
          </>
        )}
      </main>

      {toast && (
        <div className={`toast ${toast.type}`}>{toast.msg}</div>
      )}
    </div>
  )
}
