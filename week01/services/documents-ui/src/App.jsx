import { useState, useEffect, useCallback, useRef } from 'react'
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

const api = {
  documents: () => apiFetch('/api/documents'),
  chunks: id => apiFetch(`/api/documents/${id}/chunks`),
  delete: id => apiFetch(`/api/documents/${id}`, { method: 'DELETE' }),
  upload: file => {
    const form = new FormData()
    form.append('file', file)
    return apiFetch('/api/documents/upload', { method: 'POST', body: form })
  },
}

const MINIO_CONSOLE_URL = 'http://localhost:9001'

const IN_PROGRESS_STATUSES = new Set(['pending', 'chunking', 'embedding'])

// ── Sub-components ───────────────────────────────────────────────

function Spinner({ size = 14 }) {
  return <span className="spinner" style={{ width: size, height: size }} />
}

function StatusSteps({ status, errorMessage }) {
  if (status === 'ready') return null

  const steps = [
    {
      label: 'Upload',
      state: 'done',
    },
    {
      label: 'Chunking',
      state: status === 'chunking' ? 'active'
           : ['embedding', 'ready'].includes(status) ? 'done'
           : status === 'error' ? 'error'
           : 'pending',
    },
    {
      label: 'Embedding',
      state: status === 'embedding' ? 'active'
           : status === 'ready' ? 'done'
           : status === 'error' ? 'error'
           : 'pending',
    },
  ]

  return (
    <div className="status-steps">
      {steps.map((step, i) => (
        <span key={step.label}>
          <span className={`step-pill ${step.state}`}>
            <span className="step-dot" />
            {step.label}
          </span>
          {i < steps.length - 1 && <span className="step-sep">›</span>}
        </span>
      ))}
      {status === 'error' && errorMessage && (
        <span className="step-error-msg" title={errorMessage}>Error</span>
      )}
    </div>
  )
}

function UploadArea({ onUpload }) {
  const inputRef = useRef(null)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  async function handleFiles(files) {
    const pdf = Array.from(files).find(f => f.name.toLowerCase().endsWith('.pdf'))
    if (!pdf) return
    setUploading(true)
    try {
      await onUpload(pdf)
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div className="upload-area">
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={e => handleFiles(e.target.files)}
      />
      <button
        className={`upload-btn ${dragOver ? 'drag-over' : ''}`}
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        {uploading ? (
          <><Spinner size={13} /> Uploading…</>
        ) : (
          <>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M7 1v8M4 4l3-3 3 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M1 10v1a2 2 0 002 2h8a2 2 0 002-2v-1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
            </svg>
            Upload PDF
          </>
        )}
      </button>
    </div>
  )
}

function MinioPanel({ autoRefresh }) {
  return (
    <div className="minio-panel">
      <div className="minio-head">
        <div className="minio-title">
          <span className="minio-dot" />
          Storage
        </div>
        {autoRefresh && (
          <span className="live-indicator" title="Auto-refresh">
            <span className="live-dot" /> live
          </span>
        )}
      </div>
      <a href={MINIO_CONSOLE_URL} target="_blank" rel="noreferrer" className="minio-link">
        <span>Open MinIO Console</span>
        <span className="minio-arrow">↗</span>
      </a>
    </div>
  )
}

function DocItem({ doc, selected, onSelect, onDelete }) {
  const [hovered, setHovered] = useState(false)
  const isReady = doc.status === 'ready' || !doc.status
  const isError = doc.status === 'error'

  return (
    <div
      className={`doc-item ${selected ? 'active' : ''} ${isError ? 'doc-error' : ''}`}
      onClick={() => isReady && onSelect(doc)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="doc-icon">
        {!isReady && !isError
          ? <Spinner size={14} />
          : (
          <svg width="14" height="16" viewBox="0 0 14 16" fill="none">
            <path d="M2 0h7l5 5v11H2V0z" fill="#1e2235" stroke={isError ? '#5c1d23' : '#3d4168'} strokeWidth="1.2"/>
            <path d="M9 0v5h5" fill="none" stroke={isError ? '#5c1d23' : '#3d4168'} strokeWidth="1.2"/>
            <path d="M4 8h6M4 11h4" stroke={isError ? '#f87171' : '#4f52e0'} strokeWidth="1" strokeLinecap="round"/>
          </svg>
          )}
      </div>
      <div className="doc-body">
        <div className="doc-name" title={doc.filename}>
          {doc.filename.replace(/\.pdf$/i, '')}
        </div>
        <div className="doc-sub">
          {isReady && <span className="badge">{doc.chunk_count} chunks</span>}
          {!isReady && !isError && <span className="badge badge-processing">{doc.chunk_count} chunks</span>}
          <span className="doc-date">{new Date(doc.processed_at).toLocaleDateString()}</span>
        </div>
        <StatusSteps status={doc.status} errorMessage={doc.error_message} />
      </div>
      {hovered && (
        <button
          className="del-btn"
          title="Delete document"
          onClick={e => { e.stopPropagation(); onDelete(doc) }}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
      )}
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
                  color: v >= 0 ? '#a5b4fc' : '#f87171',
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

function ChunkCard({ chunk }) {
  const headings = Array.isArray(chunk.headings) ? chunk.headings : []
  const pages = Array.isArray(chunk.page_numbers) ? chunk.page_numbers : []

  return (
    <div className="chunk-card">
      <div className="chunk-meta">
        <span className="chunk-num">#{chunk.chunk_index + 1}</span>
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
  const lastDocCountRef = useRef(0)
  const intervalRef = useRef(null)

  const showToast = useCallback((msg, type = 'error') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }, [])

  const loadDocs = useCallback(async (silent = false) => {
    try {
      const next = await api.documents()
      setDocs(prev => {
        if (silent && next.length > lastDocCountRef.current && lastDocCountRef.current > 0) {
          const newDoc = next.find(d => !prev.some(p => p.id === d.id))
          if (newDoc) showToast(`New: ${newDoc.filename}`, 'success')
        }
        lastDocCountRef.current = next.length
        return next
      })
      return next
    } catch (e) {
      if (!silent) showToast(e.message)
      return []
    } finally {
      if (!silent) setLoadingDocs(false)
    }
  }, [showToast])

  const scheduleRefresh = useCallback((docs) => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    const hasInProgress = docs.some(d => IN_PROGRESS_STATUSES.has(d.status))
    const interval = hasInProgress ? 2000 : 5000
    intervalRef.current = setInterval(async () => {
      const next = await loadDocs(true)
      scheduleRefresh(next)
    }, interval)
  }, [loadDocs])

  useEffect(() => {
    loadDocs().then(initial => scheduleRefresh(initial))
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [loadDocs, scheduleRefresh])

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

  const handleDelete = useCallback(async doc => {
    if (!window.confirm(`Delete "${doc.filename}"?\n\nThis removes chunks from the database AND the PDF from MinIO.`)) return
    try {
      await api.delete(doc.id)
      if (selected?.id === doc.id) { setSelected(null); setChunks([]) }
      await loadDocs()
      showToast(`Deleted ${doc.filename}`, 'success')
    } catch (e) {
      showToast(e.message)
    }
  }, [selected, loadDocs, showToast])

  const handleUpload = useCallback(async file => {
    try {
      await api.upload(file)
      const next = await loadDocs()
      scheduleRefresh(next)
      showToast(`Uploaded ${file.name} — processing…`, 'success')
    } catch (e) {
      showToast(e.message)
    }
  }, [loadDocs, scheduleRefresh, showToast])

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="brand">
            <div className="brand-dot" />
            <span>RAG Manager</span>
          </div>
          <span className="doc-tally">{docs.length}</span>
        </div>

        <UploadArea onUpload={handleUpload} />
        <MinioPanel autoRefresh={true} />

        <div className="doc-list-header">Documents</div>
        <div className="doc-list">
          {loadingDocs ? (
            <div className="state-msg"><Spinner /> Loading…</div>
          ) : docs.length === 0 ? (
            <div className="state-msg">No documents yet. Upload a PDF above.</div>
          ) : (
            docs.map(doc => (
              <DocItem
                key={doc.id}
                doc={doc}
                selected={selected?.id === doc.id}
                onSelect={selectDoc}
                onDelete={handleDelete}
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
                <path d="M4 0h28l16 16v40H4V0z" fill="#141824" stroke="#2d3148" strokeWidth="1.5"/>
                <path d="M32 0v16h16" fill="none" stroke="#2d3148" strokeWidth="1.5"/>
                <path d="M12 24h24M12 32h18M12 40h20" stroke="#3d4168" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <p className="empty-title">Select a document</p>
            <p className="empty-sub">Upload a PDF or choose from the sidebar</p>
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
