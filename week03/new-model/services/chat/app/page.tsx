'use client'

import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const HISTORY_LIMIT = 10

interface Chunk {
  id: number
  filename: string
  text: string
  page_numbers: number[] | null
  similarity: number
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Chunk[]
  error?: boolean
}

function tightenLists(md: string): string {
  return md
    .replace(/^([ \t]*[-*+].*)\n\n(?=[ \t]*[-*+])/gm, '$1\n')   // blank between bullet items
    .replace(/^([ \t]*\d+\..*)\n\n(?=[ \t]*\d+\.)/gm, '$1\n')   // blank between numbered items
    .replace(/([:：])\n\n(?=[ \t]*[-*+])/gm, '$1\n')             // blank after colon before list
}

function Citations({ citations }: { citations: Chunk[] }) {
  const [open, setOpen] = useState(false)
  if (!citations.length) return null
  return (
    <div className="citations">
      <button className="citations-head" onClick={() => setOpen(o => !o)}>
        <span className="citations-caret">{open ? '▾' : '▸'}</span>
        <span>Sources</span>
        <span className="citations-count">{citations.length}</span>
      </button>
      {open && (
        <div className="citations-list">
          {citations.map(c => (
            <div key={c.id} className="citation">
              <div className="citation-meta">
                <span className="citation-file">{c.filename}</span>
                {c.page_numbers?.length ? (
                  <span className="citation-page">p.{c.page_numbers.join(', ')}</span>
                ) : null}
                <span className="citation-sim">{(c.similarity * 100).toFixed(1)}%</span>
              </div>
              <div className="citation-text">{c.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ChatMessage({ msg }: { msg: Message }) {
  const isAssistant = msg.role === 'assistant'
  return (
    <div className={`msg ${msg.role}${msg.error ? ' error' : ''}`}>
      <span className="msg-role">{msg.role === 'user' ? 'You' : 'Claude'}</span>
      <div className="msg-bubble">
        {isAssistant && !msg.error ? (
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {tightenLists(msg.content)}
            </ReactMarkdown>
          </div>
        ) : (
          msg.content
        )}
      </div>
      {isAssistant && <Citations citations={msg.citations ?? []} />}
    </div>
  )
}

const STORAGE_KEY = 'rag-chat-history'

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>(() => {
    if (typeof window === 'undefined') return []
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    } catch { return [] }
  })
  const [input, setInput] = useState('')
  const [pending, setPending] = useState(false)
  const [model, setModel] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, pending])

  function clearHistory() {
    setMessages([])
    localStorage.removeItem(STORAGE_KEY)
  }

  async function send() {
    const text = input.trim()
    if (!text || pending) return

    const userMsg: Message = { role: 'user', content: text }
    const next = [...messages, userMsg]
    setMessages(next)
    setInput('')
    setPending(true)

    const history = messages
      .slice(-HISTORY_LIMIT * 2)
      .map(m => ({ role: m.role, content: m.content }))

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history, k: 5 }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }))
        throw new Error(err.error ?? `HTTP ${res.status}`)
      }
      const data = await res.json()
      if (data.model) setModel(data.model)
setMessages([...next, { role: 'assistant', content: data.answer, citations: data.citations ?? [] }])
    } catch (e) {
      setMessages([...next, { role: 'assistant', content: (e as Error).message, error: true, citations: [] }])
    } finally {
      setPending(false)
    }
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="chat">
      <header className="chat-header">
        <div className="brand">
          <span className="brand-dot" />
          <span>RAG Chat</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {model && <span className="model-tag">{model}</span>}
          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearHistory} title="Clear history">
              Clear
            </button>
          )}
        </div>
      </header>

      <div className="messages" ref={scrollRef}>
        {messages.length === 0 && !pending && (
          <div className="empty">
            <div className="empty-icon">✦</div>
            <p>Ask anything about the indexed documents.</p>
            <p className="hint">e.g. &quot;How do I cancel my subscription?&quot; · &quot;What if my package is stolen?&quot;</p>
          </div>
        )}
        {messages.map((m, i) => <ChatMessage key={i} msg={m} />)}
        {pending && (
          <div className="thinking">
            <span className="dots"><span /><span /><span /></span>
            <span>Retrieving + generating…</span>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={e => { e.preventDefault(); send() }}>
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="Ask a question…"
          disabled={pending}
          rows={1}
          autoFocus
        />
        <button type="submit" disabled={pending || !input.trim()}>Send</button>
      </form>
    </div>
  )
}
