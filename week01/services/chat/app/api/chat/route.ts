import Anthropic from '@anthropic-ai/sdk'
import { NextRequest, NextResponse } from 'next/server'

const DOCUMENT_API_URL = process.env.DOCUMENT_API_URL ?? 'http://document-api:8000'
const CLAUDE_MODEL = process.env.CLAUDE_MODEL ?? 'claude-sonnet-4-6'

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })

const SYSTEM_TEMPLATE = `You are a helpful assistant for an e-commerce help center. \
Answer the user's question using ONLY the information in the context chunks below. \
If the answer is not in the context, say you don't have that information.

Cite the source filename in your answer when relevant, like (refund_policy.pdf).

Formatting rules:
- Use tight markdown lists (no blank lines between list items).
- Never add blank lines between bullet points.
- Keep responses concise.

CONTEXT CHUNKS:
{context}`

interface Chunk {
  id: number
  chunk_index: number
  text: string
  headings: string[] | null
  page_numbers: number[] | null
  document_id: number
  filename: string
  similarity: number
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatRequest {
  message: string
  history?: ChatMessage[]
  k?: number
}

function formatContext(chunks: Chunk[]): string {
  return chunks
    .map(r => {
      const bread = r.headings?.length ? r.headings.join(' > ') : ''
      const pageStr = r.page_numbers?.length ? `p.${r.page_numbers.join(',')}` : ''
      const parts = [r.filename, bread, pageStr].filter(Boolean).join(' | ')
      return `[${parts}]\n${r.text}`
    })
    .join('\n\n---\n\n')
}

export async function POST(req: NextRequest) {
  try {
    const body: ChatRequest = await req.json()
    const { message, history = [], k = 5 } = body

    if (!message?.trim()) {
      return NextResponse.json({ error: 'message required' }, { status: 400 })
    }

    let searchRes: Response
    try {
      searchRes = await fetch(
        `${DOCUMENT_API_URL}/api/search?q=${encodeURIComponent(message)}&k=${k}`,
      )
    } catch (e) {
      console.error('document-api unreachable:', e)
      return NextResponse.json(
        { error: `document-api unreachable at ${DOCUMENT_API_URL}` },
        { status: 502 },
      )
    }

    if (!searchRes.ok) {
      const body = await searchRes.text()
      console.error('document-api search error:', searchRes.status, body)
      return NextResponse.json(
        { error: `document-api returned ${searchRes.status}: ${body}` },
        { status: 502 },
      )
    }

    const { results: chunks }: { results: Chunk[] } = await searchRes.json()

    if (!chunks.length) {
      return NextResponse.json({
        answer: 'No documents have been indexed yet. Upload PDFs via MinIO first.',
        citations: [],
      })
    }

    const systemPrompt = SYSTEM_TEMPLATE.replace('{context}', formatContext(chunks))

    const messages: Anthropic.MessageParam[] = [
      ...history
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .filter(m => m.content.trim())
        .map(m => ({ role: m.role, content: m.content })),
      { role: 'user', content: message },
    ]

    const response = await anthropic.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: 1024,
      system: systemPrompt,
      messages,
    })

    const answer = response.content
      .filter(b => b.type === 'text')
      .map(b => (b as Anthropic.TextBlock).text)
      .join('')

    return NextResponse.json({ answer, citations: chunks, model: CLAUDE_MODEL })
  } catch (e) {
    console.error('chat route error:', e)
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
