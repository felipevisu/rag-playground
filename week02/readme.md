# Métricas 

Você tem 180 chunks no Postgres. Alguém faz uma pergunta. Sua busca semântica devolve os 5 chunks mais parecidos com a pergunta.

Pergunta que importa: **a busca devolveu os chunks certos?**

Para responder isso, você precisa saber de antemão quais eram os certos. É exatamente isso que o `questions.json` faz:

```json
{
  "question": "O que o PL 1502/2026 define como racismo ambiental?",
  "expected_chunks": [ { "chunk_index": 1, ... } ]
}
```

Tradução: "para essa pergunta, a resposta está no chunk 1 do PL_1502. Se sua busca não trouxer esse chunk, ela errou."

Isso é o gabarito (ground truth). Temos hoje 38 perguntas com gabarito.

O que as métricas são: São só formas diferentes de transformar "acertou/errou" em um número, para podermos comparar duas versões do RAG.

Exemplo prático de por que isso importa. Amanhã trocamos `all-MiniLM-L6-v2` por outro modelo de embedding. Ficou melhor ou pior? Com métrica podemos avaliar:
MiniLM      → 0.72
modelo novo → 0.85

Por que 5 métricas e não 1: porque "acertou" tem significados diferentes, e cada métrica responde uma pergunta diferente.

| Métrica | Pergunta que ela responde |
|---|---|
| Hit Rate | Trouxe **pelo menos um** chunk certo? |
| Recall | Trouxe **quantos %** dos chunks certos? |
| Precision | Dos que trouxe, **quantos %** prestavam? |
| MRR | O chunk certo veio **no topo** ou lá embaixo? |
| nDCG | Combina "achou" + "posição" numa nota só |

Todas rodam sobre a mesma busca. São 5 leituras do mesmo resultado.

## Ferramenta

<img src="/screenshot3.png" alt="" />

## Exemplo 1 - Resposta que precisa de apenas um chunk

### Cenário

Documento: `PL_1502_2026`

```json
{
  "type": "single_chunk",
  "question": "O que o PL 1502/2026 define como racismo ambiental?",
  "expected_answer": "Práticas, políticas ou omissões que expõem de forma diferencial e desproporcional grupos raciais, étnicos ou territoriais a riscos e danos ambientais e climáticos, bem como a negação ou restrição de acesso a mecanismos de proteção, reparação e participação (art. 2º, II).",
  "expected_chunks": [
    {
      "chunk_id": 2,
      "chunk_index": 1,
      "pages": [
        2
      ],
      "headings": [
        "Projeto de Lei Ordinária Nº                   , DE 2026. (Do Sr. Rubens Pereira Júnior)"
      ],
      "text_preview": "Parágrafo único. A Política prevista neste artigo reger-se-á pelos princípios da precaução, prevenção, equidade intergeracional, reconhecimento e proteção de direitos territoriais, participação e controle social, reparaç"
    }
  ]
}
```

<img src="/screenshot1.png" alt="results" />

### Analizando os resultados

**Hit Rate = 1**
O teste esperava pelo menos um chunk correto que era o chunk de id=2 e index=1 fosse retornado na resposta e isto aconteceu, portanto a pontuação aqui foi 1

**Recall = 1**
Quantos dos chunks esperados vieram na lista de chunks? O testera esperava 1 e este 1 estava nos resultados, portando pontuação foi de 1.

**Previsions = 0.2**
Dos 5 chunks retornados apenas um era essencial para a resposta, portanto 1/5 = 0.2.

**RR = 0.333**
Em qual posição da lista estava o melhor resultado esperado? Como o chunk estava na posição 3, 1/3 = 0.33.

**nDCG = 0.5**
Combina se acho o chunk, e a posição em que ele estava em uma única nota.
DCG 0.500 = 1/log₂(3+1)
IDCG 1.000 = ideal com 1 esperado(s) nos ranks 1..1

## Exemplo 2 - Resposta que precisa de multiplos chunks

### Cenário

Documento: `PL_1502_2026`

```json
{
  "type": "multi_chunk",
  "question": "Quais são as fontes de recursos do Fundo Nacional de Adaptação e Reparação Climática (FNARC) e como é composto o seu Comitê Gestor?",
  "expected_answer": "O FNARC é criado no art. 3º como fundo especial. As fontes estão no § 1º: dotações da LOA e créditos adicionais, transferências voluntárias e de fundos/agências multilaterais, receitas vinculadas por lei específica, contribuições de agentes responsáveis por emissões e degradação, doações/legados/convênios e recursos de instrumentos de precificação de carbono. O Comitê Gestor (§ 3º) inclui representantes de Ministérios da área ambiental e econômica, do órgão federal de políticas indigenistas e étnico-raciais, do Ministério da Cidadania, de estados e municípios, de povos indígenas, quilombolas, comunidades tradicionais e movimentos sociais, e da sociedade civil e especialistas independentes.",
  "expected_chunks": [
    {
      "chunk_id": 3,
      "chunk_index": 2,
      "pages": [
        3
      ],
      "headings": [
        "Projeto de Lei Ordinária Nº                   , DE 2026. (Do Sr. Rubens Pereira Júnior)"
      ],
      "text_preview": "V - reassentamento assistido: conjunto de medidas de relocação, proteção social, reparação e reconstrução de meios de vida, com garantia de moradia digna, direitos territoriais e recuperação de renda, decorrentes de impa"
    },
    {
      "chunk_id": 4,
      "chunk_index": 3,
      "pages": [
        3,
        4
      ],
      "headings": [
        "Projeto de Lei Ordinária Nº                   , DE 2026. (Do Sr. Rubens Pereira Júnior)"
      ],
      "text_preview": "- § 1º São fontes de recursos do FNARC, sem prejuízo de outras previstas em lei: - a) dotações consignadas na Lei Orçamentária Anual (LOA) e créditos adicionais; - b) transferências voluntárias e transferências de fundos"
    },
    {
      "chunk_id": 5,
      "chunk_index": 4,
      "pages": [
        4
      ],
      "headings": [
        "Projeto de Lei Ordinária Nº                   , DE 2026. (Do Sr. Rubens Pereira Júnior)"
      ],
      "text_preview": "- § 3º O FNARC será gerido por Comitê Gestor, órgão colegiado deliberativo, cuja composição, atribuições, quórum, mandato e regras de funcionamento serão definidos em regulamento, assegurada, no mínimo, a participação: -"
    }
  ]
}
```

<img src="/screenshot2.png" alt="results" />

### Analizando os resultados

**Hit Rate = 1**
O teste esperava pelo menos um chunk correto que era o chunk de id 3, 4 ou 5, fossem retornados na resposta e isto aconteceu, portanto a pontuação aqui foi 1

**Recall = 0.667**
Quantos dos chunks esperados vieram na lista de chunks? O testera esperava 3 e apenas 2 estavam nos resultados, portando pontuação foi de 2/3 = 0.667.

**Previsions = 0.4**
Dos 5 chunks retornados apenas dois eram essenciais para a resposta, portanto 2/5 = 0.4.

**RR = 1**
Em qual posição da lista estava o melhor resultado esperado? Como o chunk estava na posição 1, 1/1 = 1.

**nDCG = 0.765**
Combina se acho o chunk, e a posição em que ele estava em uma única nota.
DCG 1.631 = 1/log₂(1+1) + 1/log₂(2+1)
IDCG 2.131 = ideal com 3 esperado(s) nos ranks 1..3