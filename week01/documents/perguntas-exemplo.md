# Perguntas de exemplo para o RAG

Corpus indexado: 19 PDFs de Projetos de Lei de 2026 (tema meio ambiente),
baixados da API de Dados Abertos da Câmara. 180 chunks, embeddings
`all-MiniLM-L6-v2` (384 dim).

## Justiça climática e responsabilidade

PL 752/2026, PL 1502/2026, PL 993/2026, PL 2586/2026

- O que é racismo ambiental segundo o PL 1502?
- Como o Fundo Nacional de Adaptação e Reparação Climática seria financiado?
- O PL 993 altera a Lei 6.938 de que forma? O que passa a entrar na obrigação de indenizar?
- Que projetos tratam de desinformação climática e ambiental?
- Quais populações são consideradas vulneráveis aos impactos das mudanças climáticas?

## Sustação de atos do Executivo

PL 182/2026, PL 253/2026, PL 261/2026, PL 263/2026, PL 271/2026, PL 352/2026, PL 385/2026

- Quais decretos e portarias do MMA estão sendo sustados?
- Quais projetos tratam da Reserva de Desenvolvimento Sustentável Córregos dos Vales do Norte de Minas?
- Qual a justificativa para sustar a classificação do tambaqui?
- O que é o Corredor Ecológico Carajás–Bacajá e por que sustar seu reconhecimento?
- O que a Agenda Regulatória do IBAMA (Portaria 76) estabelece?

## Proteção animal

PL 255/2026, PL 561/2026, PL 770/2026, PL 261/2026

- O que é "animal comunitário" juridicamente?
- Como funcionaria o Disque Animal?
- O que o Programa Nacional de Solidariedade Veterinária prevê distribuir?
- Que argumentos o projeto sobre criação de passeriformes usa para justificar a atividade?

## Resíduos, agrotóxicos e fiscalização

PL 415/2026, PL 742/2026, PL 1928/2026

- Que exceções o PL 742 abre na proibição de pulverização aérea de agrotóxicos?
- Como o PL 415 trata chorume e lodos de ETA/ETE?
- Qual o processo de perdimento de bens usados em infrações à defesa agropecuária?

## Tributário

PL 42/2026

- O que o PL 42 muda nos limites de alíquotas do Imposto Seletivo?

## Comparativas

Puxam chunks de vários documentos — bom teste de retrieval.

- Compare PL 752 e PL 1502: ambos criam Política Nacional de Justiça Climática. Qual a diferença?
- Que projetos alteram a Lei 6.938/1981 e o que cada um muda?
- Quais projetos protegem populações tradicionais ou comunidades extrativistas?

## O que este RAG NÃO responde bem

Limitações reais da arquitetura, não bugs:

- **Contagens** ("quantos projetos falam de X") — retrieval top-k vê só alguns chunks, nunca o corpus inteiro.
- **Ordenação / superlativos** ("qual o projeto mais recente") — sem metadados no índice.
- **Tramitação, situação, autoria, datas** — estão em `pls/metadata.csv`, que não é indexado. Só o texto do PDF entra nos chunks.

## Nota sobre o corpus

`pls/metadados.csv` tem 20 linhas, mas só existem 19 PDFs. Dois projetos
distintos compartilham número/ano:

```
id 2600769 | PL 261/2026 — criação legalizada de passeriformes
id 2617563 | PL 261/2026 — susta o Decreto 12.888
```

`download.py` nomeia o arquivo como `PL_{numero}_{ano}.pdf`; número+ano não é
único entre tipos de proposição (PL vs PDL), então um sobrescreveu o outro.
Corrigir incluindo `siglaTipo` ou o `id` no nome do arquivo.
