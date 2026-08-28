"""BM25 do zero, em Python puro. Sem dependências.

    python bm25.py
"""

import math
import re
from collections import Counter, defaultdict

CHUNKS = [
    "O serviço de backup executa automaticamente todos os dias às 3h da manhã. "
    "Cada execução cria um job com identificador numérico único.",

    "Quando um job falha, o código de erro fica no log. O erro ERR_DISK_FULL "
    "indica que o volume de destino ficou sem espaço em disco durante a cópia.",

    "Para restaurar um backup, use o comando restore informando o identificador "
    "do job. A restauração sobrescreve os arquivos existentes no destino.",

    "O erro ERR_AUTH_DENIED aparece quando as credenciais do serviço expiraram. "
    "Renove o token de acesso no painel e execute o job novamente.",

    "O espaço em disco do servidor deve ser monitorado semanalmente. "
    "Recomenda-se manter pelo menos 20% do volume livre para os backups.",

    "Para desativar a rotina noturna, edite o agendamento no painel e remova a "
    "janela de execução. Nenhuma cópia será criada até reativar o agendamento.",
]


# ===========================================================================
# 1. ANALYZER  —  texto vira tokens
# ===========================================================================
STOPWORDS = frozenset("""
a ao aos as às até com como da das de dela dele delas deles depois do dos e é
ela elas ele eles em entre era eram essa essas esse esses esta estas este estes
eu foi foram há isso isto já lhe lhes mais mas me mesmo meu meus muito na nas
nem no nos nós nossa nosso num numa o os ou para pela pelas pelo pelos por qual
quando que quem se sem ser será seu seus só sua suas também te tem têm ter teu
teus tu tua tuas um uma umas uns você vocês
""".split())

SUFIXOS = tuple(sorted({
    "amento", "imento", "ador", "ando", "endo", "indo", "aram", "aria",
    "adas", "ados", "ava", "ada", "ado", "ida", "ido",
    "ar", "er", "ir", "ou", "am", "em", "es", "as", "os", "a", "e", "o", "s",
}, key=len, reverse=True))

TAMANHO_MINIMO = 4

USAR_STOPWORDS = True
USAR_STEMMING = True


def stem(palavra):
    """Corta um sufixo flexional, se sobrar radical suficiente.
    Identificadores (err_disk_full, 3h, 20) passam intactos."""
    if not palavra.isalpha() or len(palavra) <= TAMANHO_MINIMO:
        return palavra
    for sufixo in SUFIXOS:
        if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= TAMANHO_MINIMO:
            return palavra[:-len(sufixo)]
    return palavra


def tokenizar(texto):
    """O MESMO analyzer roda na indexação e na busca. Se divergirem, nada casa
    — é o bug nº 1 de quem escreve busca léxica na mão."""
    tokens = []
    for token in re.findall(r"\w+", texto.lower()):
        if USAR_STOPWORDS and token in STOPWORDS:
            continue
        if USAR_STEMMING:
            token = stem(token)
        tokens.append(token)
    return tokens


# ===========================================================================
# 2. ÍNDICE INVERTIDO
#
# Em vez de "documento -> quais termos tem", guardamos "termo -> em quais
# documentos está, e quantas vezes". Assim uma busca por 5 termos visita as
# 5 postings lists desses termos, e NUNCA toca nos outros documentos.
#
#   POSTINGS["disc"] == {1: 1, 4: 1}   -> 'disc' está nos docs 1 e 4, 1x cada
# ===========================================================================
POSTINGS = {}     # termo -> {doc_id: frequência}
DOC_LEN = []      # doc_id -> tamanho em tokens
N = 0             # número de documentos
AVGDL = 0.0       # tamanho médio


def reindexar():
    """(Re)constrói o índice com o analyzer atual. Rodado uma vez, na carga."""
    global POSTINGS, DOC_LEN, N, AVGDL

    postings = defaultdict(dict)
    DOC_LEN = []

    for doc_id, chunk in enumerate(CHUNKS):
        tokens = tokenizar(chunk)
        DOC_LEN.append(len(tokens))
        for termo, freq in Counter(tokens).items():
            postings[termo][doc_id] = freq

    POSTINGS = dict(postings)
    N = len(DOC_LEN)
    AVGDL = sum(DOC_LEN) / N


def tf(termo, doc_id):
    """Quantas vezes o termo aparece NAQUELE documento."""
    return POSTINGS.get(termo, {}).get(doc_id, 0)


def df(termo):
    """Em quantos documentos o termo aparece. Agora é O(1): é o tamanho da
    postings list, que já está pronta."""
    return len(POSTINGS.get(termo, ()))


def idf(termo):
    """idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

    Variante do Lucene. O '1 +' de fora garante idf positivo — sem ele, um
    termo presente em mais da metade do corpus teria idf NEGATIVO e conter a
    palavra buscada REDUZIRIA o score do documento."""
    d = df(termo)
    return math.log(1 + (N - d + 0.5) / (d + 0.5))


reindexar()


# ===========================================================================
# 3. RANQUEAMENTO
#
#                              f(t,D) · (k1 + 1)
#     score(D,Q) = Σ  IDF(t) · ─────────────────────────────────
#                 t∈Q          f(t,D) + k1 · (1 - b + b·|D|/avgdl)
# ===========================================================================
K1 = 1.5    # saturação: 0 ignora a frequência; alto = quase linear
B = 0.75    # normalização por tamanho: 0 desliga; 1 é total


def contribuicao(termo, doc_id, freq):
    """A parcela que UM termo contribui ao score de UM documento.
    Devolve (contribuição, saturação) — a segunda só para o explain."""
    norma = K1 * (1 - B + B * DOC_LEN[doc_id] / AVGDL)
    saturacao = (freq * (K1 + 1)) / (freq + norma)
    return idf(termo) * saturacao, saturacao


def buscar(query, top_k=None):
    """Percorre as postings lists dos termos da query, acumulando score por
    documento. Isto é 'term-at-a-time': o laço externo é sobre TERMOS, não
    sobre documentos. Documento que não contém nenhum termo nunca é visitado.

    Devolve [(score, doc_id, partes), ...] ordenado do melhor para o pior."""
    termos = tokenizar(query)

    scores = defaultdict(float)
    partes = defaultdict(list)

    for termo in dict.fromkeys(termos):          # dedup, na ordem da query
        postings = POSTINGS.get(termo)
        if not postings:                         # termo não existe no corpus
            continue
        for doc_id, freq in postings.items():    # só quem TEM o termo
            valor, saturacao = contribuicao(termo, doc_id, freq)
            scores[doc_id] += valor
            partes[doc_id].append((termo, freq, idf(termo), saturacao, valor))

    resultados = [(s, d, partes[d]) for d, s in scores.items()]
    # score desc; doc_id como desempate, para a saída ser determinística
    resultados.sort(key=lambda r: (-r[0], r[1]))
    return resultados if top_k is None else resultados[:top_k]


def candidatos(query):
    """Os documentos que a busca vai realmente visitar."""
    alvos = set()
    for termo in tokenizar(query):
        alvos.update(POSTINGS.get(termo, {}))
    return alvos


# ===========================================================================
# 4. SAÍDA
# ===========================================================================
def imprimir_ranking(query):
    resultados = buscar(query)
    visitados = candidatos(query)
    maior = max((s for s, _, _ in resultados), default=1.0) or 1.0

    print(f'query : "{query}"')
    print(f"tokens: {tokenizar(query)}")
    print(f"visita: {len(visitados)} de {N} documentos "
          f"(os outros {N - len(visitados)} têm score 0 por construção)\n")

    for posicao, (score, doc_id, _) in enumerate(resultados, 1):
        barra = "#" * round(20 * score / maior)
        print(f"  {posicao}. chunk {doc_id + 1}  {score:7.4f}  {barra:<20} "
              f"{CHUNKS[doc_id][:44]}...")

    for doc_id in sorted(set(range(N)) - visitados):
        print(f"  -. chunk {doc_id + 1}   0.0000  {'':<20} "
              f"{CHUNKS[doc_id][:44]}...")


def explicar(query, quantos=2):
    print(f'explain: "{query}"')
    print(f"N={N}  AVGDL={AVGDL:.1f}  K1={K1}  B={B}\n")
    for score, doc_id, partes in buscar(query, top_k=quantos):
        print(f"  chunk {doc_id + 1}  ({DOC_LEN[doc_id]} tokens)  score = {score:.4f}")
        for termo, freq, peso, saturacao, valor in sorted(partes, key=lambda p: -p[4]):
            print(f"      {termo:<10} tf={freq}  idf={peso:.4f}"
                  f"  x sat={saturacao:.4f}  = {valor:+.4f}")
        print()


if __name__ == "__main__":
    print(f"índice: {N} documentos, {len(POSTINGS)} termos, AVGDL={AVGDL:.1f}")
    print(f'POSTINGS["disc"]   = {POSTINGS.get("disc")}')
    print(f'POSTINGS["erro"]   = {POSTINGS.get("erro")}')
    print(f'POSTINGS["backup"] = {POSTINGS.get("backup")}')

    print("\n" + "=" * 74)
    explicar("por que o job falhou com erro de disco cheio")

    print("=" * 74)
    imprimir_ranking("por que o job falhou com erro de disco cheio")

    print("\n" + "=" * 74)
    print("A resposta certa é o chunk 6 — e ele nem é visitado:\n")
    imprimir_ranking("como cancelar o backup automático da madrugada")

    print("\n" + "=" * 74)
    print("Mesma intenção, com as palavras do documento:\n")
    imprimir_ranking("desativar agendamento noturno")


