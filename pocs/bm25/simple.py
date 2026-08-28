import re
import math
from collections import Counter

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

def tokenizar(texto):
    """Texto -> lista de tokens. Minúsculas, e todo token é uma sequência de
    letras/dígitos/underscore. O resto (pontuação, espaço) é separador."""
    return re.findall(r"\w+", texto.lower())


DOCS = [tokenizar(c) for c in CHUNKS]
TF = [Counter(d) for d in DOCS] 
N = len(DOCS)
AVGDL = sum(len(d) for d in DOCS) / N
VOCABULARIO = sorted({t for d in DOCS for t in d})


def tf(termo, doc_id):
    """Term frequency: quantas vezes o termo aparece NAQUELE documento."""
    return TF[doc_id][termo]


def df(termo):
    """Document frequency: em quantos documentos o termo aparece.

    Conta DOCUMENTOS, não ocorrências. 'erro' aparece 2x no chunk 2, mas
    contribui 1 para o df."""
    return sum(1 for contador in TF if termo in contador)


def idf(termo):
    """Inverse document frequency, na variante que o Lucene usa:

        idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

    Termo raro -> idf alto. Termo em todo lugar -> idf perto de zero.
    O '1 +' de fora garante que o idf nunca fique negativo."""
    d = df(termo)
    return math.log(1 + (N - d + 0.5) / (d + 0.5))


K1 = 1.5
B = 0.75


def score_bm25(termos, doc_id):
    """Pontua um par (query, documento).

    Devolve (score, partes), onde partes é a conta aberta — um item por termo
    que casou. O score é só a soma das contribuições, nada mais."""
    tamanho = len(DOCS[doc_id])

    # Depende só do documento, então é calculado uma vez para todos os termos.
    # tamanho == AVGDL  -> norma == K1        (documento mediano, sem ajuste)
    # tamanho >  AVGDL  -> norma >  K1        (documento longo, penalizado)
    norma = K1 * (1 - B + B * tamanho / AVGDL)

    total = 0.0
    partes = []
    for termo in dict.fromkeys(termos):     # dedup, mantendo a ordem da query
        f = tf(termo, doc_id)
        if f == 0:
            continue                        # termo ausente contribui zero

        saturacao = (f * (K1 + 1)) / (f + norma)
        contribuicao = idf(termo) * saturacao

        total += contribuicao
        partes.append((termo, f, idf(termo), saturacao, contribuicao))

    return total, partes


def buscar(query, top_k=None):
    """Devolve [(score, doc_id, partes), ...] ordenado do melhor para o pior."""
    termos = tokenizar(query)

    resultados = []
    for doc_id in range(N):
        score, partes = score_bm25(termos, doc_id)
        resultados.append((score, doc_id, partes))

    # Ordena por score decrescente. O doc_id é o critério de desempate — sem
    # ele, dois documentos com o mesmo score sairiam em ordem imprevisível.
    resultados.sort(key=lambda r: (-r[0], r[1]))

    return resultados if top_k is None else resultados[:top_k]


def imprimir_ranking(query):
    resultados = buscar(query)
    maior = max(s for s, _, _ in resultados) or 1.0

    print(f'query: "{query}"')
    print(f"tokens: {tokenizar(query)}\n")
    print(f"  {'#':<3} {'chunk':<8} {'score':>7}  {'':<22} termos que casaram")
    print("  " + "-" * 74)

    for posicao, (score, doc_id, partes) in enumerate(resultados, 1):
        barra = "#" * round(20 * score / maior)
        casados = ", ".join(termo for termo, *_ in partes) or "-- nenhum"
        print(f"  {posicao:<3} chunk {doc_id + 1:<2} {score:>7.4f}  {barra:<22} {casados}")

    print()
    for posicao, (_, doc_id, _) in enumerate(resultados, 1):
        print(f"  [{posicao}] {CHUNKS[doc_id][:66]}...")


if __name__ == "__main__":
    imprimir_ranking("por que o job falhou com erro de disco cheio")