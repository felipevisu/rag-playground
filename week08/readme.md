# week08 — multi-query

**Antes:** a pergunta é buscada como veio. **Agora:** ela vira 5 versões, cada uma busca, e os rankings se juntam (RRF).

```
"Que tipos de denúncia o 'Disque Animal' recebe e a quais órgãos elas são encaminhadas?"
  ├─ "Categorias de denúncias atendidas pelo Disque Animal e órgãos responsáveis"
  ├─ "Espécies de denúncias recebidas pelo Disque Animal e destinatários administrativos"
  ├─ "Modalidades de denúncias do Disque Animal e instituições para as quais são repassadas"
  └─ "Disque Animal - tipos de denúncias e encaminhamento a órgãos competentes"
```

Mais formulações = mais vocabulário. Um chunk que aparece bem em várias sobe.

```
GET /api/buckets/<id>/search?q=...&k=5&variants=4
GET /api/buckets/<id>/search?q=...&k=5&variants=4&rerank=mminilm-rerank   # rerank lê a pergunta original
```

As reescritas foram geradas **uma vez** pelo Claude (`claude-haiku-4-5`) e
estão em `services/chunks-api/rewrites.json`. Buscar custa zero tokens e o eval
é reprodutível. Pergunta fora do arquivo → 404.

## Rodando

```bash
docker compose up --build        # ui :3000 · api :8000/docs
python services/chunks-api/gen_rewrites.py   # só se as perguntas mudarem (ANTHROPIC_API_KEY no .env)
```

No eval (:8080), seletor **variantes** (1–5).

**Próximo:** um chunk não sabe de que lei ele é. → [week09](../week09)
