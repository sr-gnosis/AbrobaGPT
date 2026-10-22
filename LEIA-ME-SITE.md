# Site local + menu Dev do AbrobaGPT

Extraia este zip **na raiz do projeto** (ao lado de `train.py`, `tui.py` e `main.py`).
Ele só adiciona `server.py`, a pasta `web/` e este arquivo. Nada do que já existe é alterado,
então `python main.py` (TUI), `python chat.py` e `python train.py` continuam funcionando igual.

## Rodar
    python server.py            # http://localhost:8000
    python server.py --open     # abre o navegador junto
    python server.py --port 9000

Não precisa instalar nada além do `requirements.txt` que você já tem (o servidor usa só a
biblioteca padrão; o torch só é carregado quando você conversa ou treina).

## O que tem
- `/` — o site (página inicial).
- `/dev.html` — menu **Dev**: Chat com o modelo (checkpoint, temperatura, máx. tokens, top-k),
  Treino (fonte, dataset, preset de LR, iterações, early stopping, retomar; gráfico de loss e logs ao vivo)
  e Checkpoints (lista e "Usar no chat").

## Segurança
O servidor escuta só em `127.0.0.1` e recusa requisições de outros hosts/sites. Mesmo assim,
não o exponha na internet: o menu Dev pode iniciar treinos e carregar modelos.

## Observações
- O treino roda como processo separado (`train.py`, os mesmos argumentos que a TUI usa) e o botão
  "Parar" o encerra. Se você fechar o servidor com Ctrl+C, o treino em andamento também é encerrado.
- Evite conversar e treinar ao mesmo tempo na mesma GPU: os dois competem por memória.
- Não rode o treino pela TUI e pelo menu Dev ao mesmo tempo.
