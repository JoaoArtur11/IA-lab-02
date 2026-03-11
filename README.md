# Transformer Encoder - implementação manual com NumPy

Esse código implementa um encoder Transformer do zero usando só NumPy e Pandas, sem nenhuma biblioteca de deep learning. Foi feito pra entender como funciona por dentro.

---

## O que o código faz

Pega uma frase em português, tokeniza ela na mão, converte pra embeddings e passa por 6 camadas de encoder (self-attention + feed-forward + residual + layer norm), igual o paper "Attention is All You Need".

---

## Requisitos

Você precisa ter Python instalado (testado no 3.10+) e as seguintes bibliotecas:

```
numpy
pandas
```

Se não tiver, instala com:

```bash
pip install numpy pandas
```

---

## Como rodar

1. Clona ou baixa o arquivo `transformer_encoder.py`

2. No terminal, entra na pasta onde o arquivo está:

```bash
cd caminho/para/a/pasta
```

3. Roda com:

```bash
python transformer_encoder.py
```

---

## O que vai aparecer no terminal

O script imprime algumas coisas ao longo da execução:

- o vocabulário em formato de tabela
- os tokens e IDs da frase de entrada
- o shape do tensor X de entrada
- confirmação de cada camada processada
- o shape final do tensor Z
- os primeiros 10 valores do vetor do primeiro token

Exemplo de saída esperada:

```
vocabulario:
    palavra  id
0         o   0
1     banco   1
2  bloqueou   2
3    cartao   3
4       meu   4
5    ontem    5

frase: o banco bloqueou meu cartao ontem
tokens: ['o', 'banco', 'bloqueou', 'meu', 'cartao', 'ontem']
ids: [0, 1, 2, 3, 4, 5]

shape do X: (1, 6, 64)

  camada 1 ok, shape: (1, 6, 64)
  camada 2 ok, shape: (1, 6, 64)
  ...
  camada 6 ok, shape: (1, 6, 64)

shape final: (1, 6, 64)

primeiros 10 valores do token 0:
[ 0.34  -1.12  0.87 ... ]
```

---

## Estrutura do código

```
transformer_encoder.py
│
├── vocabulario e embeddings    → prepara os dados de entrada
├── softmax / relu / layer_norm → funções auxiliares
├── SelfAttention               → mecanismo de atenção (Q, K, V)
├── FeedForward                 → rede densa com ReLU no meio
├── EncoderLayer                → junta atenção + FFN + residual
└── TransformerEncoder          → empilha 6 camadas e roda tudo
```

---

## Observações

- Os pesos são inicializados aleatoriamente, então os valores de saída vão mudar a cada execução — a menos que o `np.random.seed(42)` esteja no topo do arquivo (já está).
- O código não treina nada, só faz o forward pass.
- A frase de entrada tá hardcoded, mas dá pra trocar fácil lá no começo do arquivo.
