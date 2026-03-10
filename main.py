import numpy as np
import pandas as pd

np.random.seed(42)


# ---- PARTE 1: preparando os dados ----

# palavras que vao aparecer na frase (vocabulario manual)
vocabulario = {
    "o": 0,
    "banco": 1,
    "bloqueou": 2,
    "cartao": 3,
    "meu": 4,
    "ontem": 5
}

# so pra visualizar melhor
df_vocab = pd.DataFrame(list(vocabulario.items()), columns=["palavra", "id"])
print("vocabulario:")
print(df_vocab)
print()


frase = "o banco bloqueou meu cartao ontem"

# separando as palavras
tokens = frase.split()

# pegando o numero de cada palavra
ids_frase = [vocabulario[t] for t in tokens]

print("frase:", frase)
print("tokens:", tokens)
print("ids:", ids_frase)
print()


# dimensoes que vou usar
batch_size = 1
seq_len = len(ids_frase)
d_model = 64
d_ff = 128
n_layers = 6

# tabela de embeddings (valores aleatorios por enquanto)
vocab_size = len(vocabulario)
embedding_table = np.random.randn(vocab_size, d_model)

# montando o X de entrada
X = embedding_table[ids_frase]
X = np.expand_dims(X, axis=0)  # adiciona dimensao do batch

print("shape do X:", X.shape)  # deve ser (1, 6, 64)
print()


# ---- PARTE 2: funcoes do encoder ----

# softmax numericamente estavél (aprendi que precisa subtrair o max)
def softmax(x, axis=-1):
    x2 = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x2)
    return ex / np.sum(ex, axis=axis, keepdims=True)


# layer norm basica
def layer_norm(x, epsilon=1e-6):
    m = np.mean(x, axis=-1, keepdims=True)
    v = np.var(x, axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + epsilon)


def relu(x):
    return np.maximum(0, x)


# mecanismo de self-attention
class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        self.d_k = d_model

        # pesos Q, K e V inicializados aleatoriamente
        self.WQ = np.random.randn(d_model, d_model)
        self.WK = np.random.randn(d_model, d_model)
        self.WV = np.random.randn(d_model, d_model)

    def forward(self, X):
        # projetando Q, K e V
        Q = X @ self.WQ
        K = X @ self.WK
        V = X @ self.WV

        # transpoe K pra poder multiplicar
        KT = np.transpose(K, (0, 2, 1))

        # calcula os scores de atencao
        scores = Q @ KT

        # divide pela raiz de d_k pra nao explodir os gradientes
        scores = scores / np.sqrt(self.d_k)

        # vira probabilidade com softmax
        pesos = softmax(scores, axis=-1)

        # combina os valores usando os pesos
        saida = pesos @ V

        return saida


class FeedForward:
    def __init__(self, d_model, d_ff):
        # primeira camada linear
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.random.randn(d_ff)

        # segunda camada linear
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.random.randn(d_model)

    def forward(self, X):
        h = X @ self.W1 + self.b1
        h = relu(h)  # nao linear aqui
        out = h @ self.W2 + self.b2
        return out


# uma camada do encoder (atencao + ffn + residual)
class EncoderLayer:
    def __init__(self, d_model, d_ff):
        self.attn = SelfAttention(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, X):
        # self attention
        att = self.attn.forward(X)

        # residual + norma
        X = layer_norm(X + att)

        # feed forward
        ff = self.ffn.forward(X)

        # residual + norma de novo
        X = layer_norm(X + ff)

        return X


# empilha n camadas de encoder
class TransformerEncoder:
    def __init__(self, n_layers, d_model, d_ff):
        self.camadas = [EncoderLayer(d_model, d_ff) for _ in range(n_layers)]

    def forward(self, X):
        for i, camada in enumerate(self.camadas):
            X = camada.forward(X)
            print(f"  camada {i+1} ok, shape: {X.shape}")
        return X


# ---- PARTE 3: rodando o encoder ----

encoder = TransformerEncoder(n_layers=n_layers, d_model=d_model, d_ff=d_ff)
Z = encoder.forward(X)

print()
print("shape final:", Z.shape)
print()

# conferindo os primeiros valores do primeiro token
print("primeiros 10 valores do token 0:")
print(Z[0, 0, :10])