# Projeto Integrador III-B: Deep Learning para Classificação de Linhas de Transporte Intermunicipal

## Pontifícia Universidade Católica de Goiás - Escola Politécnica
### Curso: Big Data e Inteligência Artificial


#### Acesso ao notebook via colab: [https://colab.research.google.com/drive/1IgHJiieC1wEktxMvCgHg9J2FvHtgBZh9?authuser=7#scrollTo=Fh9MMPj7iMl-](https://colab.research.google.com/drive/1IgHJiieC1wEktxMvCgHg9J2FvHtgBZh9?usp=sharing)


---

## 📋 Sumário

1. [Visão Geral](#visão-geral)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Desenvolvimento da Solução de Deep Learning](#desenvolvimento-da-solução-de-deep-learning)
   - [Arquitetura do Modelo MLP](#arquitetura-do-modelo-mlp)
   - [Justificativa da Arquitetura](#justificativa-da-arquitetura)
   - [Pipeline de Treinamento e Validação](#pipeline-de-treinamento-e-validação)
   - [Avaliação de Desempenho](#avaliação-de-desempenho)
4. [Como Executar](#como-executar)
5. [Resultados](#resultados)
6. [Referências](#referências)

---

## Visão Geral

Este projeto implementa uma solução de **Deep Learning** utilizando um **Perceptron Multicamadas (MLP)** para classificar automaticamente o tipo de linha de transporte rodoviário intermunicipal de passageiros no Estado de Goiás.

**Concedente:** Agência Goiana de Regulação, Controle e Fiscalização de Serviços Públicos (AGR)

**Autores:**
- Fellipy Bernardes da Silva
- Fernanda Santalucia Bonjardim
- Hellen Almeida de Oliveira
- Hugo de Assis Furtado
- Vitória Gonçalves Lordeiro

---

## Estrutura do Projeto

```
Projeto Integrador 3B/
├── projeto_integrador_3b_deep_learning.ipynb  # Notebook principal
├── README.md                                   # Documentação técnica
├── requirements.txt                            # Dependências do projeto
├── empresas-autorizadas-termos-de-autorizacao-e-precos-de-passagens-.csv
├── reajuste-de-tarifas-de-transporte-rodoviario-intermunicipal-6.csv
├── terminais-rodoviarios-de-passageiros-6.csv
└── venv/                                       # Ambiente virtual Python
```

### Datasets Utilizados

| Dataset | Descrição | Registros |
|---------|-----------|-----------|
| `empresas-autorizadas-*.csv` | Empresas, linhas e itinerários autorizados | 285 |
| `reajuste-de-tarifas-*.csv` | Coeficientes tarifários por tipo de serviço | 102 |
| `terminais-rodoviarios-*.csv` | Terminais rodoviários e situação operacional | 194 |

---

## Desenvolvimento da Solução de Deep Learning

### Classes de Classificação (Variável Alvo)

O modelo classifica as linhas de transporte em 5 categorias:

| Classe | Descrição | Quantidade |
|--------|-----------|------------|
| **Convencional** | Linhas regulares de transporte intermunicipal | 235 (82.5%) |
| **Semiurbano** | Serviços de curta distância entre municípios próximos | 41 (14.4%) |
| **Viagens Semidiretas** | Com poucas paradas intermediárias | 4 (1.4%) |
| **Expresso** | Serviços com menos paradas, mais rápidos | 3 (1.1%) |
| **Viagens Diretas** | Sem paradas intermediárias | 2 (0.7%) |

> **Observação:** O dataset apresenta desbalanceamento significativo, com predominância da classe "Convencional".

<img width="1521" height="533" alt="image" src="https://github.com/user-attachments/assets/c7a28a06-4082-4812-913a-c804b61b7e84" />

<img width="840" height="709" alt="image" src="https://github.com/user-attachments/assets/ef71a257-bf80-4665-bb29-087d0b0d1b8b" />





---

### Arquitetura do Modelo MLP

O modelo implementado é um **Perceptron Multicamadas (MLP)** com a seguinte arquitetura:

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMADA DE ENTRADA                        │
│                    (47 features)                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              CAMADA OCULTA 1 (128 neurônios)                │
│         Dense(128) + ReLU + BatchNorm + Dropout(0.3)        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              CAMADA OCULTA 2 (64 neurônios)                 │
│         Dense(64) + ReLU + BatchNorm + Dropout(0.3)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              CAMADA OCULTA 3 (32 neurônios)                 │
│              Dense(32) + ReLU + Dropout(0.2)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   CAMADA DE SAÍDA                           │
│              Dense(5) + Softmax (5 classes)                 │
└─────────────────────────────────────────────────────────────┘
```

#### Especificações Técnicas

| Componente | Configuração |
|------------|--------------|
| **Entrada** | 47 features (8 numéricas + 39 one-hot encoded) |
| **Camadas Ocultas** | 3 camadas (128 → 64 → 32 neurônios) |
| **Função de Ativação** | ReLU (camadas ocultas), Softmax (saída) |
| **Regularização** | Dropout (0.3, 0.3, 0.2) + BatchNormalization |
| **Inicialização** | He Normal |
| **Otimizador** | Adam (learning rate = 0.001) |
| **Função de Perda** | Categorical Crossentropy |
| **Saída** | 5 classes (probabilidades) |

---

### Justificativa da Arquitetura

#### 1. Escolha do MLP

O Perceptron Multicamadas foi escolhido por ser:
- **Adequado para dados tabulares**: Os dados do projeto são predominantemente categóricos e numéricos estruturados
- **Eficiente para classificação multiclasse**: Capacidade de aprender fronteiras de decisão não-lineares
- **Interpretável**: Arquitetura mais simples comparada a redes convolucionais ou recorrentes
- **Computacionalmente eficiente**: Treinamento rápido para datasets de pequeno/médio porte

#### 2. Número de Camadas e Neurônios

A arquitetura em **pirâmide decrescente** (128 → 64 → 32) foi escolhida para:
- **Camada 1 (128)**: Capturar padrões de alto nível e interações entre features
- **Camada 2 (64)**: Refinar representações intermediárias
- **Camada 3 (32)**: Consolidar informações antes da classificação final

#### 3. Técnicas de Regularização

| Técnica | Justificativa |
|---------|---------------|
| **Dropout (0.2-0.3)** | Previne overfitting desligando neurônios aleatoriamente durante o treino |
| **BatchNormalization** | Estabiliza e acelera o treinamento, permitindo learning rates mais altos |
| **Early Stopping** | Interrompe o treino quando não há melhora na validação (paciência: 15 épocas) |

#### 4. Função de Ativação ReLU

$$f(x) = \max(0, x)$$

Escolhida por:
- Mitigar o problema de vanishing gradients
- Computacionalmente eficiente
- Permite aprendizado de representações esparsas

#### 5. Softmax para Classificação Multiclasse

$$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Converte os logits em probabilidades, onde a soma de todas as classes é 1.

---

### Pipeline de Treinamento e Validação

#### Etapa 1: Engenharia de Features

**Features Numéricas Extraídas:**

| Feature | Descrição |
|---------|-----------|
| `NUM_SEQUENCIAL` | Número sequencial da linha |
| `E_VARIANTE` | Indica se é variante de outra linha (0/1) |
| `QTD_RODOVIAS` | Quantidade de rodovias no itinerário |
| `PASSA_CAPITAL` | Se passa por Goiânia (0/1) |
| `PASSA_ANAPOLIS` | Se passa por Anápolis (0/1) |
| `TAM_ITINERARIO` | Tamanho da descrição do itinerário |
| `TERMINAL_ORIGEM_OPERA` | Se terminal de origem está operando (0/1) |
| `TERMINAL_DESTINO_OPERA` | Se terminal de destino está operando (0/1) |

**Features Categóricas (One-Hot Encoded):**
- `CODIGO_EMPRESA` (27 categorias)
- `PROP_TERMINAL_ORIGEM` (4 categorias)
- `PROP_TERMINAL_DESTINO` (4 categorias)

#### Etapa 2: Pré-processamento

```python
# Normalização das features numéricas
scaler = StandardScaler()
X_scaled[features_numericas] = scaler.fit_transform(X[features_numericas])

# Conversão para float32 (compatibilidade TensorFlow)
X_scaled_array = X_scaled.astype(np.float32).values
```

#### Etapa 3: Divisão dos Dados

| Conjunto | Proporção | Amostras |
|----------|-----------|----------|
| Treino | 60% | 171 |
| Validação | 20% | 57 |
| Teste | 20% | 57 |

**Estratificação:** Mantém proporção das classes em todos os conjuntos.

#### Etapa 4: Balanceamento de Classes

Devido ao desbalanceamento severo, foram aplicados **pesos de classe**:

```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
```

Isso penaliza mais fortemente erros em classes minoritárias durante o treinamento.

#### Etapa 5: Callbacks de Treinamento

| Callback | Configuração | Função |
|----------|--------------|--------|
| **EarlyStopping** | patience=15, restore_best_weights=True | Para o treino se val_loss não melhorar |
| **ReduceLROnPlateau** | factor=0.5, patience=5, min_lr=1e-6 | Reduz learning rate quando estagnado |

#### Etapa 6: Treinamento

```python
historico = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
```

---

### Avaliação de Desempenho

#### Métricas por Conjunto

| Conjunto | Loss | Acurácia |
|----------|------|----------|
| Treino | ~0.30 | ~90% |
| Validação | ~0.45 | ~85% |
| Teste | ~0.50 | ~82% |

> **Nota:** Os valores exatos variam a cada execução devido à natureza estocástica do treinamento.

#### Matriz de Confusão

A matriz de confusão permite visualizar:
- **Diagonal principal**: Classificações corretas
- **Fora da diagonal**: Erros de classificação
- Classes minoritárias podem ter zero predições devido ao desbalanceamento

#### Métricas de Classificação

Para cada classe, são calculados:

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **Precision** | TP / (TP + FP) | % de predições positivas corretas |
| **Recall** | TP / (TP + FN) | % de amostras reais identificadas |
| **F1-Score** | 2 × (P × R) / (P + R) | Média harmônica de Precision e Recall |

#### Curvas de Aprendizado

O notebook gera gráficos de:
- **Loss vs Épocas**: Treino e Validação
- **Acurácia vs Épocas**: Treino e Validação

Permite identificar:
- **Overfitting**: Gap crescente entre treino e validação
- **Underfitting**: Alta loss em ambos os conjuntos
- **Convergência**: Estabilização das métricas

---

## Como Executar

### Pré-requisitos

- Python 3.10+
- pip ou conda

### Instalação

1. **Clonar ou baixar o repositório**

2. **Criar ambiente virtual (recomendado)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Instalar dependências**
   ```bash
   pip install -r requirements.txt
   ```

### Execução

1. **Abrir o Jupyter Notebook**
   ```bash
   jupyter notebook projeto_integrador_3b_deep_learning.ipynb
   ```

2. **Executar todas as células sequencialmente** (Kernel → Restart & Run All)

### Artefatos Gerados

Após a execução, serão gerados:
- `modelo_mlp_classificacao_linhas.h5` - Modelo treinado
- `label_encoder.pkl` - Codificador de labels
- `scaler.pkl` - Normalizador de features
- `distribuicao_tipos_linha.png` - Gráfico de distribuição
- `top_empresas.png` - Top 5 empresas
- `curvas_aprendizado.png` - Curvas de treino
- `matriz_confusao.png` - Matriz de confusão
- `matriz_confusao_normalizada.png` - Matriz normalizada

---

## Resultados

### Principais Descobertas

1. **Desbalanceamento Severo**: 82.5% das linhas são convencionais, dificultando a classificação das classes minoritárias.

2. **Features Discriminativas**: A variável `E_VARIANTE` mostrou-se importante, pois linhas de serviços especiais (Semiurbano, Expresso) frequentemente são variantes de linhas convencionais.

3. **Concentração Empresarial**: Poucas empresas operam a maioria das linhas, com destaque para "Juarez Mendes de Melo" (45 linhas).

### Limitações

- **Dados Limitados**: Apenas 285 registros, insuficientes para treinar redes mais complexas
- **Classes Raras**: "Viagens Diretas" (2 amostras) e "Expresso" (3 amostras) são difíceis de classificar
- **Features Disponíveis**: A base não possui informações como distância em km, tempo de viagem ou demanda de passageiros

### Melhorias Futuras

- Aplicar técnicas de oversampling (SMOTE) para classes minoritárias
- Incluir features geográficas (coordenadas, distância euclidiana)
- Testar arquiteturas mais complexas com mais dados
- Implementar validação cruzada k-fold

---

## Referências

1. AGR - Agência Goiana de Regulação. **Portal de Dados Abertos do Estado de Goiás**. Disponível em: https://dadosabertos.go.gov.br/

2. GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. **Deep Learning**. MIT Press, 2016.

3. CHOLLET, F. **Deep Learning with Python**. Manning Publications, 2021.

4. TensorFlow Documentation. Disponível em: https://www.tensorflow.org/

5. Scikit-learn Documentation. Disponível em: https://scikit-learn.org/

6. BRASIL. **Resolução Nº 7/2018** - Diretrizes para a Extensão na Educação Superior Brasileira. MEC/CNE.

---

**Goiânia, 2025**

*Projeto desenvolvido como requisito da disciplina Projeto Integrador III-B do curso de Big Data e Inteligência Artificial da PUC Goiás.*

