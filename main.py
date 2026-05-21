# """
# ╔══════════════════════════════════════════════════════════════════╗
# ║        CALCULADORA DE ENGAJAMENTO DE TIKTOK                     ║
# ║        Stack: Streamlit · TensorFlow · NumPy · Pandas           ║
# ╚══════════════════════════════════════════════════════════════════╝

# Como executar:
#     pip install streamlit tensorflow numpy pandas
#     streamlit run tiktok_engajamento.py
# """

# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras

# # ──────────────────────────────────────────────────────────────────
# # 1. CONFIGURAÇÃO DA PÁGINA
# # ──────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="TikTok Engagement Calculator",
#     page_icon="🎵",
#     layout="centered",
# )

# # CSS customizado para interface moderna e escura no estilo TikTok
# st.markdown("""
# <style>
#     /* Fundo geral */
#     .stApp {
#         background: linear-gradient(135deg, #0a0a0f 0%, #1a0a1f 50%, #0a0f1a 100%);
#     }

#     /* Container principal */
#     .main .block-container {
#         max-width: 700px;
#         padding: 2rem 2rem;
#     }

#     /* Título principal */
#     .titulo-principal {
#         text-align: center;
#         font-family: 'Arial Black', sans-serif;
#         font-size: 2.8rem;
#         font-weight: 900;
#         background: linear-gradient(90deg, #FF2D55, #00F5FF, #FFD60A);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.2rem;
#         letter-spacing: 2px;
#     }

#     .subtitulo {
#         text-align: center;
#         color: rgba(255,255,255,0.45);
#         font-size: 0.9rem;
#         margin-bottom: 2rem;
#         letter-spacing: 1px;
#     }

#     /* Cards de resultado */
#     .card-viral {
#         background: linear-gradient(135deg, rgba(255,214,10,0.15), rgba(255,45,85,0.1));
#         border: 1px solid rgba(255,214,10,0.4);
#         border-radius: 16px;
#         padding: 28px;
#         text-align: center;
#         margin-top: 1.5rem;
#     }

#     .card-flop {
#         background: rgba(255,255,255,0.04);
#         border: 1px solid rgba(255,255,255,0.12);
#         border-radius: 16px;
#         padding: 28px;
#         text-align: center;
#         margin-top: 1.5rem;
#     }

#     .resultado-numero {
#         font-size: 3rem;
#         font-weight: 900;
#         font-family: 'Arial Black', sans-serif;
#         margin: 0.3rem 0;
#     }

#     .resultado-label {
#         font-size: 0.8rem;
#         text-transform: uppercase;
#         letter-spacing: 2px;
#         color: rgba(255,255,255,0.5);
#     }

#     /* Métricas laterais */
#     .metrica-box {
#         background: rgba(255,255,255,0.04);
#         border: 0.5px solid rgba(255,255,255,0.1);
#         border-radius: 12px;
#         padding: 16px;
#         text-align: center;
#     }

#     /* Labels dos sliders e selects */
#     .stSelectbox label, .stSlider label {
#         color: rgba(255,255,255,0.7) !important;
#         font-weight: 500;
#     }

#     /* Botão principal */
#     .stButton > button {
#         background: linear-gradient(90deg, #FF2D55, #FF6B35);
#         color: white;
#         font-weight: 800;
#         font-size: 1.1rem;
#         letter-spacing: 2px;
#         border: none;
#         border-radius: 12px;
#         padding: 0.8rem 2rem;
#         width: 100%;
#         transition: opacity 0.2s;
#         text-transform: uppercase;
#     }

#     .stButton > button:hover {
#         opacity: 0.85;
#     }

#     /* Seção de dica */
#     .dica-box {
#         background: rgba(255,45,85,0.08);
#         border: 0.5px solid rgba(255,45,85,0.25);
#         border-radius: 10px;
#         padding: 14px 18px;
#         font-size: 0.85rem;
#         color: rgba(255,255,255,0.65);
#         margin-top: 1rem;
#         line-height: 1.7;
#     }

#     /* Divider customizado */
#     hr { border-color: rgba(255,255,255,0.08) !important; }

#     /* Textos gerais */
#     p, li { color: rgba(255,255,255,0.75); }
# </style>
# """, unsafe_allow_html=True)


# # ──────────────────────────────────────────────────────────────────
# # 2. GERAÇÃO DE DADOS SIMULADOS
# # ──────────────────────────────────────────────────────────────────
# @st.cache_resource
# def gerar_dados_treino():
#     """
#     Gera um dataset simulado com 2.000 amostras representando
#     posts de TikTok com diferentes categorias e número de hashtags.

#     Features:
#         - categoria_encoded : inteiro 0–4 mapeado para cada nicho
#         - hashtags_norm     : número de hashtags normalizado [0, 1]

#     Target:
#         - alcance_norm : alcance estimado normalizado [0, 1]
#     """
#     np.random.seed(42)
#     N = 2000

#     # Boost base por categoria (simula engajamento natural de cada nicho)
#     # Dança=4, Humor=3, Gameplay=2, Lifestyle=1, Música=0
#     categoria_boost = {4: 1.4, 3: 1.3, 2: 1.1, 1: 1.05, 0: 1.2}

#     categorias   = np.random.randint(0, 5, N)           # 5 categorias
#     hashtags_raw = np.random.randint(0, 31, N)          # 0–30 hashtags

#     alcances = []
#     for cat, h in zip(categorias, hashtags_raw):
#         boost = categoria_boost[cat]

#         # Curva de hashtags: cresce até ~7, depois decresce (penalização)
#         if h == 0:
#             hash_score = 0.25
#         elif h <= 7:
#             hash_score = 0.25 + (h / 7) * 0.75
#         else:
#             hash_score = 1.0 - min(0.70, ((h - 7) / 23) * 0.70)

#         # Alcance = boost × score com ruído gaussiano
#         alcance = boost * hash_score * 100_000
#         alcance += np.random.normal(0, 5_000)
#         alcances.append(max(500, alcance))

#     alcances = np.array(alcances, dtype=np.float32)

#     # Normalização min-max para o TensorFlow
#     cat_norm  = categorias.astype(np.float32) / 4.0
#     hash_norm = hashtags_raw.astype(np.float32) / 30.0
#     alc_norm  = (alcances - alcances.min()) / (alcances.max() - alcances.min())

#     X = np.stack([cat_norm, hash_norm], axis=1)
#     y = alc_norm

#     # Guarda os limites para desnormalizar depois
#     meta = {"alc_min": float(alcances.min()), "alc_max": float(alcances.max())}
#     return X, y, meta


# # ──────────────────────────────────────────────────────────────────
# # 3. CONSTRUÇÃO E TREINO DO MODELO TENSORFLOW
# # ──────────────────────────────────────────────────────────────────
# @st.cache_resource
# def treinar_modelo():
#     """
#     Rede Neural Sequencial com:
#         - Camada de entrada (2 features)
#         - 2 camadas ocultas com ativação ReLU e Dropout
#         - Camada de saída com ativação Sigmoid (saída [0,1])

#     Treinado com Adam + MSE por 60 épocas.
#     O cache do Streamlit garante que o treino rode uma única vez.
#     """
#     X, y, meta = gerar_dados_treino()

#     modelo = keras.Sequential([
#         keras.layers.Input(shape=(2,)),
#         keras.layers.Dense(64, activation="relu"),
#         keras.layers.Dropout(0.1),
#         keras.layers.Dense(32, activation="relu"),
#         keras.layers.Dropout(0.1),
#         keras.layers.Dense(1, activation="sigmoid"),
#     ], name="tiktok_engagement_model")

#     modelo.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=0.003),
#         loss="mse",
#         metrics=["mae"]
#     )

#     modelo.fit(X, y, epochs=60, batch_size=64, verbose=0)
#     return modelo, meta


# # ──────────────────────────────────────────────────────────────────
# # 4. FUNÇÃO DE PREDIÇÃO
# # ──────────────────────────────────────────────────────────────────
# CATEGORIAS = {
#     "💃 Dança":    4,
#     "😂 Humor":    3,
#     "🎮 Gameplay": 2,
#     "🌟 Lifestyle":1,
#     "🎵 Música":   0,
# }

# def prever_alcance(modelo, meta, categoria_idx, n_hashtags):
#     """
#     Normaliza os inputs, roda a predição e desnormaliza o resultado.

#     Returns:
#         alcance (int): alcance estimado em número de visualizações
#     """
#     cat_norm  = np.array([[categoria_idx / 4.0, n_hashtags / 30.0]], dtype=np.float32)
#     pred_norm = modelo.predict(cat_norm, verbose=0)[0][0]

#     # Desnormalização: recupera a escala original
#     alcance = pred_norm * (meta["alc_max"] - meta["alc_min"]) + meta["alc_min"]
#     return int(alcance)


# ──────────────────────────────────────────────────────────────────
# 5. INTERFACE STREAMLIT
# ──────────────────────────────────────────────────────────────────
# st.markdown('<div class="titulo-principal">🎵 TIKTOK SCORE</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitulo">CALCULADORA DE ENGAJAMENTO · POWERED BY IA</div>', unsafe_allow_html=True)

# # ── Carrega modelo com spinner ─────────────────────────────────────
# with st.spinner("🧠 Treinando modelo de IA..."):
#     modelo, meta = treinar_modelo()

# st.divider()

# # ── Inputs do usuário ──────────────────────────────────────────────
# col1, col2 = st.columns([1.2, 1])

# with col1:
#     categoria_label = st.selectbox(
#         "🎬 Categoria do vídeo",
#         options=list(CATEGORIAS.keys()),
#         help="Cada categoria possui um multiplicador de engajamento diferente."
#     )

# with col2:
#     n_hashtags = st.slider(
#         "🏷️ Número de hashtags",
#         min_value=0,
#         max_value=30,
#         value=7,
#         step=1,
#         help="Ótimo: 5–9 hashtags. Excesso reduz o alcance orgânico."
#     )

# # ── Botão de previsão ──────────────────────────────────────────────
# if st.button("⚡ PREVER ENGAJAMENTO"):
#     categoria_idx = CATEGORIAS[categoria_label]
#     alcance       = prever_alcance(modelo, meta, categoria_idx, n_hashtags)

#     # Limiar de viralização: 70.000 views
#     LIMIAR_VIRAL = 70_000
#     is_viral     = alcance >= LIMIAR_VIRAL

#     # Formata o número para exibição
#     if alcance >= 1_000_000:
#         alcance_fmt = f"{alcance/1_000_000:.1f}M"
#     elif alcance >= 1_000:
#         alcance_fmt = f"{alcance/1_000:.0f}K"
#     else:
#         alcance_fmt = str(alcance)

#     # Percentual de potencial viral (0–100)
#     viral_pct = min(100, int((alcance / 140_000) * 100))

#     # ── Card de resultado ─────────────────────────────────────────
#     if is_viral:
#         st.markdown(f"""
#         <div class="card-viral">
#             <div style="font-size:3.5rem; margin-bottom:4px;">🔥</div>
#             <div style="font-size:2.2rem; font-weight:900; color:#FFD60A; letter-spacing:3px;">VIRAL!</div>
#             <div class="resultado-numero" style="color:#00F5FF;">{alcance_fmt}</div>
#             <div class="resultado-label">visualizações estimadas</div>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div class="card-flop">
#             <div style="font-size:3.5rem; margin-bottom:4px;">💀</div>
#             <div style="font-size:2.2rem; font-weight:900; color:rgba(255,255,255,0.35); letter-spacing:3px;">FLOPADO</div>
#             <div class="resultado-numero" style="color:rgba(255,255,255,0.5);">{alcance_fmt}</div>
#             <div class="resultado-label">visualizações estimadas</div>
#         </div>
#         """, unsafe_allow_html=True)

#     st.write("")

#     # ── Métricas secundárias ──────────────────────────────────────
#     m1, m2, m3 = st.columns(3)

#     eng_rate = round(min(18.0, (alcance / 140_000) * 15 + 1.5), 1)
#     score    = min(100, int((alcance / 140_000) * 100))

#     with m1:
#         st.metric("💹 Engajamento", f"{eng_rate}%")
#     with m2:
#         st.metric("🎯 Viral Score", f"{score}/100")
#     with m3:
#         st.metric("🏷️ Hashtags", f"{n_hashtags} tags")

#     # ── Barra de progresso ────────────────────────────────────────
#     st.write("")
#     st.caption("📊 Potencial viral")
#     st.progress(viral_pct / 100)

#     # ── Tabela de dados do post ───────────────────────────────────
#     st.write("")
#     df_resultado = pd.DataFrame({
#         "Parâmetro":       ["Categoria", "Hashtags", "Alcance est.", "Engajamento", "Status"],
#         "Valor":           [
#             categoria_label,
#             f"{n_hashtags} hashtags",
#             alcance_fmt,
#             f"{eng_rate}%",
#             "🔥 Viral" if is_viral else "💀 Flopado"
#         ]
#     })
#     st.dataframe(df_resultado, use_container_width=True, hide_index=True)

#     # ── Dica personalizada ────────────────────────────────────────
#     if n_hashtags == 0:
#         dica = "⚠️ <strong>Sem hashtags = sem distribuição.</strong> Adicione pelo menos 3–5 hashtags relevantes para ativar o algoritmo de descoberta do TikTok."
#     elif n_hashtags > 15:
#         dica = f"⚠️ <strong>Hashtags em excesso.</strong> Você usou {n_hashtags} tags — o TikTok penaliza posts com muitas hashtags. Tente reduzir para 5–9."
#     elif is_viral:
#         dica = f"✅ <strong>Configuração ideal!</strong> {categoria_label} + {n_hashtags} hashtags é uma combinação forte. Poste entre 19h–22h para maximizar o alcance orgânico."
#     else:
#         dica = f"💡 <strong>Dica de otimização:</strong> Experimente aumentar o apelo visual nos primeiros 3 segundos e use um CTA claro. A categoria {categoria_label} tem potencial — refine o gancho!"

#     st.markdown(f'<div class="dica-box">{dica}</div>', unsafe_allow_html=True)


# # ──────────────────────────────────────────────────────────────────
# # 6. RODAPÉ COM EXPLICAÇÃO DO MODELO
# # ──────────────────────────────────────────────────────────────────
# st.divider()
# with st.expander("🧠 Como funciona o modelo de IA?"):
#     st.markdown("""
#     **Arquitetura:** Rede Neural Sequencial (TensorFlow / Keras)

#     ```
#     Input (2 features)  →  Dense(64, ReLU)  →  Dropout(0.1)
#                         →  Dense(32, ReLU)  →  Dropout(0.1)
#                         →  Dense(1, Sigmoid)
#     ```

#     **Features de entrada:**
#     - `categoria_encoded` — índice da categoria normalizado [0, 1]
#     - `hashtags_norm`     — número de hashtags normalizado [0, 1]

#     **Dados de treino:** 2.000 amostras sintéticas com ruído gaussiano,
#     simulando o comportamento real do algoritmo do TikTok.

#     **Otimizador:** Adam · **Loss:** MSE · **Épocas:** 60

#     **Regras embutidas nos dados:**
#     - Hashtags ótimas ≈ 5–9 (curva em sino)
#     - Categorias com maior boost: 💃 Dança > 😂 Humor > 🎵 Música
#     - Limiar viral: **70.000 views**
#     """)

# st.caption("⚡ Calculadora de Engajamento TikTok · Powered by TensorFlow + Streamlit")
# ============================================================
#  Filtro Antispam de DM — Instagram
#  Stack: Streamlit + TensorFlow/Keras (TextVectorization)
#  Módulo 4 — Prompt CRAFT
# ============================================================

import re
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models

# ── Configuração da página ───────────────────────────────────
st.set_page_config(
    page_title="Filtro Antispam DM · Instagram",
    page_icon="🛡️",
    layout="centered",
)

# ── Estilo visual ────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0f0f0f; color: #f0f0f0; }
  .block-container { max-width: 700px; padding-top: 2rem; }
  .card {
    border-radius: 14px; padding: 1.4rem 1.6rem;
    margin-top: 1.2rem; border: 1px solid;
  }
  .card-safe   { background:#0d2b1a; border-color:#2ecc71; }
  .card-spam   { background:#2b0d0d; border-color:#e74c3c; }
  .badge {
    display:inline-block; border-radius:20px;
    padding:4px 14px; font-size:.85rem; font-weight:600;
  }
  .badge-safe  { background:#2ecc71; color:#000; }
  .badge-spam  { background:#e74c3c; color:#fff; }
  .signal-chip {
    display:inline-block; border-radius:20px;
    padding:3px 11px; font-size:.78rem; margin:3px 3px 0 0;
  }
  .chip-danger  { background:#4a1111; color:#ff6b6b; border:1px solid #e74c3c; }
  .chip-warning { background:#3a2a00; color:#ffc107; border:1px solid #ffc107; }
  .chip-ok      { background:#0d2b1a; color:#2ecc71; border:1px solid #2ecc71; }
  hr { border-color: #2a2a2a; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  1. DADOS DE TREINAMENTO SIMULADOS
# ════════════════════════════════════════════════════════════

MENSAGENS_SPAM = [
    "PARABÉNS você ganhou R$5000 clique aqui para resgatar bit.ly/premio",
    "URGENTE verifique sua conta agora senha bloqueada acesse o link",
    "Ganhe dinheiro rápido trabalhando de casa cadastre-se grátis",
    "Você foi selecionado para sorteio exclusivo clique e resgate já",
    "Sua conta será suspensa confirme seus dados pessoais urgente",
    "Invista em bitcoin e ganhe 300% ao mês acesse agora",
    "Nude fotos vídeos exclusivos acesse t.me/link agora grátis",
    "Parabéns ganhador do prêmio de R$10000 resgate com pix hoje",
    "Oferta limitada só hoje desconto 90% clique no link abaixo",
    "VERIFICAÇÃO necessária sua conta Instagram será deletada acesse",
    "Ganhe seguidores reais grátis clique aqui 10000 seguidores já",
    "Sua senha foi comprometida acesse link para recuperar conta",
    "Promoção exclusiva só para você clique e ganhe dinheiro fácil",
    "ÚLTIMO AVISO confirme email ou perde acesso à conta instagram",
    "Trabalhe de casa ganhe R$3000 por dia sem experiência clique",
    "Você ganhou iPhone 15 Pro sorteio oficial clique para resgatar",
    "Acesse bit.ly/golpe123 e ganhe créditos grátis no seu celular",
    "GRÁTIS GRÁTIS GRÁTIS clique no link e ganhe prêmio exclusivo agora",
    "Sua conta foi hackeada troque sua senha clicando aqui urgente",
    "Crypto investimento garantido lucro 500 reais por dia cadastre",
]

MENSAGENS_SEGURAS = [
    "Oi adorei seu conteúdo sobre moda sustentável top demais",
    "Olá tudo bem queria saber se você topa uma parceria collab",
    "Boa tarde tenho uma proposta de trabalho para conversar",
    "Parabéns pelo seu portfólio ficou incrível o projeto novo",
    "Você toparia gravar um vídeo junto para o canal da agência",
    "Oi vi seu post sobre viagem e queria algumas dicas por favor",
    "Tenho uma dúvida sobre o seu último tutorial pode me ajudar",
    "Nossa marca adoraria te enviar produtos para resenha obrigada",
    "Olá somos uma agência de marketing e temos interesse em parceria",
    "Vi seu trabalho no feed e fiquei impressionado com a qualidade",
    "Boa noite queria contratar uma sessão de fotos com você",
    "Você faz freela de design gráfico tem portfolio para ver",
    "Amei sua última receita vou tentar fazer hoje em casa",
    "Olá sou fotógrafa e adoraria trocar experiências com você",
    "Oi segui seu perfil há anos e você me inspira muito obrigada",
    "Podemos conversar sobre orçamento para campanha da minha loja",
    "Seu curso de edição de vídeo vale a pena tem desconto",
    "Parabéns pelo trabalho você cresceu muito no último ano",
    "Tenho interesse em divulgar meu produto no seu perfil qual valor",
    "Olá vi que você cobre eventos quero contratar para meu casamento",
]

textos = MENSAGENS_SPAM + MENSAGENS_SEGURAS
rotulos = np.array([1]*len(MENSAGENS_SPAM) + [0]*len(MENSAGENS_SEGURAS), dtype="float32")


# ════════════════════════════════════════════════════════════
#  2. PRÉ-PROCESSAMENTO — limpeza de texto
# ════════════════════════════════════════════════════════════

def limpar_texto(texto: str) -> str:
    """Remove caracteres irrelevantes, normaliza espaços e caixa."""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\.\S+", " URL ", texto)   # URLs
    texto = re.sub(r"\d{8,}", " NUMERO ", texto)           # números longos
    texto = re.sub(r"[^a-záéíóúãõâêôçüàì\s]", " ", texto) # acentuação PT-BR
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

textos_limpos = [limpar_texto(t) for t in textos]


# ════════════════════════════════════════════════════════════
#  3. CAMADA TextVectorization (Tokenização + Vetorização)
# ════════════════════════════════════════════════════════════

VOCAB_SIZE   = 500
SEQUENCE_LEN = 30

vectorize_layer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LEN,
    standardize="lower_and_strip_punctuation",
)
vectorize_layer.adapt(textos_limpos)


# ════════════════════════════════════════════════════════════
#  4. MODELO — Embedding + Dense
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Treinando modelo de NLP…")
def treinar_modelo():
    inp = tf.keras.Input(shape=(1,), dtype=tf.string)
    x   = vectorize_layer(inp)
    x   = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32, mask_zero=True)(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1,  activation="sigmoid")(x)

    modelo = models.Model(inp, out)
    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    xs = np.array(textos_limpos)
    modelo.fit(
        xs, rotulos,
        epochs=60,
        batch_size=8,
        validation_split=0.15,
        verbose=0,
    )
    return modelo

modelo = treinar_modelo()


# ════════════════════════════════════════════════════════════
#  5. SINAIS LINGUÍSTICOS (heurística explicável)
# ════════════════════════════════════════════════════════════

SINAIS = {
    "Link encurtado":      (r"bit\.ly|t\.me|tinyurl|goo\.gl|ow\.ly", "danger"),
    "Pedido de senha":     (r"senha|password|login|token|código de acesso", "danger"),
    "Promessa de prêmio":  (r"ganhou|prêmio|ganhador|sorteio|resgate", "danger"),
    "Urgência fabricada":  (r"urgente|último aviso|imediatamente|bloqueada", "warning"),
    "Menção financeira":   (r"pix|bitcoin|crypto|dinheiro rápido|lucro", "warning"),
    "CAPS excessivo":      (None, "warning"),   # tratado separado
    "Número longo":        (r"\d{9,}", "warning"),
    "Oferta irreal":       (r"grátis|free|300%|500%|90% de desconto", "warning"),
}

def detectar_sinais(texto_original: str) -> list[dict]:
    encontrados = []
    lower = texto_original.lower()
    for nome, (padrao, tipo) in SINAIS.items():
        if padrao and re.search(padrao, lower):
            encontrados.append({"label": nome, "tipo": tipo})
    palavras = texto_original.split()
    caps = sum(1 for p in palavras if p.isupper() and len(p) > 2)
    if caps >= 3:
        encontrados.append({"label": "CAPS excessivo", "tipo": "warning"})
    if not encontrados:
        encontrados.append({"label": "Nenhum sinal suspeito", "tipo": "ok"})
    return encontrados


# ════════════════════════════════════════════════════════════
#  6. INTERFACE — Streamlit
# ════════════════════════════════════════════════════════════

# ── Cabeçalho ───────────────────────────────────────────────
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("## 🛡️")
with col2:
    st.markdown("## Filtro Antispam de DM")
    st.caption("Instagram · TensorFlow + TextVectorization · Streamlit")

st.markdown("---")

# ── Exemplos rápidos ─────────────────────────────────────────
st.markdown("**Carregar exemplo:**")
c1, c2 = st.columns(2)
exemplo_escolhido = ""
if c1.button("⚠️  Exemplo Spam", use_container_width=True):
    exemplo_escolhido = ("PARABÉNS!! Você GANHOU R$5.000 no nosso SORTEIO EXCLUSIVO! "
                         "Clique AGORA para resgatar: bit.ly/premio2025 — "
                         "URGENTE só até hoje!")
if c2.button("✅  Exemplo Seguro", use_container_width=True):
    exemplo_escolhido = ("Oi, tudo bem? Adorei muito o seu conteúdo sobre moda sustentável! "
                         "Sou da agência CreativeHub e queria propor uma parceria para nossa "
                         "próxima campanha. Podemos conversar sobre o briefing?")

# ── Caixa de texto ───────────────────────────────────────────
dm_texto = st.text_area(
    label="Cole aqui a DM recebida:",
    value=exemplo_escolhido,
    height=130,
    placeholder="Ex.: Parabéns! Você ganhou R$5.000 — clique no link para resgatar...",
)

# ── Botão de análise ─────────────────────────────────────────
analisar = st.button("🔍  Analisar mensagem", type="primary", use_container_width=True)

# ── Resultado ────────────────────────────────────────────────
if analisar:
    if not dm_texto.strip():
        st.warning("Cole uma mensagem antes de analisar.", icon="✏️")
    else:
        with st.spinner("Processando com TensorFlow…"):
            texto_proc = limpar_texto(dm_texto)
            entrada    = np.array([texto_proc])
            score      = float(modelo.predict(entrada, verbose=0)[0][0])

        is_spam    = score > 0.5
        confianca  = score if is_spam else 1 - score
        sinais     = detectar_sinais(dm_texto)

        # ── Card de resultado ───────────────────────────────
        if is_spam:
            st.markdown(f"""
            <div class="card card-spam">
              <span class="badge badge-spam">⚠️  SPAM / GOLPE</span>
              <p style="margin:.8rem 0 .3rem;font-size:1rem;">
                Esta mensagem apresenta <strong>padrões de golpe</strong>.
                Não clique em links nem forneça dados pessoais.
              </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card card-safe">
              <span class="badge badge-safe">✅  SEGURA</span>
              <p style="margin:.8rem 0 .3rem;font-size:1rem;">
                Nenhum padrão suspeito relevante detectado.
                A mensagem parece <strong>legítima</strong>.
              </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probabilidade de spam ───────────────────────────
        st.markdown("**Probabilidade de spam (modelo TensorFlow)**")
        st.progress(score, text=f"{score*100:.1f}%")

        # ── Confiança da classificação ──────────────────────
        st.markdown("**Confiança da classificação**")
        st.progress(confianca, text=f"{confianca*100:.1f}%")

        # ── Sinais linguísticos ─────────────────────────────
        st.markdown("**Sinais detectados na mensagem:**")
        chips_html = ""
        for s in sinais:
            classe = f"chip-{s['tipo']}"
            chips_html += f'<span class="signal-chip {classe}">{s["label"]}</span>'
        st.markdown(chips_html, unsafe_allow_html=True)

        # ── Detalhes técnicos (expansível) ──────────────────
        with st.expander("🔬 Detalhes técnicos do modelo"):
            st.markdown(f"""
| Parâmetro               | Valor |
|-------------------------|-------|
| Texto pré-processado    | `{texto_proc[:80]}…` |
| Score bruto (sigmoid)   | `{score:.6f}` |
| Vocabulário (tokens)    | `{VOCAB_SIZE}` |
| Tamanho da sequência    | `{SEQUENCE_LEN}` |
| Épocas de treinamento   | `60` |
| Camadas                 | Embedding → GlobalAvgPool → Dense(32) → Dropout → Dense(16) → Dense(1) |
""")

# ── Rodapé ───────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Modelo treinado com exemplos simulados de DMs reais do Instagram. "
    "Para uso em produção, expanda o dataset com milhares de exemplos rotulados."
)