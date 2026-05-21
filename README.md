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
