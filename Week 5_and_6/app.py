import streamlit as st
import pandas as pd
import numpy as np
from dataset import GENRES, load_movie_df, get_movie_matrix, get_movie_vector
from recommender import recommend_similar_movies, get_all_pairwise_similarities, dot_product, cosine_similarity
from linear_algebra import run_pca_scratch
from ml_engine import UserPreferenceLearner
import visualizer as viz

# Page Configuration
st.set_page_config(
    page_title="Movie Recommender - Linear Algebra & ML",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic Dark UI
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container Styling */
    .header-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }

    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        transition: transform 0.2s ease;
    }
    .card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }

    /* Recommendation Badge */
    .rec-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Topic Pill */
    .topic-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="header-box">
    <div class="header-title">🎬 Movie Recommendation System</div>
    <div class="header-subtitle">Powered by <b>Linear Algebra</b> (Vectors, Matrices, Dot Product, PCA) & <b>Machine Learning</b> (Derivatives, Gradients, Chain Rule)</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Syllabus Topics Summary
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/movie-projector.png", width=70)
    st.title("📌 Syllabus Topics")
    st.markdown("""
    This application covers all **7 Linear Algebra & Calculus** topics:
    
    1. 📐 **Vectors**: Movie representations
    2. 📊 **Matrices**: Movie-Feature matrix
    3. 🎯 **Dot Product**: Similarity scoring
    4. 🧩 **Eigenvalues & Eigenvectors**: PCA 2D projection
    5. 📉 **Derivatives**: MSE rating loss
    6. 🧭 **Gradients**: Multivariable preference gradient
    7. ⛓️ **Chain Rule**: Gradient descent backprop
    """)
    st.divider()
    st.info("💡 **Tip**: Navigate through the tabs to explore similarity recommendations, user preference learning, PCA visualization, and mathematical theory.")

# Load dataset
df_movies = load_movie_df()
titles_list = df_movies["Title"].tolist()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Movie Similarity Recommender",
    "🧠 ML Preference Learner (Gradient Descent)",
    "📐 Linear Algebra & PCA Explorer",
    "📚 Mathematical Syllabus Guide"
])

# ==========================================
# TAB 1: MOVIE SIMILARITY RECOMMENDER
# ==========================================
with tab1:
    st.subheader("🎬 Movie-to-Movie Recommendation Engine")
    st.write("Select a movie to find the most similar movies using vector similarity (Dot Product or Cosine Similarity).")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_movie = st.selectbox("Select Target Movie:", titles_list, index=0)
    with col2:
        num_recs = st.slider("Recommendations:", min_value=1, max_value=10, value=5)
    with col3:
        similarity_metric = st.radio("Similarity Metric:", ["Cosine Similarity", "Dot Product"], index=0)

    metric_key = "cosine" if similarity_metric == "Cosine Similarity" else "dot"

    if st.button("🚀 Find Similar Movies", type="primary"):
        rec_df, target_title, target_vec, top_recs = recommend_similar_movies(
            selected_movie, top_n=num_recs, metric=metric_key
        )

        st.markdown(f"### 🍿 Movies Similar to **{target_title}**")

        c1, c2 = st.columns([3, 2])

        with c1:
            st.dataframe(
                rec_df,
                hide_index=True,
                use_container_width=True
            )

        with c2:
            st.markdown("#### 🎯 Target Feature Vector")
            vec_df = pd.DataFrame([target_vec], columns=GENRES, index=[target_title])
            st.dataframe(vec_df, use_container_width=True)

        st.divider()
        st.markdown("### 🧮 Mathematical Dot Product Breakdown")
        st.write("Below is the step-by-step vector dot product calculation for the top recommendation:")

        top_rec_item = top_recs[0]
        top_title = top_rec_item["Title"]
        top_vec = top_rec_item["Vector"]
        elem_prod = top_rec_item["ElemProduct"]
        dp_sum = top_rec_item["Dot Product"]

        math_breakdown_df = pd.DataFrame({
            "Genre Feature": GENRES,
            f"Target ({target_title})": target_vec,
            f"Candidate ({top_title})": top_vec,
            "Elementwise Product (u_i * v_i)": elem_prod
        })

        st.dataframe(math_breakdown_df, hide_index=True, use_container_width=True)

        st.latex(rf"\mathbf{{u}} \cdot \mathbf{{v}} = \sum_{{i=1}}^{{8}} u_i v_i = {dp_sum:.4f}")

# ==========================================
# TAB 2: ML USER PREFERENCE LEARNER
# ==========================================
with tab2:
    st.subheader("🧠 Machine Learning Model: User Preference Optimization")
    st.markdown(r"""
    Rate a few movies below to let the machine learning model learn your personal genre preference vector $\mathbf{w}$ 
    using **Gradient Descent**, **Derivatives**, **Gradients**, and the **Chain Rule**.
    """)

    st.markdown("#### 🌟 Rate Sample Movies (1 = Poor, 5 = Excellent):")

    sample_movies = ["Avengers", "Titanic", "Superbad", "Interstellar", "Spirited Away", "The Dark Knight"]
    user_ratings = {}

    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]

    for idx, m_title in enumerate(sample_movies):
        with cols[idx % 3]:
            # Default rating values to create interesting initial preferences
            default_val = 5.0 if m_title in ["Avengers", "Interstellar", "The Dark Knight"] else 2.0
            r = st.slider(f"Rate **{m_title}**", min_value=1.0, max_value=5.0, value=default_val, step=0.5, key=f"rate_{m_title}")
            user_ratings[m_title] = r

    st.divider()

    hcol1, hcol2 = st.columns(2)
    with hcol1:
        learning_rate = st.select_slider("Learning Rate (α):", options=[0.01, 0.05, 0.1, 0.2, 0.5], value=0.05)
    with hcol2:
        epochs = st.slider("Training Epochs:", min_value=10, max_value=200, value=80, step=10)

    if st.button("⚡ Train Model & Learn Preferences", type="primary"):
        # Construct rated movies list
        rated_list = []
        for m_title, r_val in user_ratings.items():
            vec = get_movie_vector(m_title)
            rated_list.append({"title": m_title, "vector": vec, "rating": r_val})

        # Initialize & Train Learner
        learner = UserPreferenceLearner(n_features=8, learning_rate=learning_rate)
        history = learner.train_step(rated_list, epochs=epochs)

        st.success(f"✅ Gradient Descent Model trained successfully across {epochs} epochs! Final Loss: {history[-1]['loss']:.4f}")

        vcol1, vcol2 = st.columns(2)

        with vcol1:
            fig_loss = viz.plot_loss_curve(history)
            st.plotly_chart(fig_loss, use_container_width=True)

        with vcol2:
            fig_weights = viz.plot_user_weights(learner.w, GENRES)
            st.plotly_chart(fig_weights, use_container_width=True)

        st.divider()

        st.markdown("### 🍿 Recommended Movies Based on Learned Preferences")
        rec_unrated_df = learner.recommend_unrated(list(user_ratings.keys()), top_n=5)
        
        col_rec1, col_rec2 = st.columns([3, 2])
        with col_rec1:
            st.dataframe(rec_unrated_df[["Movie", "Predicted Rating"]], hide_index=True, use_container_width=True)

        with col_rec2:
            st.markdown("#### 📐 Model Parameter Vector")
            w_df = pd.DataFrame([learner.w], columns=GENRES, index=["Weight (w)"])
            st.dataframe(w_df, use_container_width=True)
            st.metric("Learned Bias (b)", f"{learner.b:.2f}")

        # Calculus Breakdown Expander
        with st.expander("🔍 Step-by-Step Gradient Descent Calculus Math Breakdown"):
            st.markdown(r"""
            #### 1. Forward Pass (Prediction)
            $$\hat{y}_i = \mathbf{w} \cdot \mathbf{v}_i + b$$
            
            #### 2. Loss Function (Mean Squared Error)
            $$L_i = \frac{1}{2} (\hat{y}_i - y_i)^2$$
            
            #### 3. Derivative of Loss w.r.t Prediction
            $$\frac{\partial L_i}{\partial \hat{y}_i} = (\hat{y}_i - y_i) = e_i \quad (\text{Error})$$
            
            #### 4. Chain Rule for Gradient Vector
            $$\frac{\partial L_i}{\partial w_j} = \frac{\partial L_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial w_j} = e_i \cdot v_{ij}$$
            $$\nabla_{\mathbf{w}} L_i = e_i \mathbf{v}_i = (\hat{y}_i - y_i) \mathbf{v}_i$$
            
            #### 5. Gradient Descent Update Rule
            $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla_{\mathbf{w}} L$$
            $$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$
            """)

# ==========================================
# TAB 3: LINEAR ALGEBRA & PCA EXPLORER
# ==========================================
with tab3:
    st.subheader("📐 Linear Algebra & PCA Explorer")
    st.markdown(r"""
    Explore the fundamental linear algebra structures powering movie recommendations:
    **Movie-Feature Matrix ($X$)**, **Covariance Matrix ($C$)**, **Eigenvalues ($\lambda$)**, **Eigenvectors ($\mathbf{e}$)**, and 
    **Principal Component Analysis (PCA)** dimensionality reduction.
    """)

    pca_results = run_pca_scratch(n_components=2)
    titles = pca_results["titles"]
    Z = pca_results["Z"]
    eigenvalues = pca_results["eigenvalues"]
    eigenvectors = pca_results["eigenvectors"]
    exp_var = pca_results["explained_variance_ratio"]
    C = pca_results["C"]

    pcol1, pcol2 = st.columns([3, 2])

    with pcol1:
        fig_pca = viz.plot_pca_2d(titles, Z, exp_var)
        st.plotly_chart(fig_pca, use_container_width=True)

    with pcol2:
        st.markdown("#### 🧩 Eigenvalues & Variance Explained")
        eig_df = pd.DataFrame({
            "Component": [f"PC {i+1}" for i in range(len(eigenvalues))],
            "Eigenvalue (λ)": np.round(eigenvalues, 4),
            "Variance Explained (%)": np.round(exp_var * 100, 2)
        })
        st.dataframe(eig_df, hide_index=True, use_container_width=True)
        st.info(f"💡 The top 2 Principal Components account for **{(exp_var[0]+exp_var[1])*100:.1f}%** of total dataset variance.")

    st.divider()

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### 📊 Movie Feature Matrix (X)")
        st.dataframe(df_movies.head(10), use_container_width=True)

    with col_m2:
        st.markdown("#### 📊 Genre Covariance Matrix (C)")
        fig_cov = viz.plot_covariance_heatmap(C, GENRES)
        st.plotly_chart(fig_cov, use_container_width=True)

    with st.expander("🔍 View Top Eigenvectors Matrix (W_k)"):
        eig_vec_df = pd.DataFrame(eigenvectors[:, :2], index=GENRES, columns=["PC 1 Vector", "PC 2 Vector"])
        st.dataframe(eig_vec_df, use_container_width=True)

# ==========================================
# TAB 4: MATHEMATICAL SYLLABUS GUIDE
# ==========================================
with tab4:
    st.subheader("📚 7-Topic Mathematical Syllabus Mapping")
    st.markdown("""
    This reference section explains how each of the 7 required syllabus topics is rigorously integrated into the recommender architecture.
    """)

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "1. Vectors",
        "2. Matrices",
        "3. Dot Product",
        "4. Eigenvalues & Eigenvectors",
        "5. Derivatives",
        "6. Gradients",
        "7. Chain Rule"
    ])

    with t1:
        st.markdown(r"""
        ### 1. Vectors
        Each movie $m_i$ is represented as a feature vector $\mathbf{v}_i \in \mathbb{R}^d$ in an 8-dimensional genre space:
        $$\mathbf{v}_{\text{Avengers}} = \begin{bmatrix} 1.0 & 0.2 & 0.0 & 0.9 & 0.3 & 0.4 & 0.0 & 0.8 \end{bmatrix}^T$$
        Where dimensions represent: `[Action, Comedy, Romance, Sci-Fi, Drama, Thriller, Animation, Fantasy]`.
        """)

    with t2:
        st.markdown(r"""
        ### 2. Matrices
        All movie vectors are concatenated into a **Movie-Feature Matrix** $X \in \mathbb{R}^{N \times d}$ where $N=45$ movies and $d=8$ genres.
        Matrix operations allow batch processing of similarity scores:
        $$S = X_{\text{norm}} X_{\text{norm}}^T$$
        Where $S_{ij}$ is the cosine similarity between movie $i$ and movie $j$.
        """)

    with t3:
        st.markdown(r"""
        ### 3. Dot Product
        The dot product measures alignment between two movie vectors $\mathbf{u}$ and $\mathbf{v}$:
        $$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)$$
        Cosine similarity normalizes for magnitude:
        $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
        """)

    with t4:
        st.markdown(r"""
        ### 4. Eigenvalues & Eigenvectors (PCA)
        Principal Component Analysis solves the characteristic equation of the sample covariance matrix $C$:
        $$C \mathbf{e}_k = \lambda_k \mathbf{e}_k$$
        - **Eigenvalues ($\lambda_k$)**: Represent variance magnitude along principal direction $k$.
        - **Eigenvectors ($\mathbf{e}_k$)**: Define orthogonal directions of maximal feature variance.
        """)

    with t5:
        st.markdown(r"""
        ### 5. Derivatives
        The user preference learner optimizes a Mean Squared Error (MSE) loss function:
        $$L(\mathbf{w}, b) = \frac{1}{2} (\hat{y} - y)^2$$
        The derivative w.r.t the model output $\hat{y}$ quantifies the rating prediction error:
        $$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = e$$
        """)

    with t6:
        st.markdown(r"""
        ### 6. Gradients
        The gradient vector $\nabla_{\mathbf{w}} L$ collects partial derivatives across all 8 genre features:
        $$\nabla_{\mathbf{w}} L = \left[ \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \dots, \frac{\partial L}{\partial w_8} \right]^T$$
        It points in the direction of steepest increase of recommendation error.
        """)

    with t7:
        st.markdown(r"""
        ### 7. Chain Rule
        By the multivariable Chain Rule, the gradient of the loss w.r.t feature weight $w_j$ is:
        $$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_j} = (\hat{y} - y) \cdot v_j$$
        This enables backpropagation to update user preference weights:
        $$\mathbf{w} \leftarrow \mathbf{w} - \alpha (\hat{y} - y) \mathbf{v}_i$$
        """)

# Footer
st.divider()
st.caption("🚀 Built for PBL Hack-O-Week | Python, NumPy, Pandas, Scikit-learn, Plotly, Streamlit")
