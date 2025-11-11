import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import (
    binom, poisson, geom, nbinom, hypergeom,
    uniform, norm, expon, gamma, beta, weibull_min, t
)
from scipy.stats import gaussian_kde
from scipy.special import gamma as gamma_func

st.set_page_config(page_title="Probabilidad y Estadística Interactiva", layout="wide")

# ===========================
# HELPER FUNCTIONS
# ===========================
def get_discrete_range(dist_name, params, observed_max=None):
    """Determina el rango razonable de valores para distribuciones discretas"""
    if dist_name == "Binomial":
        n = params['n']
        return np.arange(0, n + 1)
    elif dist_name == "Poisson":
        lmbda = params['lambda']
        # Rango hasta donde la probabilidad es significativa (sin límite superior arbitrario)
        k_max = int(np.ceil(lmbda + 10 * np.sqrt(max(lmbda, 1))))
        k_max = max(k_max, 20)  # Solo mínimo razonable para visualización
        if observed_max:
            k_max = max(k_max, int(observed_max) + 10)
        return np.arange(0, k_max + 1)
    elif dist_name == "Geométrica":
        p = params['p']
        if p > 0 and p <= 1:
            # Hasta el percentil 99.9 (sin límite arbitrario)
            k_max = int(np.ceil(-np.log(0.001) / np.log(1 - p)))
            k_max = max(k_max, 20)
            return np.arange(1, k_max + 1)
        return np.arange(1, 100)
    elif dist_name == "Binomial Negativa":
        r, p = params['r'], params['p']
        if p > 0 and p <= 1:
            mean = r * (1-p) / p
            std = np.sqrt(r * (1-p) / p**2)
            # 10 desviaciones estándar (sin límite arbitrario)
            k_max = int(np.ceil(mean + 10 * std))
            k_max = max(k_max, 20)
            return np.arange(0, k_max + 1)
        return np.arange(0, 100)
    elif dist_name == "Hipergeométrica":
        n = params['n']
        return np.arange(0, n + 1)
    return np.arange(0, 100)

# ===========================
# TÍTULO PRINCIPAL
# ===========================
st.title("🎓 Probabilidad y Estadística Interactiva")
st.markdown("**Explora conceptos estadísticos mediante simulaciones y cálculos interactivos**")

# ===========================
# SIDEBAR - NAVEGACIÓN
# ===========================
st.sidebar.title("📚 Navegación")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "🎯 Distribuciones Discretas",
        "📈 Distribuciones Continuas",
        "📊 Conceptos Estadísticos",
        "🔬 Inferencia Estadística"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("**Tip:** Ajusta los parámetros y observa cómo cambian las distribuciones y estadísticos.")

# ===========================
# SECCIÓN 1: DISTRIBUCIONES DISCRETAS
# ===========================
if seccion == "🎯 Distribuciones Discretas":
    st.header("🎯 Distribuciones Discretas")
    st.markdown("Simula y calcula probabilidades para distribuciones discretas")
    
    discrete_dist = st.selectbox(
        "Selecciona la distribución",
        ["Binomial", "Poisson", "Geométrica", "Binomial Negativa", "Hipergeométrica"]
    )
    
    tab1, tab2 = st.tabs(["📊 Simulación", "🔢 Calculadora"])
    
    # ===== TAB: SIMULACIÓN =====
    with tab1:
        st.subheader("🔧 Simulación")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        params = {}
        
        if discrete_dist == "Binomial":
            with col1:
                params['n'] = st.number_input("n (ensayos)", min_value=1, value=10, step=1)
            with col2:
                params['p'] = st.number_input("p (prob. éxito)", min_value=0.0, max_value=1.0, value=0.5, format="%.4f")
            with col3:
                num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data = np.random.binomial(int(params['n']), params['p'], num_sims)
            theo_mean = params['n'] * params['p']
            theo_var = params['n'] * params['p'] * (1 - params['p'])
            
        elif discrete_dist == "Poisson":
            with col1:
                params['lambda'] = st.number_input("λ (tasa)", min_value=0.01, value=5.0, format="%.4f")
            with col2:
                num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            with col3:
                st.markdown("—")
            
            sim_data = np.random.poisson(params['lambda'], num_sims)
            theo_mean = params['lambda']
            theo_var = params['lambda']
            
        elif discrete_dist == "Geométrica":
            with col1:
                params['p'] = st.number_input("p (prob. éxito)", min_value=0.001, max_value=1.0, value=0.2, format="%.4f")
            with col2:
                num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            with col3:
                st.markdown("—")
            
            sim_data = np.random.geometric(params['p'], num_sims)
            theo_mean = 1 / params['p']
            theo_var = (1 - params['p']) / params['p']**2
            
        elif discrete_dist == "Binomial Negativa":
            with col1:
                params['r'] = st.number_input("r (éxitos)", min_value=1, value=5, step=1)
            with col2:
                params['p'] = st.number_input("p (prob. éxito)", min_value=0.001, max_value=1.0, value=0.5, format="%.4f")
            with col3:
                num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data = np.random.negative_binomial(int(params['r']), params['p'], num_sims)
            theo_mean = params['r'] * (1 - params['p']) / params['p']
            theo_var = params['r'] * (1 - params['p']) / params['p']**2
            
        elif discrete_dist == "Hipergeométrica":
            with col1:
                params['N'] = st.number_input("N (población)", min_value=10, value=50, step=1)
            with col2:
                max_K = int(params['N']) - 1
                params['K'] = st.number_input("K (éxitos en pob.)", min_value=1, max_value=max_K, value=min(20, max_K), step=1)
            with col3:
                max_n = int(params['N'])
                params['n'] = st.number_input("n (muestra)", min_value=1, max_value=max_n, value=min(10, max_n), step=1)
            
            num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data = np.random.hypergeometric(int(params['K']), int(params['N'])-int(params['K']), int(params['n']), num_sims)
            theo_mean = params['n'] * params['K'] / params['N']
            theo_var = params['n'] * (params['K']/params['N']) * (1-params['K']/params['N']) * (params['N']-params['n'])/(params['N']-1)
        
        # Gráfico de simulación vs teórico
        k_range = get_discrete_range(discrete_dist, params, sim_data.max())
        
        # Calcular PMF teórica
        if discrete_dist == "Binomial":
            pmf_theo = binom.pmf(k_range, params['n'], params['p'])
        elif discrete_dist == "Poisson":
            pmf_theo = poisson.pmf(k_range, params['lambda'])
        elif discrete_dist == "Geométrica":
            pmf_theo = geom.pmf(k_range, params['p'])
        elif discrete_dist == "Binomial Negativa":
            pmf_theo = nbinom.pmf(k_range, params['r'], params['p'])
        elif discrete_dist == "Hipergeométrica":
            pmf_theo = hypergeom.pmf(k_range, params['N'], params['K'], params['n'])
        
        # Crear figura con Plotly
        fig = go.Figure()
        
        # PMF teórica
        fig.add_trace(go.Bar(
            x=k_range,
            y=pmf_theo,
            name='PMF teórica',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Histograma de simulación
        hist, bin_edges = np.histogram(sim_data, bins=np.arange(k_range.min(), k_range.max()+2)-0.5, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=hist,
            name='Frecuencias simuladas',
            marker_color='coral',
            opacity=0.6
        ))
        
        # Media simulada
        fig.add_vline(x=np.mean(sim_data), line_dash="dash", line_color="red",
                      annotation_text=f"Media sim: {np.mean(sim_data):.2f}")
        
        fig.update_layout(
            xaxis_title="k",
            yaxis_title="Probabilidad / Frecuencia",
            barmode='overlay',
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Resumen estadístico
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Media simulada", f"{np.mean(sim_data):.4f}")
        with col_b:
            st.metric("Varianza simulada", f"{np.var(sim_data):.4f}")
        with col_c:
            st.metric("Media teórica", f"{theo_mean:.4f}")
        with col_d:
            st.metric("Varianza teórica", f"{theo_var:.4f}")
    
    # ===== TAB: CALCULADORA =====
    with tab2:
        st.subheader("🔢 Calculadora (PMF y CDF)")
        
        calc_cols = st.columns([1, 1, 1])
        
        calc_params = {}
        
        if discrete_dist == "Binomial":
            with calc_cols[0]:
                calc_params['n'] = st.number_input("n (ensayos)", min_value=1, value=10, step=1, key="calc_n")
            with calc_cols[1]:
                calc_params['p'] = st.number_input("p (prob. éxito)", min_value=0.0, max_value=1.0, value=0.5, format="%.4f", key="calc_p")
            with calc_cols[2]:
                k_calc = st.number_input("k", min_value=0, max_value=int(calc_params['n']), value=int(calc_params['n']//2), step=1)
            
            pmf_val = binom.pmf(k_calc, int(calc_params['n']), calc_params['p'])
            cdf_val = binom.cdf(k_calc, int(calc_params['n']), calc_params['p'])
            
        elif discrete_dist == "Poisson":
            with calc_cols[0]:
                calc_params['lambda'] = st.number_input("λ (tasa)", min_value=0.01, value=5.0, format="%.4f", key="calc_lam")
            with calc_cols[1]:
                k_calc = st.number_input("k", min_value=0, value=int(calc_params['lambda']), step=1)
            with calc_cols[2]:
                st.markdown("—")
            
            pmf_val = poisson.pmf(k_calc, calc_params['lambda'])
            cdf_val = poisson.cdf(k_calc, calc_params['lambda'])
            
        elif discrete_dist == "Geométrica":
            with calc_cols[0]:
                calc_params['p'] = st.number_input("p (prob. éxito)", min_value=0.001, max_value=1.0, value=0.2, format="%.4f", key="calc_p_geom")
            with calc_cols[1]:
                k_calc = st.number_input("k", min_value=1, value=int(1/calc_params['p']), step=1)
            with calc_cols[2]:
                st.markdown("—")
            
            pmf_val = geom.pmf(k_calc, calc_params['p'])
            cdf_val = geom.cdf(k_calc, calc_params['p'])
            
        elif discrete_dist == "Binomial Negativa":
            with calc_cols[0]:
                calc_params['r'] = st.number_input("r (éxitos)", min_value=1, value=5, step=1, key="calc_r")
            with calc_cols[1]:
                calc_params['p'] = st.number_input("p (prob. éxito)", min_value=0.001, max_value=1.0, value=0.5, format="%.4f", key="calc_p_nb")
            with calc_cols[2]:
                mean_val = int(calc_params['r'] * (1-calc_params['p']) / calc_params['p'])
                k_calc = st.number_input("k", min_value=0, value=mean_val, step=1)
            
            pmf_val = nbinom.pmf(k_calc, int(calc_params['r']), calc_params['p'])
            cdf_val = nbinom.cdf(k_calc, int(calc_params['r']), calc_params['p'])
            
        elif discrete_dist == "Hipergeométrica":
            with calc_cols[0]:
                calc_params['N'] = st.number_input("N (población)", min_value=10, value=50, step=1, key="calc_N")
            with calc_cols[1]:
                calc_params['K'] = st.number_input("K (éxitos)", min_value=1, max_value=int(calc_params['N']), value=min(20, int(calc_params['N'])-1), step=1, key="calc_K")
            with calc_cols[2]:
                calc_params['n'] = st.number_input("n (muestra)", min_value=1, max_value=int(calc_params['N']), value=min(10, int(calc_params['N'])), step=1, key="calc_n_hyper")
            
            k_calc = st.number_input("k", min_value=0, max_value=int(calc_params['n']), value=min(5, int(calc_params['n'])), step=1, key="calc_k_hyper")
            
            pmf_val = hypergeom.pmf(k_calc, int(calc_params['N']), int(calc_params['K']), int(calc_params['n']))
            cdf_val = hypergeom.cdf(k_calc, int(calc_params['N']), int(calc_params['K']), int(calc_params['n']))
        
        # Mostrar resultados
        result_cols = st.columns(2)
        with result_cols[0]:
            st.metric(f"PMF: P(X = {k_calc})", f"{pmf_val:.6f}")
        with result_cols[1]:
            st.metric(f"CDF: P(X ≤ {k_calc})", f"{cdf_val:.6f}")

# ===========================
# SECCIÓN 2: DISTRIBUCIONES CONTINUAS
# ===========================
elif seccion == "📈 Distribuciones Continuas":
    st.header("📈 Distribuciones Continuas")
    st.markdown("Simula y calcula probabilidades para distribuciones continuas")
    
    continuous_dist = st.selectbox(
        "Selecciona la distribución",
        ["Normal", "Uniforme", "Exponencial", "Gamma", "Beta", "Weibull"]
    )
    
    tab1, tab2 = st.tabs(["📊 Simulación", "🔢 Calculadora"])
    
    # ===== TAB: SIMULACIÓN =====
    with tab1:
        st.subheader("🔧 Simulación")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        cont_params = {}
        
        if continuous_dist == "Normal":
            with col1:
                cont_params['mu'] = st.number_input("μ (media)", value=0.0, format="%.4f")
            with col2:
                cont_params['sigma'] = st.number_input("σ (desv. est.)", min_value=0.01, value=1.0, format="%.4f")
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.normal(cont_params['mu'], cont_params['sigma'], num_sims_cont)
            theo_mean_cont = cont_params['mu']
            theo_var_cont = cont_params['sigma']**2
            
        elif continuous_dist == "Uniforme":
            with col1:
                cont_params['a'] = st.number_input("a (mínimo)", value=0.0, format="%.4f")
            with col2:
                cont_params['b'] = st.number_input("b (máximo)", min_value=cont_params['a']+0.01, value=cont_params['a']+1.0, format="%.4f")
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.uniform(cont_params['a'], cont_params['b'], num_sims_cont)
            theo_mean_cont = (cont_params['a'] + cont_params['b']) / 2
            theo_var_cont = ((cont_params['b'] - cont_params['a'])**2) / 12
            
        elif continuous_dist == "Exponencial":
            with col1:
                cont_params['lambda'] = st.number_input("λ (tasa)", min_value=0.01, value=1.0, format="%.4f")
            with col2:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            with col3:
                st.markdown("—")
            
            sim_data_cont = np.random.exponential(1/cont_params['lambda'], num_sims_cont)
            theo_mean_cont = 1 / cont_params['lambda']
            theo_var_cont = 1 / cont_params['lambda']**2
            
        elif continuous_dist == "Gamma":
            with col1:
                cont_params['shape'] = st.number_input("α (forma)", min_value=0.01, value=2.0, format="%.4f")
            with col2:
                cont_params['scale'] = st.number_input("β (escala)", min_value=0.01, value=1.0, format="%.4f")
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.gamma(cont_params['shape'], cont_params['scale'], num_sims_cont)
            theo_mean_cont = cont_params['shape'] * cont_params['scale']
            theo_var_cont = cont_params['shape'] * cont_params['scale']**2
            
        elif continuous_dist == "Beta":
            with col1:
                cont_params['alpha'] = st.number_input("α", min_value=0.01, value=2.0, format="%.4f")
            with col2:
                cont_params['beta'] = st.number_input("β", min_value=0.01, value=2.0, format="%.4f")
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.beta(cont_params['alpha'], cont_params['beta'], num_sims_cont)
            alpha, beta_p = cont_params['alpha'], cont_params['beta']
            theo_mean_cont = alpha / (alpha + beta_p)
            theo_var_cont = (alpha * beta_p) / ((alpha + beta_p)**2 * (alpha + beta_p + 1))
            
        elif continuous_dist == "Weibull":
            with col1:
                cont_params['shape'] = st.number_input("k (forma)", min_value=0.01, value=1.5, format="%.4f")
            with col2:
                cont_params['scale'] = st.number_input("λ (escala)", min_value=0.01, value=1.0, format="%.4f")
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.weibull(cont_params['shape'], num_sims_cont) * cont_params['scale']
            k = cont_params['shape']
            lam = cont_params['scale']
            theo_mean_cont = lam * gamma_func(1 + 1/k)
            theo_var_cont = lam**2 * (gamma_func(1 + 2/k) - gamma_func(1 + 1/k)**2)
        
        # Gráfico de simulación vs teórico
        fig_cont = go.Figure()
        
        # Histograma de simulación
        fig_cont.add_trace(go.Histogram(
            x=sim_data_cont,
            name='Datos simulados',
            histnorm='probability density',
            marker_color='coral',
            opacity=0.6,
            nbinsx=50
        ))
        
        # PDF teórica
        x_range = np.linspace(sim_data_cont.min(), sim_data_cont.max(), 500)
        
        if continuous_dist == "Normal":
            pdf_theo = norm.pdf(x_range, cont_params['mu'], cont_params['sigma'])
        elif continuous_dist == "Uniforme":
            pdf_theo = uniform.pdf(x_range, cont_params['a'], cont_params['b']-cont_params['a'])
        elif continuous_dist == "Exponencial":
            pdf_theo = expon.pdf(x_range, scale=1/cont_params['lambda'])
        elif continuous_dist == "Gamma":
            pdf_theo = gamma.pdf(x_range, cont_params['shape'], scale=cont_params['scale'])
        elif continuous_dist == "Beta":
            pdf_theo = beta.pdf(x_range, cont_params['alpha'], cont_params['beta'])
        elif continuous_dist == "Weibull":
            pdf_theo = weibull_min.pdf(x_range, cont_params['shape'], scale=cont_params['scale'])
        
        fig_cont.add_trace(go.Scatter(
            x=x_range,
            y=pdf_theo,
            name='PDF teórica',
            line=dict(color='blue', width=2)
        ))
        
        # Media simulada
        fig_cont.add_vline(x=np.mean(sim_data_cont), line_dash="dash", line_color="red",
                           annotation_text=f"Media sim: {np.mean(sim_data_cont):.2f}")
        
        fig_cont.update_layout(
            xaxis_title="x",
            yaxis_title="Densidad",
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cont, use_container_width=True)
        
        # Resumen estadístico
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Media simulada", f"{np.mean(sim_data_cont):.4f}")
        with col_b:
            st.metric("Varianza simulada", f"{np.var(sim_data_cont):.4f}")
        with col_c:
            st.metric("Media teórica", f"{theo_mean_cont:.4f}")
        with col_d:
            st.metric("Varianza teórica", f"{theo_var_cont:.4f}")
    
    # ===== TAB: CALCULADORA =====
    with tab2:
        st.subheader("🔢 Calculadora (Área bajo la curva)")
        
        calc_cols_cont = st.columns([1, 1, 1])
        
        calc_params_cont = {}
        
        if continuous_dist == "Normal":
            with calc_cols_cont[0]:
                calc_params_cont['mu'] = st.number_input("μ (media)", value=0.0, format="%.4f", key="calc_mu")
            with calc_cols_cont[1]:
                calc_params_cont['sigma'] = st.number_input("σ (desv. est.)", min_value=0.01, value=1.0, format="%.4f", key="calc_sigma")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = calc_params_cont['mu'] - calc_params_cont['sigma']
            default_b = calc_params_cont['mu'] + calc_params_cont['sigma']
            
        elif continuous_dist == "Uniforme":
            with calc_cols_cont[0]:
                calc_params_cont['a'] = st.number_input("a (mínimo)", value=0.0, format="%.4f", key="calc_a")
            with calc_cols_cont[1]:
                calc_params_cont['b'] = st.number_input("b (máximo)", min_value=calc_params_cont['a']+0.01, value=calc_params_cont['a']+1.0, format="%.4f", key="calc_b")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = calc_params_cont['a']
            default_b = calc_params_cont['b']
            
        elif continuous_dist == "Exponencial":
            with calc_cols_cont[0]:
                calc_params_cont['lambda'] = st.number_input("λ (tasa)", min_value=0.01, value=1.0, format="%.4f", key="calc_lambda")
            with calc_cols_cont[1]:
                st.markdown("—")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = 0
            default_b = 2 / calc_params_cont['lambda']
            
        elif continuous_dist == "Gamma":
            with calc_cols_cont[0]:
                calc_params_cont['shape'] = st.number_input("α (forma)", min_value=0.01, value=2.0, format="%.4f", key="calc_shape_gamma")
            with calc_cols_cont[1]:
                calc_params_cont['scale'] = st.number_input("β (escala)", min_value=0.01, value=1.0, format="%.4f", key="calc_scale_gamma")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = 0
            default_b = calc_params_cont['shape'] * calc_params_cont['scale']
            
        elif continuous_dist == "Beta":
            with calc_cols_cont[0]:
                calc_params_cont['alpha'] = st.number_input("α", min_value=0.01, value=2.0, format="%.4f", key="calc_alpha")
            with calc_cols_cont[1]:
                calc_params_cont['beta'] = st.number_input("β", min_value=0.01, value=2.0, format="%.4f", key="calc_beta")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = 0.2
            default_b = 0.8
            
        elif continuous_dist == "Weibull":
            with calc_cols_cont[0]:
                calc_params_cont['shape'] = st.number_input("k (forma)", min_value=0.01, value=1.5, format="%.4f", key="calc_shape_weib")
            with calc_cols_cont[1]:
                calc_params_cont['scale'] = st.number_input("λ (escala)", min_value=0.01, value=1.0, format="%.4f", key="calc_scale_weib")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = 0
            default_b = calc_params_cont['scale']
        
        # Entrada del intervalo
        int_cols = st.columns(2)
        with int_cols[0]:
            a_val = st.number_input("a (límite inferior)", value=float(default_a), format="%.4f")
        with int_cols[1]:
            b_val = st.number_input("b (límite superior)", value=float(default_b), format="%.4f")
        
        # Calcular probabilidad
        if continuous_dist == "Normal":
            prob = norm.cdf(b_val, calc_params_cont['mu'], calc_params_cont['sigma']) - norm.cdf(a_val, calc_params_cont['mu'], calc_params_cont['sigma'])
            x_plot = np.linspace(calc_params_cont['mu']-4*calc_params_cont['sigma'], calc_params_cont['mu']+4*calc_params_cont['sigma'], 500)
            y_plot = norm.pdf(x_plot, calc_params_cont['mu'], calc_params_cont['sigma'])
        elif continuous_dist == "Uniforme":
            prob = uniform.cdf(b_val, calc_params_cont['a'], calc_params_cont['b']-calc_params_cont['a']) - uniform.cdf(a_val, calc_params_cont['a'], calc_params_cont['b']-calc_params_cont['a'])
            x_plot = np.linspace(calc_params_cont['a']-1, calc_params_cont['b']+1, 500)
            y_plot = uniform.pdf(x_plot, calc_params_cont['a'], calc_params_cont['b']-calc_params_cont['a'])
        elif continuous_dist == "Exponencial":
            prob = expon.cdf(b_val, scale=1/calc_params_cont['lambda']) - expon.cdf(a_val, scale=1/calc_params_cont['lambda'])
            x_plot = np.linspace(0, 5/calc_params_cont['lambda'], 500)
            y_plot = expon.pdf(x_plot, scale=1/calc_params_cont['lambda'])
        elif continuous_dist == "Gamma":
            prob = gamma.cdf(b_val, calc_params_cont['shape'], scale=calc_params_cont['scale']) - gamma.cdf(a_val, calc_params_cont['shape'], scale=calc_params_cont['scale'])
            x_plot = np.linspace(0, calc_params_cont['shape']*calc_params_cont['scale']+4*np.sqrt(calc_params_cont['shape'])*calc_params_cont['scale'], 500)
            y_plot = gamma.pdf(x_plot, calc_params_cont['shape'], scale=calc_params_cont['scale'])
        elif continuous_dist == "Beta":
            prob = beta.cdf(b_val, calc_params_cont['alpha'], calc_params_cont['beta']) - beta.cdf(a_val, calc_params_cont['alpha'], calc_params_cont['beta'])
            x_plot = np.linspace(0, 1, 500)
            y_plot = beta.pdf(x_plot, calc_params_cont['alpha'], calc_params_cont['beta'])
        elif continuous_dist == "Weibull":
            prob = weibull_min.cdf(b_val, calc_params_cont['shape'], scale=calc_params_cont['scale']) - weibull_min.cdf(a_val, calc_params_cont['shape'], scale=calc_params_cont['scale'])
            x_plot = np.linspace(0, calc_params_cont['scale']*3, 500)
            y_plot = weibull_min.pdf(x_plot, calc_params_cont['shape'], scale=calc_params_cont['scale'])
        
        # Mostrar resultado
        st.metric(f"P({a_val:.4f} ≤ X ≤ {b_val:.4f})", f"{prob:.6f}")
        
        # Gráfico con área sombreada
        fig_area = go.Figure()
        
        # PDF completa
        fig_area.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            name='PDF',
            line=dict(color='blue', width=2),
            fill=None
        ))
        
        # Área sombreada
        x_fill = x_plot[(x_plot >= a_val) & (x_plot <= b_val)]
        y_fill = y_plot[(x_plot >= a_val) & (x_plot <= b_val)]
        
        if len(x_fill) > 0:
            fig_area.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='tozeroy',
                name=f'P({a_val:.2f} ≤ X ≤ {b_val:.2f})',
                line=dict(color='red'),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
        
        fig_area.update_layout(
            xaxis_title="x",
            yaxis_title="Densidad",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_area, use_container_width=True)

# ===========================
# SECCIÓN 3: CONCEPTOS ESTADÍSTICOS
# ===========================
elif seccion == "📊 Conceptos Estadísticos":
    st.header("📊 Conceptos Estadísticos Fundamentales")
    
    concepto = st.selectbox(
        "Selecciona un concepto",
        ["PCA y Regresión", "t-SNE", "Histogramas y Bines", "Correlación", "Ley de Grandes Números", "Teorema Límite Central", "Exactitud y Precisión"]
    )
    
    # ===== PCA Y REGRESIÓN =====
    if concepto == "PCA y Regresión":
        st.subheader("🎯 PCA, Regresión, Covarianza, KDE y Elipse")
        st.markdown("""
        Esta herramienta genera **datos bivariados correlacionados** y visualiza:
        - 🔵 Nube de puntos con datos correlacionados
        - 📈 Línea de regresión lineal
        - ➡️ Componentes principales (PCA)
        - ⭕ Elipse de covarianza (95% confianza)
        - 📊 Densidades marginales (KDE) de X e Y
        
        **¿Qué es PCA?** El Análisis de Componentes Principales encuentra las direcciones de máxima varianza en los datos.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            rho = st.slider("Coeficiente de correlación ρ", -0.999, 0.999, -0.9, 0.01)
        with col2:
            n = st.number_input("Número de puntos", min_value=100, max_value=10000, value=500, step=100)
        
        # Generate correlated data
        np.random.seed(0)
        mean = [0, 0]
        cov = [[1, rho],
               [rho, 1]]
        
        data = np.random.multivariate_normal(mean, cov, int(n))
        x, y = data[:,0], data[:,1]
        
        # Covariance and PCA
        C = np.cov(x, y)
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        origin = np.mean(data, axis=0)
        
        # Regression line
        beta = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 300)
        y_line = beta[0] * x_line + beta[1]
        
        # Covariance ellipse (95%)
        chi2_val = 5.991  # 95% confidence for 2D
        width = 2 * np.sqrt(eigvals[0] * chi2_val)
        height = 2 * np.sqrt(eigvals[1] * chi2_val)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        
        # KDE marginal densities
        kde_x = gaussian_kde(x)
        kde_y = gaussian_kde(y)
        
        x_grid = np.linspace(min(x), max(x), 300)
        y_grid = np.linspace(min(y), max(y), 300)
        
        # Create main figure
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Datos',
            marker=dict(size=4, color='steelblue', opacity=0.5),
            showlegend=True
        ))
        
        # Regression line
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Regresión lineal',
            line=dict(color='red', width=2),
            showlegend=True
        ))
        
        # PCA components (eigenvectors) - usando annotations para flechas
        colors = ['green', 'orange']
        annotations = []
        for i in range(2):
            vec = eigvecs[:, i] * np.sqrt(eigvals[i]) * 3
            fig.add_trace(go.Scatter(
                x=[origin[0], origin[0] + vec[0]],
                y=[origin[1], origin[1] + vec[1]],
                mode='lines',
                name=f'PC{i+1} (λ={eigvals[i]:.3f})',
                line=dict(color=colors[i], width=3),
                showlegend=True
            ))
            # Agregar anotación con flecha al final del vector
            annotations.append(
                dict(
                    ax=origin[0], ay=origin[1],
                    x=origin[0] + vec[0], y=origin[1] + vec[1],
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=colors[i],
                    opacity=0.8
                )
            )
        
        # Covariance ellipse (95%)
        t = np.linspace(0, 2*np.pi, 400)
        ellipse_x = origin[0] + (width/2)*np.cos(t)*np.cos(np.radians(angle)) - (height/2)*np.sin(t)*np.sin(np.radians(angle))
        ellipse_y = origin[1] + (width/2)*np.cos(t)*np.sin(np.radians(angle)) + (height/2)*np.sin(t)*np.cos(np.radians(angle))
        
        fig.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode='lines',
            name='Elipse 95%',
            line=dict(color='purple', width=2, dash='dash'),
            showlegend=True
        ))
        
        # Marginal KDE for X (shifted down)
        kde_x_vals = kde_x(x_grid)
        shift = min(y) - 0.5  # Shift below the data
        fig.add_trace(go.Scatter(
            x=x_grid,
            y=kde_x_vals + shift,
            mode='lines',
            name='KDE marginal X',
            line=dict(color='cyan', width=2),
            #fill='tozeroy',
            #fillcolor='rgba(0, 255, 255, 0.2)',
            showlegend=True
        ))
        
        # Marginal KDE for Y (shifted left)
        kde_y_vals = kde_y(y_grid)
        shift = min(x) - 0.5  # Shift to the left of the data
        fig.add_trace(go.Scatter(
            x=kde_y_vals + shift,
            y=y_grid,
            mode='lines',
            name='KDE marginal Y',
            line=dict(color='magenta', width=2),
            #fill='tozerox',
            #fillcolor='rgba(255, 0, 255, 0.2)',
            showlegend=True
        ))
        
        # IMPORTANTE: Ejes con la misma escala
        axis_range = [min(min(x), min(y)) - 1, max(max(x), max(y)) + 1]
        
        fig.update_layout(
            title=f"PCA, Regresión, Covarianza, KDE y Elipse (ρ = {rho:.2f})",
            xaxis_title="X",
            yaxis_title="Y",
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                range=axis_range
            ),
            yaxis=dict(
                range=axis_range
            ),
            height=600,
            showlegend=True,
            hovermode='closest',
            annotations=annotations
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar estadísticas
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Correlación muestral", f"{np.corrcoef(x, y)[0, 1]:.4f}")
        with col_b:
            st.metric("Varianza explicada PC1", f"{eigvals[0]/(eigvals[0]+eigvals[1])*100:.2f}%")
        with col_c:
            st.metric("Pendiente regresión", f"{beta[0]:.4f}")
        
        st.info("""
        **💡 Interpretación:**
        - **PC1 (verde):** Dirección de máxima varianza
        - **PC2 (naranja):** Dirección perpendicular a PC1
        - **Elipse morada:** Contiene ~95% de los datos
        - **KDE marginal:** Muestra la distribución de cada variable individualmente
        """)
    
    # ===== t-SNE =====
    elif concepto == "t-SNE":
        st.subheader("🔮 t-SNE (t-Distributed Stochastic Neighbor Embedding)")
        st.markdown("""
        **t-SNE** es un algoritmo de reducción de dimensionalidad que preserva las distancias locales.
        Es especialmente útil para visualizar datos de alta dimensión en 2D o 3D.
        
        **¿Cómo funciona?**
        1. Calcula similitudes entre puntos en el espacio original (distribución Gaussiana)
        2. Inicializa puntos aleatoriamente en el espacio reducido
        3. Ajusta las posiciones para que las similitudes se preserven (distribución t-Student)
        4. Itera minimizando la divergencia KL entre las distribuciones
        """)
        
        # Parámetros de control
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_samples_per_class = st.slider("Muestras por dígito", 10, 100, 30, 10)
        with col2:
            perplexity_val = st.slider("Perplexity", 5, 50, 30, 5)
        with col3:
            lr_val = st.slider("Learning rate", 10, 1000, 200, 50)
        
        col4, col5 = st.columns(2)
        with col4:
            n_iterations = st.slider("Iteraciones totales", 250, 2000, 1000, 250)
        with col5:
            n_frames = st.slider("Número de frames a capturar", 4, 16, 9, 1)
        
        # Cargar MNIST digits desde sklearn
        with st.spinner('Cargando MNIST digits...'):
            from sklearn.datasets import load_digits
            from sklearn.manifold import TSNE
            digits = load_digits()
            X_full = digits.data  # 1797 samples, 64 features (8x8 images)
            y_full = digits.target  # Labels 0-9
        
        # Muestreo balanceado por clase
        np.random.seed(42)
        X_samples = []
        y_samples = []
        
        for digit in range(10):
            # Obtener índices de este dígito
            indices = np.where(y_full == digit)[0]
            # Muestrear aleatoriamente n_samples_per_class
            sampled_indices = np.random.choice(indices, size=min(n_samples_per_class, len(indices)), replace=False)
            X_samples.append(X_full[sampled_indices])
            y_samples.append(y_full[sampled_indices])
        
        X = np.vstack(X_samples)
        y = np.concatenate(y_samples)
        
        # Colores para cada dígito (paleta de 10 colores)
        color_palette = [
            '#1f77b4',  # 0: azul
            '#ff7f0e',  # 1: naranja
            '#2ca02c',  # 2: verde
            '#d62728',  # 3: rojo
            '#9467bd',  # 4: púrpura
            '#8c564b',  # 5: marrón
            '#e377c2',  # 6: rosa
            '#7f7f7f',  # 7: gris
            '#bcbd22',  # 8: amarillo-verde
            '#17becf'   # 9: cian
        ]
        colors = [color_palette[label] for label in y]
        
        st.success(f"✅ Cargados {len(X)} dígitos MNIST ({n_samples_per_class} por clase, clases 0-9)")
        
        # Ejecutar t-SNE con sklearn capturando frames intermedios
        with st.spinner('Ejecutando t-SNE...'):
            frames = []
            kl_divergences = []
            
            # Calcular en cuántas iteraciones capturar cada frame
            # Asegurar que todas las iteraciones sean >= 250 (mínimo de sklearn)
            capture_iterations = np.linspace(250, n_iterations, n_frames, dtype=int)
            
            for i, n_iter in enumerate(capture_iterations):
                # Ejecutar t-SNE hasta n_iter iteraciones
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity_val,
                    learning_rate=lr_val,
                    n_iter=n_iter,
                    random_state=42,
                    init='random',
                    method='barnes_hut',
                    verbose=0
                )
                Y_current = tsne.fit_transform(X)
                frames.append(Y_current.copy())
                kl_divergences.append(tsne.kl_divergence_)
            
            Y_final = frames[-1]
            kl_divergence_final = kl_divergences[-1]
        
        st.success(f"✅ t-SNE completado con {len(frames)} frames capturados (KL final: {kl_divergence_final:.4f})")
        
        # Crear paneles con subplots
        n_frames_display = len(frames)
        
        # Determinar layout de subplots
        if n_frames_display <= 3:
            rows, cols = 1, n_frames_display
        elif n_frames_display <= 6:
            rows, cols = 2, 3
        elif n_frames_display <= 9:
            rows, cols = 3, 3
        elif n_frames_display <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
            frames = frames[:16]  # Limitar a 16 frames
            n_frames_display = 16
        
        # Crear subplots
        subplot_titles = [f"Iter {capture_iterations[i]}" for i in range(n_frames_display)]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # Agregar cada frame a su subplot
        for idx, frame in enumerate(frames):
            row = idx // cols + 1
            col = idx % cols + 1
            
            fig.add_trace(
                go.Scatter(
                    x=frame[:, 0],
                    y=frame[:, 1],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors,
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f"Dígito {label}" for label in y],
                    hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Hacer ejes cuadrados (misma escala)
            fig.update_xaxes(scaleanchor=f"y{idx+1}", scaleratio=1, row=row, col=col)
            fig.update_yaxes(row=row, col=col)
        
        fig.update_layout(
            title_text=f"Evolución de t-SNE en MNIST (perplexity={perplexity_val}, lr={lr_val})",
            height=200 * rows,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Leyenda de colores
        st.markdown("### 🎨 Leyenda de Dígitos")
        legend_cols = st.columns(10)
        for i, (digit, color) in enumerate(zip(range(10), color_palette)):
            with legend_cols[i]:
                st.markdown(f"<div style='text-align: center;'><span style='color: {color}; font-size: 24px;'>●</span><br><b>{digit}</b></div>", unsafe_allow_html=True)
        
        # ===== GRÁFICO DE DIVERGENCIA KL =====
        st.subheader("📉 Convergencia: Divergencia KL vs Iteraciones")
        
        fig_kl = go.Figure()
        
        fig_kl.add_trace(go.Scatter(
            x=capture_iterations.tolist(),
            y=kl_divergences,
            mode='lines+markers',
            name='KL Divergence',
            line=dict(color='crimson', width=2),
            marker=dict(size=8, color='blue', symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(220, 20, 60, 0.2)'
        ))
        
        fig_kl.update_layout(
            xaxis_title="Iteración",
            yaxis_title="Divergencia KL(P||Q) [escala log]",
            yaxis_type="log",  # ⭐ ESCALA LOGARÍTMICA
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        
        st.plotly_chart(fig_kl, use_container_width=True)
        
        # Métricas de convergencia
        col_kl1, col_kl2, col_kl3 = st.columns(3)
        with col_kl1:
            st.metric("KL Inicial", f"{kl_divergences[0]:.4f}")
        with col_kl2:
            st.metric("KL Final", f"{kl_divergences[-1]:.4f}")
        with col_kl3:
            reduction = (kl_divergences[0] - kl_divergences[-1]) / kl_divergences[0] * 100
            st.metric("Reducción", f"{reduction:.2f}%", delta=f"-{reduction:.2f}%")
        
        st.info("""
        **📉 Interpretación de la Divergencia KL (escala logarítmica):**
        - **Escala log:** Permite ver tanto valores altos iniciales como bajos finales en un mismo gráfico
        - **Inicio alto (>2.0):** Las distribuciones P y Q son muy diferentes
        - **Descenso rápido:** El algoritmo está aprendiendo la estructura (fase de "early exaggeration")
        - **Descenso gradual:** Optimización y refinamiento de clusters
        - **Plateau final (<0.5):** Convergencia alcanzada - diferencia mínima entre P y Q
        - **Puntos azules:** Momentos donde se capturaron los frames visualizados arriba
        
        💡 En escala log, una línea recta indica descenso exponencial (muy bueno).
        """)
        
        # Comparación lado a lado: Original vs Final
        st.subheader("📊 Comparación: Datos Originales (64D) vs t-SNE (2D)")
        
        fig_compare = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Primeras 2 componentes originales", "t-SNE embedding (2D)"),
            horizontal_spacing=0.1
        )
        
        # Original (solo primeras 2 dimensiones para visualizar)
        fig_compare.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode='markers',
                marker=dict(size=8, color=colors, opacity=0.7, line=dict(width=0.5, color='white')),
                text=[f"Dígito {label}" for label in y],
                hovertemplate='%{text}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<extra></extra>',
                name='Original',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # t-SNE Final
        fig_compare.add_trace(
            go.Scatter(
                x=Y_final[:, 0],
                y=Y_final[:, 1],
                mode='markers',
                marker=dict(size=8, color=colors, opacity=0.7, line=dict(width=0.5, color='white')),
                text=[f"Dígito {label}" for label in y],
                hovertemplate='%{text}<br>t-SNE1: %{x:.2f}<br>t-SNE2: %{y:.2f}<extra></extra>',
                name='t-SNE',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Ejes con misma escala
        fig_compare.update_xaxes(title_text="Componente 1", scaleanchor="y", scaleratio=1, row=1, col=1)
        fig_compare.update_yaxes(title_text="Componente 2", row=1, col=1)
        fig_compare.update_xaxes(title_text="t-SNE dimensión 1", scaleanchor="y2", scaleratio=1, row=1, col=2)
        fig_compare.update_yaxes(title_text="t-SNE dimensión 2", row=1, col=2)
        
        fig_compare.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Explicación
        col_a, col_b = st.columns(2)
        with col_a:
            st.info("""
            **📊 Datos Originales (64D → mostrando 2D):**
            - Cada dígito MNIST es una imagen 8×8 = 64 píxeles
            - Aquí solo vemos las primeras 2 dimensiones
            - **Difícil ver la estructura de clusters**
            """)
        with col_b:
            st.success("""
            **🎯 t-SNE (64D → 2D):**
            - Reduce 64 dimensiones a 2
            - Preserva distancias locales (vecinos cercanos)
            - **Dígitos similares quedan juntos**
            - Cada color = un dígito (0-9)
            """)
        
        # Mostrar algunos dígitos de ejemplo
        st.subheader("🔢 Ejemplos de Dígitos MNIST")
        
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        
        # Seleccionar 10 ejemplos aleatorios (uno de cada clase)
        example_indices = []
        for digit in range(10):
            indices = np.where(y == digit)[0]
            if len(indices) > 0:
                example_indices.append(indices[0])
        
        # Crear figura con ejemplos
        fig_examples, axes = plt.subplots(1, 10, figsize=(12, 1.5))
        for idx, ax in enumerate(axes):
            if idx < len(example_indices):
                img_idx = example_indices[idx]
                img = X[img_idx].reshape(8, 8)
                ax.imshow(img, cmap='gray_r')
                ax.set_title(f"{y[img_idx]}", fontsize=14, fontweight='bold')
                ax.axis('off')
        
        plt.tight_layout()
        
        # Convertir a imagen para Streamlit
        buf = BytesIO()
        fig_examples.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        st.image(buf, caption="Ejemplos de cada dígito (0-9) en el dataset", use_container_width=True)
        plt.close(fig_examples)
        
        st.warning("""
        **⚙️ Ajusta los parámetros:**
        - **Muestras por dígito:** Más muestras = más representativo pero más lento
        - **Perplexity (5-50):** Balancea atención local vs global (típico: 30)
          - Bajo (5-15): Foco muy local, clusters compactos
          - Alto (30-50): Considera más vecinos, estructura global
        - **Learning rate (10-1000):** Velocidad de optimización (típico: 200)
          - Bajo (10-100): Lento pero estable
          - Alto (500-1000): Rápido pero puede oscilar
        - **Iteraciones (250-2000):** Más iteraciones = mejor convergencia (típico: 1000)
        
        **🎯 Experimenta:**
        - ¿Algunos dígitos se superponen? → Aumenta perplexity o iteraciones
        - ¿Dígitos similares (3-8, 4-9) están cerca? → ¡Es correcto! t-SNE preserva similitud visual
        - ¿Ves outliers? → Pueden ser dígitos mal escritos o ambiguos
        - ¿KL no converge? → Aumenta iteraciones o ajusta learning rate
        """)
    
    # ===== HISTOGRAMAS Y BINES =====
    elif concepto == "Histogramas y Bines":
        st.subheader("📊 Número de Bines en un Histograma")
        st.markdown("""
        No hay un número "mejor" de bines. Diferentes números pueden revelar diferentes características de los datos.
        Experimenta con el control deslizante para encontrar el número adecuado.
        """)
        
        np.random.seed(123)
        size = 100
        values = expon.rvs(scale=1/4, size=size)
        noise = norm.rvs(loc=1, size=size)
        real_data = values + np.abs(noise)
        
        b = st.slider("Número de bines", 1, 100, 4, key="bins_slider")
        
        fig = go.Figure()
        
        # Histograma
        fig.add_trace(go.Histogram(
            x=real_data,
            nbinsx=b,
            histnorm='probability density',
            name='Histograma',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # KDE
        kde = gaussian_kde(real_data)
        x_range = np.linspace(real_data.min(), real_data.max(), 200)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode='lines',
            name='KDE',
            line=dict(color='crimson', width=2)
        ))
        
        fig.update_layout(
            title=f"Histograma con {b} bines",
            xaxis_title="X",
            yaxis_title="Densidad",
            height=450,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if b in [3, 4, 5, 6]:
            st.success("✅ Buen número de bines - se observa claramente la distribución")
        elif b < 3:
            st.warning("⚠️ Muy pocos bines - se pierde detalle")
        elif b > 20:
            st.warning("⚠️ Demasiados bines - puede haber ruido")
    
    # ===== CORRELACIÓN =====
    elif concepto == "Correlación":
        st.subheader("🔗 Explorando la Correlación")
        st.markdown("Observa cómo cambia la relación entre dos variables según el coeficiente de correlación")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            c = st.select_slider("Correlación", options=[-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                                                         0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], value=0.0)
        with col2:
            n = st.select_slider("Tamaño de muestra", options=list(range(10, 101, 10)) + [200, 500, 1000], value=100)
        with col3:
            l = st.checkbox("Regresión lineal", value=True)
        
        np.random.seed(123456)
        mean = [0, 0]
        cov = [[1, c], [c, 1]]
        x, y = np.random.multivariate_normal(mean, cov, n).T
        
        fig = px.scatter(x=x, y=y, trendline="ols" if l else None,
                         labels={'x': 'X', 'y': 'Y'},
                         title=f'Correlación = {c}, n = {n}')
        
        fig.update_traces(marker=dict(size=8, opacity=0.6, color='seagreen'))
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcular correlación real
        corr_real = np.corrcoef(x, y)[0, 1]
        st.metric("Correlación muestral", f"{corr_real:.4f}")
    
    # ===== LEY DE GRANDES NÚMEROS =====
    elif concepto == "Ley de Grandes Números":
        st.subheader("📈 Ley de los Grandes Números")
        st.markdown("""
        A medida que el tamaño de muestra crece, la media muestral converge a la media poblacional
        y la varianza de la media muestral converge a cero.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n = st.select_slider("Tamaño de muestra", options=[1, 2, 5, 10, 30, 100, 1000, 10000], value=10)
        with col2:
            dist = st.selectbox("Distribución", ["Uniforme", "Exponencial"])
        with col3:
            a = st.select_slider("Transparencia", options=[0.2, 0.5, 0.8, 1.0], value=0.2)
        
        simulations = 1000
        if dist == "Uniforme":
            X_bar = np.random.random(size=(simulations, n)).mean(axis=1)
            mu, sigma2 = 1/2, 1/12
            ll, ul = 0, 1
        else:
            X_bar = expon.rvs(scale=2, size=(simulations, n)).mean(axis=1)
            mu, sigma2 = 2, 4
            ll, ul = 0, 10
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Media Muestral vs Simulaciones', 'Distribución de Medias'),
            vertical_spacing=0.15
        )
        
        # Scatter plot de medias
        fig.add_trace(
            go.Scatter(
                x=X_bar,
                y=list(range(simulations)),
                mode='markers',
                name='X̄',
                marker=dict(size=4, opacity=a, color='blue')
            ),
            row=1, col=1
        )
        
        # Línea de media poblacional
        fig.add_vline(
            x=mu,
            line=dict(color='red', width=2),
            annotation_text='μ',
            row=1, col=1
        )
        
        # Histograma de medias
        fig.add_trace(
            go.Histogram(
                x=X_bar,
                name='Distribución',
                marker_color='lightblue',
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(range=[ll, ul], row=1, col=1)
        fig.update_xaxes(range=[ll, ul], title_text="X", row=2, col=1)
        fig.update_yaxes(title_text="Simulación", row=1, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Media muestral:** {np.mean(X_bar):.4f} | **Media poblacional (μ):** {mu:.4f}")
        st.write(f"**Varianza de X̄:** {np.var(X_bar):.4f} | **Varianza teórica:** {sigma2/n:.4f}")
    
    # ===== TEOREMA DEL LÍMITE CENTRAL =====
    elif concepto == "Teorema Límite Central":
        st.subheader("🎯 Teorema del Límite Central")
        st.markdown("""
        Para un tamaño de muestra suficientemente grande, la distribución de la media muestral
        se aproxima a una distribución normal, independientemente de la distribución original.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n = st.selectbox("Tamaño de muestra", [1, 2, 5, 10, 30, 100], index=2)
        with col2:
            dist = st.selectbox("Distribución original", ["Uniforme", "Exponencial"])
        
        simulations = 10000
        if dist == "Uniforme":
            X_bar = np.random.random(size=(simulations, n)).mean(axis=1)
            mu, sigma2 = 1/2, 1/12
        else:
            X_bar = expon.rvs(scale=2, size=(simulations, n)).mean(axis=1)
            mu, sigma2 = 2, 4
        
        fig = go.Figure()
        
        # Histograma de medias muestrales
        fig.add_trace(go.Histogram(
            x=X_bar,
            nbinsx=40,
            histnorm='probability density',
            name='X̄ (simulado)',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Curva normal teórica
        xl, xu = X_bar.min(), X_bar.max()
        x = np.linspace(xl, xu, 500)
        fig.add_trace(go.Scatter(
            x=x,
            y=norm.pdf(x, loc=mu, scale=np.sqrt(sigma2/n)),
            mode='lines',
            name='N(μ, σ²/√n)',
            line=dict(color='crimson', width=3)
        ))
        
        fig.update_layout(
            title=f'Teorema del Límite Central: {dist}, n={n}',
            xaxis_title='X̄',
            yaxis_title='Densidad',
            height=450,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if n >= 30:
            st.success("✅ Con n≥30, la aproximación normal es excelente")
        elif n >= 10:
            st.info("ℹ️ Con n≥10, la aproximación normal es razonable")
        else:
            st.warning("⚠️ Con n<10, la aproximación normal puede no ser buena")
    
    # ===== EXACTITUD Y PRECISIÓN =====
    elif concepto == "Exactitud y Precisión":
        st.subheader("🎯 Exactitud y Precisión de un Estimador")
        st.markdown("""
        - **Exactitud (Accuracy):** Qué tan cerca está el estimador del valor real (sesgo bajo)
        - **Precisión (Precision):** Qué tan consistentes son las estimaciones (varianza baja)
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n = st.selectbox("Número de dardos", [10, 20, 30, 50, 100], index=1)
        with col2:
            e = st.selectbox("Exactitud", ["Alta", "Media", "Baja"])
        with col3:
            p = st.selectbox("Precisión", ["Alta", "Media", "Baja"])
        
        smx, smy = [np.random.choice([-1, 1], size=2)][0]
        exactitud = {'Alta': (0, 0), 'Media': (0.5, 0.5), 'Baja': (2, 2)}
        precision = {'Alta': (0.1, 0.1), 'Media': (0.25, 0.25), 'Baja': (1, 1)}
        mx, my = exactitud[e]
        sx, sy = precision[p]
        mx *= smx
        my *= smy
        dart_x = norm.rvs(mx, sx, size=n)
        dart_y = norm.rvs(my, sy, size=n)
        
        fig = go.Figure()
        
        # Círculos concéntricos
        theta = np.linspace(0, 2*np.pi, 100)
        for radius, color in [(1, 'blue'), (2, 'green'), (3, 'red')]:
            fig.add_trace(go.Scatter(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))
        
        # Target
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(size=12, color='black', symbol='circle'),
            name='Target'
        ))
        
        # Dardos
        fig.add_trace(go.Scatter(
            x=dart_x,
            y=dart_y,
            mode='markers',
            marker=dict(size=10, symbol='x', color='purple'),
            name='Dardos'
        ))
        
        fig.update_layout(
            title=f'Exactitud: {e}, Precisión: {p}',
            xaxis=dict(range=[-5, 5], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-5, 5]),
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcular distancia promedio al centro
        dist_centro = np.sqrt(dart_x**2 + dart_y**2).mean()
        st.metric("Distancia promedio al centro", f"{dist_centro:.2f}")

# ===========================
# SECCIÓN 4: INFERENCIA ESTADÍSTICA
# ===========================
elif seccion == "🔬 Inferencia Estadística":
    st.header("🔬 Inferencia Estadística")
    
    inferencia = st.selectbox(
        "Selecciona un tema",
        ["Intervalos de Confianza", "Errores Tipo I y II"]
    )
    
    # ===== INTERVALOS DE CONFIANZA =====
    if inferencia == "Intervalos de Confianza":
        st.subheader("📊 Intervalo t de Confianza para la Media")
        st.markdown("""
        Simulamos muestras de una distribución normal estándar N(0,1) y construimos intervalos de confianza.
        Observa cómo el nivel de confianza y el tamaño de muestra afectan el ancho del intervalo.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            nc = st.select_slider("Nivel de confianza", options=[0.68, 0.90, 0.95, 0.99], value=0.90)
        with col2:
            n = st.selectbox("Tamaño de muestra", [10, 30, 100, 500, 1000], index=1)
        
        size = 100
        alpha = 1 - nc
        df = n - 1
        mu = 0
        
        U = norm.rvs(size=(size, n))
        Um = np.mean(U, axis=1)
        Us = np.std(U, ddof=1, axis=1)
        se = Us / np.sqrt(n)
        
        ICL, ICU = t.interval(1-alpha, df, loc=Um, scale=se)
        CI = np.vstack([ICL, ICU]).transpose()
        cont = np.where(np.logical_and(CI[:, 0] < mu, CI[:, 1] > mu))[0]
        
        fig = go.Figure()
        
        # Dibujar intervalos
        for pos, limits in enumerate(CI):
            color = 'black' if pos in cont else 'red'
            
            # Línea del intervalo
            fig.add_trace(go.Scatter(
                x=limits,
                y=[pos+1, pos+1],
                mode='lines',
                line=dict(color=color, width=1),
                showlegend=False
            ))
            
            # Media muestral
            fig.add_trace(go.Scatter(
                x=[Um[pos]],
                y=[pos+1],
                mode='markers',
                marker=dict(size=4, color=color),
                showlegend=False
            ))
        
        # Línea de media poblacional
        fig.add_vline(
            x=mu,
            line=dict(color='blue', width=2),
            annotation_text='μ'
        )
        
        fig.update_layout(
            title=f'Intervalos de Confianza {nc*100}%',
            xaxis_title='Valor',
            xaxis=dict(range=[-1.5, 1.5]),
            yaxis_title='Intervalo #',
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        pct = np.shape(cont)[0]/size*100
        st.success(f"✅ La media poblacional está en el intervalo **{pct:.1f}%** de las veces (esperado: {nc*100}%)")
        
        st.info(f"""
        **Interpretación:**
        - Intervalos en negro: contienen μ
        - Intervalos en rojo: NO contienen μ
        - Ancho promedio del intervalo: {np.mean(ICU - ICL):.4f}
        """)
    
    # ===== ERRORES TIPO I Y II =====
    elif inferencia == "Errores Tipo I y II":
        st.subheader("⚠️ Errores Tipo I, Tipo II y Poder de la Prueba")
        st.markdown("""
        **Hipótesis:**
        - H₀: μ = 50 (hipótesis nula)
        - H₁: μ < 50 (hipótesis alternativa)
        
        **Errores:**
        - **α (Tipo I):** Rechazar H₀ cuando es verdadera
        - **β (Tipo II):** No rechazar H₀ cuando es falsa
        - **Poder (1-β):** Probabilidad de rechazar H₀ cuando es falsa
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n = st.selectbox("Tamaño de muestra", [10, 25, 30, 36, 100, 500], index=3)
        with col2:
            mu1 = st.select_slider("Media verdadera (μ₁)", options=[35, 40, 43, 45, 46, 47, 48, 49, 50], value=40)
        with col3:
            alpha = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.09, 0.10, 0.20], value=0.09)
        
        mu = 50
        s = 21
        se = s / np.sqrt(n)
        
        x = np.linspace(mu-4*se, mu+4*se, 200)
        pdfx = norm.pdf(x, mu, se)
        
        xc = norm.ppf(alpha, mu, se)
        
        fig = go.Figure()
        
        # Distribución bajo H0
        fig.add_trace(go.Scatter(
            x=x,
            y=pdfx,
            mode='lines',
            name=f'H₀: μ={mu}',
            line=dict(color='blue', width=2)
        ))
        
        # Región crítica (alpha)
        x_alpha = x[x <= xc]
        fig.add_trace(go.Scatter(
            x=x_alpha,
            y=norm.pdf(x_alpha, mu, se),
            fill='tozeroy',
            name='α (Error Tipo I)',
            fillcolor='rgba(255, 165, 0, 0.4)',
            line=dict(color='orange')
        ))
        
        # Distribución bajo H1
        x1 = np.linspace(mu1-4*se, mu1+4*se, 200)
        pdfx1 = norm.pdf(x1, mu1, se)
        
        fig.add_trace(go.Scatter(
            x=x1,
            y=pdfx1,
            mode='lines',
            name=f'H₁: μ={mu1} (verdad)',
            line=dict(color='green', width=2)
        ))
        
        # Beta (Error Tipo II)
        x_beta = x1[x1 >= xc]
        fig.add_trace(go.Scatter(
            x=x_beta,
            y=norm.pdf(x_beta, mu1, se),
            fill='tozeroy',
            name='β (Error Tipo II)',
            fillcolor='rgba(0, 255, 0, 0.3)',
            line=dict(color='lightgreen')
        ))
        
        # Power
        x_power = x1[x1 < xc]
        fig.add_trace(go.Scatter(
            x=x_power,
            y=norm.pdf(x_power, mu1, se),
            fill='tozeroy',
            name='Poder (1-β)',
            fillcolor='rgba(0, 255, 255, 0.3)',
            line=dict(color='cyan')
        ))
        
        # Valor crítico
        fig.add_vline(
            x=xc,
            line=dict(color='red', dash='dash'),
            annotation_text=f'Valor crítico: {xc:.2f}'
        )
        
        fig.update_layout(
            title='Errores Tipo I, Tipo II y Poder de la Prueba',
            xaxis_title='X̄',
            yaxis_title='Densidad',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        beta = 1 - norm.cdf(xc, mu1, se)
        power = 1 - beta
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("α (Error Tipo I)", f"{alpha:.4f}")
        with col_b:
            st.metric("β (Error Tipo II)", f"{beta:.4f}")
        with col_c:
            st.metric("Poder (1-β)", f"{power:.4f}")
        
        st.info(f"""
        **Interpretación:**
        - Valor crítico: **{xc:.4f}**
        - Si X̄ < {xc:.2f}, rechazamos H₀
        - A mayor tamaño de muestra → menor β → mayor poder
        - A mayor diferencia |μ₁ - μ₀| → menor β → mayor poder
        """)

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.caption("🎓 Desarrollado con Streamlit, NumPy, SciPy y Plotly | Probabilidad y Estadística Interactiva")
