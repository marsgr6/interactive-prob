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
        k_max = int(np.ceil(lmbda + 10 * np.sqrt(max(lmbda, 1))))
        k_max = min(max(k_max, 20), 200)
        if observed_max:
            k_max = max(k_max, int(observed_max))
        return np.arange(0, k_max + 1)
    elif dist_name == "Geométrica":
        p = params['p']
        if p > 0:
            k_max = int(np.ceil(10 / p))
            return np.arange(1, min(k_max, 200))
        return np.arange(1, 50)
    elif dist_name == "Binomial Negativa":
        r, p = params['r'], params['p']
        if p > 0:
            k_max = int(np.ceil(r * (1-p) / p + 10 * np.sqrt(r * (1-p) / p**2)))
            return np.arange(0, min(k_max, 200))
        return np.arange(0, 50)
    elif dist_name == "Hipergeométrica":
        n = params['n']
        return np.arange(0, n + 1)
    return np.arange(0, 50)

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
                params['n'] = st.selectbox("n (ensayos)", list(range(1, 101)), index=9)
            with col2:
                params['p'] = st.selectbox("p (prob. éxito)", [round(i/100, 2) for i in range(0, 101)], index=50)
            with col3:
                num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data = np.random.binomial(params['n'], params['p'], num_sims)
            theo_mean = params['n'] * params['p']
            theo_var = params['n'] * params['p'] * (1 - params['p'])
            
        elif discrete_dist == "Poisson":
            with col1:
                params['lambda'] = st.selectbox("λ (tasa)", [round(0.1*i, 1) for i in range(0, 301)], index=50)
            with col2:
                num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            with col3:
                st.markdown("—")
            
            sim_data = np.random.poisson(params['lambda'], num_sims)
            theo_mean = params['lambda']
            theo_var = params['lambda']
            
        elif discrete_dist == "Geométrica":
            with col1:
                params['p'] = st.selectbox("p (prob. éxito)", [round(i/100, 2) for i in range(1, 101)], index=20)
            with col2:
                num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            with col3:
                st.markdown("—")
            
            sim_data = np.random.geometric(params['p'], num_sims)
            theo_mean = 1 / params['p']
            theo_var = (1 - params['p']) / params['p']**2
            
        elif discrete_dist == "Binomial Negativa":
            with col1:
                params['r'] = st.selectbox("r (éxitos)", list(range(1, 51)), index=4)
            with col2:
                params['p'] = st.selectbox("p (prob. éxito)", [round(i/100, 2) for i in range(1, 101)], index=50)
            with col3:
                num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data = np.random.negative_binomial(params['r'], params['p'], num_sims)
            theo_mean = params['r'] * (1 - params['p']) / params['p']
            theo_var = params['r'] * (1 - params['p']) / params['p']**2
            
        elif discrete_dist == "Hipergeométrica":
            with col1:
                params['N'] = st.selectbox("N (población)", list(range(10, 201)), index=40)
            with col2:
                params['K'] = st.selectbox("K (éxitos en pob.)", list(range(1, params['N']+1)), index=min(20, params['N']-1))
            with col3:
                params['n'] = st.selectbox("n (muestra)", list(range(1, params['N']+1)), index=min(10, params['N']-1))
            
            num_sims = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data = np.random.hypergeometric(params['K'], params['N']-params['K'], params['n'], num_sims)
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
                calc_params['n'] = st.selectbox("n (ensayos)", list(range(1, 101)), index=9, key="calc_n")
            with calc_cols[1]:
                calc_params['p'] = st.selectbox("p (prob. éxito)", [round(i/100, 2) for i in range(0, 101)], index=50, key="calc_p")
            with calc_cols[2]:
                k_options = list(range(0, calc_params['n'] + 1))
                k_calc = st.selectbox("k", k_options, index=min(calc_params['n']//2, len(k_options)-1))
            
            pmf_val = binom.pmf(k_calc, calc_params['n'], calc_params['p'])
            cdf_val = binom.cdf(k_calc, calc_params['n'], calc_params['p'])
            
        elif discrete_dist == "Poisson":
            with calc_cols[0]:
                calc_params['lambda'] = st.selectbox("λ (tasa)", [round(0.1*i, 1) for i in range(0, 301)], index=50, key="calc_lam")
            with calc_cols[1]:
                k_domain = get_discrete_range("Poisson", {'lambda': calc_params['lambda']})
                k_calc = st.selectbox("k", list(k_domain), index=int(np.clip(calc_params['lambda'], 0, len(k_domain)-1)))
            with calc_cols[2]:
                st.markdown("—")
            
            pmf_val = poisson.pmf(k_calc, calc_params['lambda'])
            cdf_val = poisson.cdf(k_calc, calc_params['lambda'])
            
        elif discrete_dist == "Geométrica":
            with calc_cols[0]:
                calc_params['p'] = st.selectbox("p (prob. éxito)", [round(i/100, 2) for i in range(1, 101)], index=20, key="calc_p_geom")
            with calc_cols[1]:
                k_domain = get_discrete_range("Geométrica", {'p': calc_params['p']})
                k_calc = st.selectbox("k", list(k_domain), index=min(10, len(k_domain)-1))
            with calc_cols[2]:
                st.markdown("—")
            
            pmf_val = geom.pmf(k_calc, calc_params['p'])
            cdf_val = geom.cdf(k_calc, calc_params['p'])
            
        elif discrete_dist == "Binomial Negativa":
            with calc_cols[0]:
                calc_params['r'] = st.selectbox("r (éxitos)", list(range(1, 51)), index=4, key="calc_r")
            with calc_cols[1]:
                calc_params['p'] = st.selectbox("p (prob. éxito)", [round(i/100, 2) for i in range(1, 101)], index=50, key="calc_p_nb")
            with calc_cols[2]:
                k_domain = get_discrete_range("Binomial Negativa", calc_params)
                k_calc = st.selectbox("k", list(k_domain), index=min(10, len(k_domain)-1))
            
            pmf_val = nbinom.pmf(k_calc, calc_params['r'], calc_params['p'])
            cdf_val = nbinom.cdf(k_calc, calc_params['r'], calc_params['p'])
            
        elif discrete_dist == "Hipergeométrica":
            with calc_cols[0]:
                calc_params['N'] = st.selectbox("N (población)", list(range(10, 201)), index=40, key="calc_N")
            with calc_cols[1]:
                calc_params['K'] = st.selectbox("K (éxitos)", list(range(1, calc_params['N']+1)), index=min(20, calc_params['N']-1), key="calc_K")
            with calc_cols[2]:
                calc_params['n'] = st.selectbox("n (muestra)", list(range(1, calc_params['N']+1)), index=min(10, calc_params['N']-1), key="calc_n_hyper")
            
            k_domain = get_discrete_range("Hipergeométrica", calc_params)
            k_calc = st.selectbox("k", list(k_domain), index=min(5, len(k_domain)-1), key="calc_k_hyper")
            
            pmf_val = hypergeom.pmf(k_calc, calc_params['N'], calc_params['K'], calc_params['n'])
            cdf_val = hypergeom.cdf(k_calc, calc_params['N'], calc_params['K'], calc_params['n'])
        
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
                cont_params['mu'] = st.number_input("μ (media)", -10.0, 10.0, 0.0, 0.5)
            with col2:
                cont_params['sigma'] = st.number_input("σ (desv. est.)", 0.1, 10.0, 1.0, 0.1)
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.normal(cont_params['mu'], cont_params['sigma'], num_sims_cont)
            theo_mean_cont = cont_params['mu']
            theo_var_cont = cont_params['sigma']**2
            
        elif continuous_dist == "Uniforme":
            with col1:
                cont_params['a'] = st.number_input("a (mínimo)", -10.0, 10.0, 0.0, 0.5)
            with col2:
                cont_params['b'] = st.number_input("b (máximo)", cont_params['a']+0.1, 20.0, cont_params['a']+1.0, 0.5)
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.uniform(cont_params['a'], cont_params['b'], num_sims_cont)
            theo_mean_cont = (cont_params['a'] + cont_params['b']) / 2
            theo_var_cont = ((cont_params['b'] - cont_params['a'])**2) / 12
            
        elif continuous_dist == "Exponencial":
            with col1:
                cont_params['lambda'] = st.number_input("λ (tasa)", 0.1, 10.0, 1.0, 0.1)
            with col2:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            with col3:
                st.markdown("—")
            
            sim_data_cont = np.random.exponential(1/cont_params['lambda'], num_sims_cont)
            theo_mean_cont = 1 / cont_params['lambda']
            theo_var_cont = 1 / cont_params['lambda']**2
            
        elif continuous_dist == "Gamma":
            with col1:
                cont_params['shape'] = st.number_input("α (forma)", 0.1, 20.0, 2.0, 0.1)
            with col2:
                cont_params['scale'] = st.number_input("β (escala)", 0.1, 10.0, 1.0, 0.1)
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.gamma(cont_params['shape'], cont_params['scale'], num_sims_cont)
            theo_mean_cont = cont_params['shape'] * cont_params['scale']
            theo_var_cont = cont_params['shape'] * cont_params['scale']**2
            
        elif continuous_dist == "Beta":
            with col1:
                cont_params['alpha'] = st.number_input("α", 0.1, 10.0, 2.0, 0.1)
            with col2:
                cont_params['beta'] = st.number_input("β", 0.1, 10.0, 2.0, 0.1)
            with col3:
                num_sims_cont = st.number_input("Nº simulaciones", 100, 100000, 5000, 500)
            
            sim_data_cont = np.random.beta(cont_params['alpha'], cont_params['beta'], num_sims_cont)
            alpha, beta_p = cont_params['alpha'], cont_params['beta']
            theo_mean_cont = alpha / (alpha + beta_p)
            theo_var_cont = (alpha * beta_p) / ((alpha + beta_p)**2 * (alpha + beta_p + 1))
            
        elif continuous_dist == "Weibull":
            with col1:
                cont_params['shape'] = st.number_input("k (forma)", 0.1, 10.0, 1.5, 0.1)
            with col2:
                cont_params['scale'] = st.number_input("λ (escala)", 0.1, 10.0, 1.0, 0.1)
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
                calc_params_cont['mu'] = st.number_input("μ (media)", -10.0, 10.0, 0.0, 0.5, key="calc_mu")
            with calc_cols_cont[1]:
                calc_params_cont['sigma'] = st.number_input("σ (desv. est.)", 0.1, 10.0, 1.0, 0.1, key="calc_sigma")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = calc_params_cont['mu'] - calc_params_cont['sigma']
            default_b = calc_params_cont['mu'] + calc_params_cont['sigma']
            
        elif continuous_dist == "Uniforme":
            with calc_cols_cont[0]:
                calc_params_cont['a'] = st.number_input("a (mínimo)", -10.0, 10.0, 0.0, 0.5, key="calc_a")
            with calc_cols_cont[1]:
                calc_params_cont['b'] = st.number_input("b (máximo)", calc_params_cont['a']+0.1, 20.0, calc_params_cont['a']+1.0, 0.5, key="calc_b")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = calc_params_cont['a']
            default_b = calc_params_cont['b']
            
        elif continuous_dist == "Exponencial":
            with calc_cols_cont[0]:
                calc_params_cont['lambda'] = st.number_input("λ (tasa)", 0.1, 10.0, 1.0, 0.1, key="calc_lambda")
            with calc_cols_cont[1]:
                st.markdown("—")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = 0
            default_b = 2 / calc_params_cont['lambda']
            
        elif continuous_dist == "Gamma":
            with calc_cols_cont[0]:
                calc_params_cont['shape'] = st.number_input("α (forma)", 0.1, 20.0, 2.0, 0.1, key="calc_shape_gamma")
            with calc_cols_cont[1]:
                calc_params_cont['scale'] = st.number_input("β (escala)", 0.1, 10.0, 1.0, 0.1, key="calc_scale_gamma")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = 0
            default_b = calc_params_cont['shape'] * calc_params_cont['scale']
            
        elif continuous_dist == "Beta":
            with calc_cols_cont[0]:
                calc_params_cont['alpha'] = st.number_input("α", 0.1, 10.0, 2.0, 0.1, key="calc_alpha")
            with calc_cols_cont[1]:
                calc_params_cont['beta'] = st.number_input("β", 0.1, 10.0, 2.0, 0.1, key="calc_beta")
            with calc_cols_cont[2]:
                st.markdown("—")
            
            default_a = 0.2
            default_b = 0.8
            
        elif continuous_dist == "Weibull":
            with calc_cols_cont[0]:
                calc_params_cont['shape'] = st.number_input("k (forma)", 0.1, 10.0, 1.5, 0.1, key="calc_shape_weib")
            with calc_cols_cont[1]:
                calc_params_cont['scale'] = st.number_input("λ (escala)", 0.1, 10.0, 1.0, 0.1, key="calc_scale_weib")
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
        ["Histogramas y Bines", "Correlación", "Ley de Grandes Números", "Teorema Límite Central", "Exactitud y Precisión"]
    )
    
    # ===== HISTOGRAMAS Y BINES =====
    if concepto == "Histogramas y Bines":
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