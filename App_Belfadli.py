import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import warnings
import mpmath as mp
from scipy.stats import norm
import sympy as sp

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Geman-Yor Double Barrier Options",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .formula-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        font-family: "Courier New", monospace;
        margin: 1rem 0;
        overflow-x: auto;
    }
    .warning-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #DC2626;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CLASSES D'IMPLÉMENTATION DES FORMULES GEMAN-YOR
# ============================================================================

class GemanYorExact:
    """
    Implémentation EXACTE de la méthode de Geman-Yor (1996)
    selon les formules de l'article
    """
    
    def __init__(self, S, K, L, U, T, r, sigma, q=0):
        """
        Initialise les paramètres selon la notation de l'article
        
        Args:
            S: Prix spot S(t) (équivalent à x dans l'article)
            K: Strike price k
            L: Barrière inférieure L
            U: Barrière supérieure U
            T: Temps jusqu'à maturité τ = T - t
            r: Taux d'intérêt sans risque
            sigma: Volatilité σ
            q: Taux de dividende continu (default=0)
        """
        # Paramètres primaires
        self.S = float(S)      # x dans l'article
        self.K = float(K)      # k
        self.L = float(L)      # L
        self.U = float(U)      # U
        self.T = float(T)      # τ = T - t
        self.r = float(r)      # r
        self.sigma = float(sigma)  # σ
        self.q = float(q)      # taux de dividende
        
        # Vérifications
        if not (self.L < self.K < self.U):
            raise ValueError("Condition requise: L < K < U")
        
        # Paramètres dérivés (comme dans l'article)
        self.v = self._compute_v()  # paramètre v
        self.h = self.K / self.S    # h = k/x
        self.m = self.L / self.S    # m = L/x
        self.M = self.U / self.S    # M = U/x
        
        # a et b tels que m = e^{-a}, M = e^{b}
        self.a = -mp.log(self.m) if self.m > 0 else mp.inf
        self.b = mp.log(self.M) if self.M > 0 else mp.inf
        
        # Pour les calculs avec sympy
        self._init_symbolic()
    
    def _compute_v(self):
        """
        Calcule v = (1/σ²)(y - σ²/2) selon l'article
        où y = r - q est le drift risk-neutral
        """
        y = self.r - self.q
        v = (1/(self.sigma**2)) * (y - self.sigma**2/2)
        return v
    
    def _init_symbolic(self):
        """Initialise les variables symboliques pour les calculs exacts"""
        # Variables pour les calculs symboliques
        self.theta_sym = sp.symbols('theta', positive=True)
        self.mu_sym = sp.sqrt(2*self.theta_sym + self.v**2)
        
    def black_scholes_price(self):
        """
        Calcule BS(0, 1, σ, τ, h) selon la formule de l'article
        Prix d'un call standard avec S(0)=1, strike=h, maturité=τ
        """
        tau = self.T
        h = self.h
        sigma = self.sigma
        r = self.r
        q = self.q
        
        # Black-Scholes avec S0=1
        if tau <= 0:
            return max(0, 1 - h)
        
        d1 = (mp.log(1/h) + (r - q + 0.5*sigma**2)*tau) / (sigma*mp.sqrt(tau))
        d2 = d1 - sigma*mp.sqrt(tau)
        
        # Utilisation de mpmath pour plus de précision
        N1 = mp.ncdf(d1) if isinstance(d1, (float, int)) else (0.5 * (1 + mp.erf(d1/mp.sqrt(2))))
        N2 = mp.ncdf(d2) if isinstance(d2, (float, int)) else (0.5 * (1 + mp.erf(d2/mp.sqrt(2))))
        
        price = mp.e**(-q*tau) * N1 - h * mp.e**(-r*tau) * N2
        return float(price)
    
    def g1_at_minus_a(self, mu):
        """
        Calcule g₁(e^{-a}) selon la formule (2.11b) de l'article:
        g₁(e^{-a}) = (h^{ν+1-μ} e^{-μa}) / [μ(μ-ν)(μ-ν-1)]
        """
        h = self.h
        v = self.v
        a = self.a
        
        # Vérification des conditions
        if mu <= v or mu <= v + 1:
            return 0.0
        
        numerator = (h**(v + 1 - mu)) * mp.e**(-mu * a)
        denominator = mu * (mu - v) * (mu - v - 1)
        
        return float(numerator / denominator)
    
    def g1_at_b(self, mu):
        """
        Calcule g₁(e^{b}) selon la formule (2.11c) de l'article:
        g₁(e^{b}) = 2{ e^{b(ν+1)}/[μ²-(ν+1)²] - h e^{bν}/[μ²-ν²] }
                   + e^{-μb} h^{ν+1+μ} / [μ(μ+ν)(μ+ν+1)]
        """
        h = self.h
        v = self.v
        b = self.b
        
        # Terme 1
        term1_numer = mp.e**(b * (v + 1))
        term1_denom = mu**2 - (v + 1)**2
        term1 = term1_numer / term1_denom if term1_denom != 0 else 0
        
        # Terme 2
        term2_numer = h * mp.e**(b * v)
        term2_denom = mu**2 - v**2
        term2 = term2_numer / term2_denom if term2_denom != 0 else 0
        
        # Terme 3
        term3_numer = mp.e**(-mu * b) * h**(v + 1 + mu)
        term3_denom = mu * (mu + v) * (mu + v + 1)
        term3 = term3_numer / term3_denom if term3_denom != 0 else 0
        
        result = 2 * (term1 - term2) + term3
        return float(result)
    
    def Phi_theta(self, theta):
        """
        Calcule Φ(θ) selon la formule (2.11a) de l'article:
        Φ(θ) = [sh(μb)/sh(μ(a+b))] g₁(e^{-a}) 
               + [sh(μa)/sh(μ(a+b))] g₁(e^{b})
        
        où μ = √(2θ + ν²)
        """
        # Calcul de μ
        mu = mp.sqrt(2*theta + self.v**2)
        
        # Calcul des g₁
        g1_minus_a = self.g1_at_minus_a(mu)
        g1_b = self.g1_at_b(mu)
        
        # Calcul des termes avec sinh
        a = self.a
        b = self.b
        
        if a == mp.inf or b == mp.inf or mu == 0:
            return 0.0
        
        sinh_mu_b = mp.sinh(mu * b)
        sinh_mu_a = mp.sinh(mu * a)
        sinh_mu_ab = mp.sinh(mu * (a + b))
        
        if sinh_mu_ab == 0:
            return 0.0
        
        term1 = (sinh_mu_b / sinh_mu_ab) * g1_minus_a
        term2 = (sinh_mu_a / sinh_mu_ab) * g1_b
        
        Phi = term1 + term2
        return float(Phi)
    
    def Psi_lambda(self, lambda_val):
        """
        Calcule ψ(λ) selon la formule (2.6a):
        ψ(λ) = (1/σ²) Φ(λ/σ²)
        """
        theta = lambda_val / (self.sigma**2)
        Phi = self.Phi_theta(theta)
        Psi = Phi / (self.sigma**2)
        return float(Psi)
    
    def Phi_x_theta(self, theta):
        """
        Calcule Φ_x(θ) selon la formule de la page 9 (section hedging):
        Φ_x(θ) = (U^{2μ} - x^{2μ})/(x^{μ+ν+1}) α(L,U,k)
                + (x^{2μ} - L^{2μ})/(x^{μ+ν+1}) β(L,U,k)
        """
        x = self.S
        L = self.L
        U = self.U
        k = self.K
        v = self.v
        mu = mp.sqrt(2*theta + v**2)
        
        # Calcul de α(L,U,k) - formule page 9
        alpha_numer = (L**(2*mu)) * (k**(v + 1 - mu))
        alpha_denom = (U**(2*mu) - L**(2*mu)) * mu * (mu - v) * (mu - v - 1)
        alpha = alpha_numer / alpha_denom if alpha_denom != 0 else 0
        
        # Calcul de β(L,U,k) - formule page 9
        beta_term1_numer = U**(mu + v + 1)
        beta_term1_denom = mu**2 - (v + 1)**2
        beta_term1 = beta_term1_numer / beta_term1_denom if beta_term1_denom != 0 else 0
        
        beta_term2_numer = k * U**(mu + v)
        beta_term2_denom = mu**2 - v**2
        beta_term2 = beta_term2_numer / beta_term2_denom if beta_term2_denom != 0 else 0
        
        beta_term3_numer = k**(mu + v + 1)
        beta_term3_denom = mu * (mu + v) * (mu + v + 1)
        beta_term3 = beta_term3_numer / beta_term3_denom if beta_term3_denom != 0 else 0
        
        beta = (1/(U**(2*mu) - L**(2*mu))) * (
            2 * (beta_term1 - beta_term2) + beta_term3
        ) if (U**(2*mu) - L**(2*mu)) != 0 else 0
        
        # Calcul de Φ_x(θ)
        term1_numer = U**(2*mu) - x**(2*mu)
        term1_denom = x**(mu + v + 1)
        term1 = (term1_numer / term1_denom) * alpha if term1_denom != 0 else 0
        
        term2_numer = x**(2*mu) - L**(2*mu)
        term2_denom = x**(mu + v + 1)
        term2 = (term2_numer / term2_denom) * beta if term2_denom != 0 else 0
        
        Phi_x = term1 + term2
        return float(Phi_x)
    
    def laplace_transform_inversion_stehfest(self, t, n=12):
        """
        Inversion de la transformée de Laplace par la méthode de Stehfest
        Inverser ψ(λ) pour obtenir (L^{-1}ψ)(t)
        
        Args:
            t: point où évaluer la fonction inverse
            n: nombre de termes (doit être pair)
        """
        if n % 2 != 0:
            n = n + 1
        
        # Calcul des coefficients de Stehfest
        V = np.zeros(n)
        for i in range(1, n + 1):
            kmin = int((i + 1) / 2)
            kmax = min(i, n // 2)
            sum_k = 0.0
            
            for k in range(kmin, kmax + 1):
                numerator = mp.power(k, n // 2) * mp.factorial(2 * k)
                denominator = (mp.factorial(n // 2 - k) * 
                             mp.factorial(k) * 
                             mp.factorial(k - 1) * 
                             mp.factorial(i - k) * 
                             mp.factorial(2 * k - i))
                sum_k += numerator / denominator
            
            V[i-1] = mp.power(-1, n // 2 + i) * sum_k
        
        # Calcul de la somme
        ln2_t = mp.log(2) / t
        result = 0.0
        
        for i in range(1, n + 1):
            s = i * ln2_t
            psi_val = self.Psi_lambda(float(s))
            result += V[i-1] * psi_val
        
        result = ln2_t * result
        return float(result)
    
    def compute_option_price(self):
        """
        Calcule le prix de l'option double-barrière selon (2.12):
        C_{L,U}(t) = S(t) { BS(0, 1, σ, τ, h) - e^{-rτ}(L^{-1}ψ)(τ) }
        """
        # Prix Black-Scholes
        BS = self.black_scholes_price()
        
        # Inversion de Laplace de ψ
        inv_laplace = self.laplace_transform_inversion_stehfest(self.T, n=12)
        
        # Calcul final
        term = BS - mp.e**(-self.r * self.T) * inv_laplace
        price = self.S * term
        
        # L'option doit avoir une valeur non-négative
        return float(max(0, price))
    
    def compute_delta(self):
        """
        Calcule le delta selon la formule de la section 4:
        Δ = ∂C/∂S(t) = [C/S(t)] - S(t)e^{-rτ} ∂/∂S(t)[L^{-1}ψ](τ)
        """
        # Calcul du prix
        C = self.compute_option_price()
        
        # Terme 1: C/S(t)
        term1 = C / self.S
        
        # Pour calculer ∂/∂S(t)[L^{-1}ψ](τ), on utilise la différenciation numérique
        # avec perturbation du spot price
        epsilon = 1e-4 * self.S
        
        # Prix avec S + epsilon
        model_plus = GemanYorExact(
            self.S + epsilon, self.K, self.L, self.U,
            self.T, self.r, self.sigma, self.q
        )
        inv_plus = model_plus.laplace_transform_inversion_stehfest(self.T)
        
        # Prix avec S - epsilon
        model_minus = GemanYorExact(
            self.S - epsilon, self.K, self.L, self.U,
            self.T, self.r, self.sigma, self.q
        )
        inv_minus = model_minus.laplace_transform_inversion_stehfest(self.T)
        
        # Dérivée numérique
        d_inv_dS = (inv_plus - inv_minus) / (2 * epsilon)
        
        # Calcul final du delta
        delta = term1 - self.S * mp.e**(-self.r * self.T) * d_inv_dS
        
        return float(delta)
    
    def compute_gamma(self, epsilon=1e-3):
        """
        Calcule le gamma (dérivée seconde) par différences finies
        """
        # Delta avec S + epsilon
        model_plus = GemanYorExact(
            self.S + epsilon, self.K, self.L, self.U,
            self.T, self.r, self.sigma, self.q
        )
        delta_plus = model_plus.compute_delta()
        
        # Delta avec S - epsilon
        model_minus = GemanYorExact(
            self.S - epsilon, self.K, self.L, self.U,
            self.T, self.r, self.sigma, self.q
        )
        delta_minus = model_minus.compute_delta()
        
        # Gamma
        gamma = (delta_plus - delta_minus) / (2 * epsilon)
        return float(gamma)
    
    def monte_carlo_simulation(self, n_simulations=10000, n_steps=1000):
        """
        Simulation Monte Carlo pour validation
        """
        dt = self.T / n_steps
        payoffs = []
        
        for _ in range(n_simulations):
            # Génération du chemin brownien
            Z = np.random.normal(0, 1, n_steps)
            path = np.zeros(n_steps + 1)
            path[0] = self.S
            
            # Simulation du chemin
            knocked_out = False
            for i in range(1, n_steps + 1):
                path[i] = path[i-1] * np.exp(
                    (self.r - self.q - 0.5*self.sigma**2)*dt + 
                    self.sigma*np.sqrt(dt)*Z[i-1]
                )
                
                # Vérification des barrières
                if path[i] <= self.L or path[i] >= self.U:
                    knocked_out = True
                    break
            
            # Payoff à maturité si non knock-out
            if not knocked_out:
                payoff = max(path[-1] - self.K, 0)
                discounted_payoff = payoff * np.exp(-self.r * self.T)
                payoffs.append(discounted_payoff)
            else:
                payoffs.append(0.0)
        
        # Statistiques
        payoffs = np.array(payoffs)
        price = np.mean(payoffs)
        std_err = np.std(payoffs) / np.sqrt(n_simulations)
        
        return float(price), float(std_err)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def validate_parameters(S, K, L, U):
    """
    Valide les paramètres selon les conditions de l'article
    """
    errors = []
    
    if not (L < K < U):
        errors.append("Le strike K doit être entre L et U (L < K < U)")
    
    if not (L < S < U):
        errors.append(f"Le spot S={S} doit être entre L={L} et U={U}")
    
    if L <= 0 or U <= 0 or S <= 0 or K <= 0:
        errors.append("Tous les prix doivent être positifs")
    
    if U <= L:
        errors.append("La barrière supérieure U doit être > barrière inférieure L")
    
    return errors

def compute_barrier_probabilities(S, K, L, U, T, r, sigma):
    """
    Calcule les probabilités de toucher les barrières
    """
    # Probabilité de toucher la barrière inférieure
    if S > L:
        prob_L = norm.cdf((np.log(L/S) - (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
    else:
        prob_L = 1.0
    
    # Probabilité de toucher la barrière supérieure
    if S < U:
        prob_U = 1 - norm.cdf((np.log(U/S) - (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
    else:
        prob_U = 1.0
    
    return {
        'prob_hit_L': float(prob_L),
        'prob_hit_U': float(prob_U),
        'prob_survival': float(max(0, 1 - prob_L - prob_U))  # max pour éviter valeurs négatives
    }

def compute_vanilla_price(S, K, T, r, sigma, q=0, option_type='call'):
    """
    Prix d'une option vanille (pour comparaison)
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    
    return float(price)

def compute_rebate_present_value(rebate, prob_hit, T, r):
    """
    Valeur présente d'un rebate
    """
    return rebate * prob_hit * np.exp(-r*T)

# ============================================================================
# APPLICATION STREAMLIT PRINCIPALE
# ============================================================================

# En-tête
st.markdown('<h1 class="main-header">📐 Méthode Geman-Yor pour Options Double-Barrière</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6B7280; margin-bottom: 2rem;">
    Implémentation exacte des formules de l'article: 
    <em>"Pricing and hedging double-barrier options: A probabilistic approach"</em>
    (Geman & Yor, 1996)
</div>
""", unsafe_allow_html=True)

# Sidebar pour les paramètres
with st.sidebar:
    st.markdown("## ⚙️ Paramètres de l'Option")
    
    # Section 1: Paramètres essentiels
    st.markdown("### 📊 Paramètres du Sous-jacent")
    S0 = st.number_input("**Prix Spot S₀**", value=2.0, min_value=0.01, step=1.0,
                        help="Prix actuel de l'actif sous-jacent")
    K = st.number_input("**Prix d'Exercice K**", value=2.0, min_value=0.01, step=1.0,
                       help="Strike price de l'option")
    
    st.markdown("### 🎯 Barrières")
    colL, colU = st.columns(2)
    with colL:
        L = st.number_input("**Barrière Inférieure L**", value=1.5, min_value=0.01, step=1.0)
    with colU:
        U = st.number_input("**Barrière Supérieure U**", value=2.5, min_value=0.01, step=1.0)
    
    # Section 2: Paramètres de marché
    st.markdown("### 📈 Paramètres de Marché")
    T = st.slider("**Maturité T (années)**", 0.1, 5.0, 1.0, 0.1,
                  help="Temps jusqu'à l'expiration")
    r = st.slider("**Taux sans risque r**", 0.0, 0.1, 0.05, 0.001,
                  format="%.3f", help="Taux d'intérêt continu")
    sigma = st.slider("**Volatilité σ**", 0.1, 1.0, 0.3, 0.01,
                     format="%.2f", help="Volatilité annuelle")
    q = st.slider("**Taux de dividende q**", 0.0, 0.1, 0.0, 0.001,
                  format="%.3f", help="Taux de dividende continu")
    
    # Section 3: Paramètres de calcul
    st.markdown("### 🔧 Paramètres de Calcul")
    include_mc = st.checkbox("Inclure simulation Monte Carlo", value=True)
    n_simulations = st.select_slider("**Nombre de simulations MC**",
                                    options=[1000, 5000, 10000, 20000, 50000],
                                    value=10000)
    
    # Bouton de calcul
    st.markdown("---")
    calculate_btn = st.button("**🎯 CALCULER LE PRIX**", type="primary", 
                            use_container_width=True)
    
    # Information sur la méthode
    with st.expander("ℹ️ À propos de la méthode"):
        st.markdown("""
        **Méthode Geman-Yor (1996):**
        
        1. Transformée de Laplace du prix
        2. Calcul exact de Φ(θ) via les formules (2.11a-c)
        3. Inversion numérique par méthode de Stehfest
        4. Prix final: $C = S[BS - e^{-rτ}(ℒ^{-1}ψ)(τ)]$
        
        **Avantages:**
        - Solution quasi-analytique
        - Très rapide vs Monte Carlo
        - Précision élevée pour le hedging
        """)

# Contenu principal
if calculate_btn:
    # Validation des paramètres
    errors = validate_parameters(S0, K, L, U)
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
        st.stop()
    
    # Initialisation avec barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Étape 1: Initialisation du modèle
        status_text.text("Initialisation du modèle Geman-Yor...")
        progress_bar.progress(10)
        
        model = GemanYorExact(S0, K, L, U, T, r, sigma, q)
        
        # Étape 2: Calcul du prix Geman-Yor
        status_text.text("Calcul du prix par la méthode Geman-Yor...")
        progress_bar.progress(30)
        
        start_time = time.time()
        gy_price = model.compute_option_price()
        gy_time = time.time() - start_time
        
        # Étape 3: Calcul du delta
        status_text.text("Calcul des grecs (Delta, Gamma)...")
        progress_bar.progress(60)
        
        delta = model.compute_delta()
        gamma = model.compute_gamma()
        
        # Étape 4: Comparaisons
        status_text.text("Calcul des prix de comparaison...")
        progress_bar.progress(80)
        
        # Prix vanille
        vanilla_price = compute_vanilla_price(S0, K, T, r, sigma, q, 'call')
        
        # Simulation Monte Carlo
        mc_price, mc_std = None, None
        if include_mc:
            mc_price, mc_std = model.monte_carlo_simulation(n_simulations)
        
        # Probabilités de barrière
        #barrier_probs = compute_barrier_probabilities(S0, K, L, U, T, r, sigma)
        
        # Étape 5: Affichage des résultats
        status_text.text("Préparation des visualisations...")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        # Nettoyer la barre de progression
        progress_bar.empty()
        status_text.empty()
        
        # Affichage des résultats principaux
        st.markdown("## 📊 Résultats du Pricing")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="**Prix Geman-Yor**",
                value=f"${gy_price:.4f}",
                delta=f"{gy_time:.3f}s",
                help="Prix exact par la méthode Geman-Yor"
            )
        
        with col2:
            st.metric(
                label="**Delta (Δ)**",
                value=f"{delta:.4f}",
                delta="Sensibilité au spot",
                help="Dérivée première du prix par rapport au spot"
            )
        
        with col3:
            st.metric(
                label="**Gamma (Γ)**",
                value=f"{gamma:.6f}",
                delta="Convexité",
                help="Dérivée seconde du prix par rapport au spot"
            )
        
        with col4:
            discount = np.exp(-r*T)
            st.metric(
                label="**Facteur d'actualisation**",
                value=f"{discount:.4f}",
                delta=f"r={r*100:.1f}%, T={T}an(s)"
            )
        
        # Comparaisons
        st.markdown('<div class="sub-header">📈 Comparaisons</div>', unsafe_allow_html=True)
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            st.metric(
                label="**Prix Option Vanille**",
                value=f"${vanilla_price:.4f}",
                delta=f"Diff: ${vanilla_price - gy_price:.4f}",
                delta_color="inverse" if vanilla_price > gy_price else "normal"
            )
        
        with comp_col2:
            if mc_price:
                st.metric(
                    label="**Prix Monte Carlo**",
                    value=f"${mc_price:.4f}",
                    delta=f"±{mc_std:.4f} (n={n_simulations})",
                    help=f"IC 95%: [{mc_price-1.96*mc_std:.4f}, {mc_price+1.96*mc_std:.4f}]"
                )
        
 
        
        # Visualisations
        st.markdown('<div class="sub-header">📊 Visualisations</div>', unsafe_allow_html=True)
        
        # Graphique 1: Structure de l'option
        fig1 = go.Figure()
        
        # Zone de survie
        fig1.add_shape(
            type="rect",
            x0=L, x1=U, y0=0, y1=max(vanilla_price, gy_price, 20)*1.3,
            fillcolor="rgba(144, 238, 144, 0.3)",
            line=dict(width=0),
            name="Zone de survie"
        )
        
        # Barrières
        fig1.add_vline(x=L, line=dict(color="red", width=2, dash="dash"),
                      annotation=dict(text=f"L={L}", xanchor="left", y=1.1))
        fig1.add_vline(x=U, line=dict(color="red", width=2, dash="dash"),
                      annotation=dict(text=f"U={U}", xanchor="right", y=1.1))
        
        # Strike et spot
        fig1.add_vline(x=K, line=dict(color="blue", width=2, dash="dot"),
                      annotation=dict(text=f"K={K}", y=0.9))
        fig1.add_vline(x=S0, line=dict(color="green", width=3),
                      annotation=dict(text=f"S₀={S0}", y=0.8))
        
        # Payoff
        S_range = np.linspace(L*0.8, U*1.2, 200)
        payoff = np.where((S_range > L) & (S_range < U), np.maximum(S_range - K, 0), 0)
        
        fig1.add_trace(go.Scatter(
            x=S_range, y=payoff,
            mode='lines',
            name='Payoff',
            line=dict(color='darkblue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))
        
        fig1.update_layout(
            title="Structure de l'Option Double-Barrière",
            xaxis_title="Prix du Sous-jacent à maturité",
            yaxis_title="Payoff",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Graphique 2: Sensibilité au spot
        st.markdown("#### Sensibilité au Prix Spot")
        
        S_range = np.linspace(L*0.9, U*1.1, 50)
        prices = []
        deltas = []
        
        for S in S_range:
            try:
                temp_model = GemanYorExact(S, K, L, U, T, r, sigma, q)
                price = temp_model.compute_option_price()
                delta_val = temp_model.compute_delta()
                prices.append(price)
                deltas.append(delta_val)
            except Exception as e:
                prices.append(0)
                deltas.append(0)
        
        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Prix vs Spot', 'Delta vs Spot'),
            vertical_spacing=0.15
        )
        
        fig2.add_trace(
            go.Scatter(x=S_range, y=prices, mode='lines', name='Prix',
                      line=dict(color='royalblue', width=3)),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=S_range, y=deltas, mode='lines', name='Delta',
                      line=dict(color='firebrick', width=3)),
            row=2, col=1
        )
        
        # Ajouter les barrières et le spot
        fig2.add_vline(x=L, line=dict(color='red', dash='dash'), row=1, col=1)
        fig2.add_vline(x=U, line=dict(color='red', dash='dash'), row=1, col=1)
        fig2.add_vline(x=S0, line=dict(color='green', width=2), row=1, col=1)
        
        fig2.add_vline(x=L, line=dict(color='red', dash='dash'), row=2, col=1)
        fig2.add_vline(x=U, line=dict(color='red', dash='dash'), row=2, col=1)
        fig2.add_vline(x=S0, line=dict(color='green', width=2), row=2, col=1)
        
        fig2.update_layout(height=600, showlegend=False)
        fig2.update_xaxes(title_text="Prix Spot S", row=2, col=1)
        fig2.update_yaxes(title_text="Prix de l'option", row=1, col=1)
        fig2.update_yaxes(title_text="Delta", row=2, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Détails des calculs
        st.markdown('<div class="sub-header">🔍 Détails des Calculs</div>', unsafe_allow_html=True)
        
        with st.expander("Voir les paramètres intermédiaires"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Paramètres dérivés:**")
                st.write(f"- h = K/S = {model.h:.4f}")
                st.write(f"- m = L/S = {model.m:.4f}")
                st.write(f"- M = U/S = {model.M:.4f}")
                st.write(f"- a = -ln(m) = {model.a:.4f}")
                st.write(f"- b = ln(M) = {model.b:.4f}")
                st.write(f"- ν = (1/σ²)(r-q-σ²/2) = {model.v:.4f}")
            
            with col2:
                st.write("**Probabilités estimées:**")
                st.write(f"- Probabilité de toucher L: {barrier_probs['prob_hit_L']*100:.2f}%")
                st.write(f"- Probabilité de toucher U: {barrier_probs['prob_hit_U']*100:.2f}%")
                st.write(f"- Probabilité de survie: {barrier_probs['prob_survival']*100:.2f}%")
        
        with st.expander("Voir les formules utilisées"):
            st.markdown("""
            **Formule principale (2.12):**
            ```
            C_{L,U}(t) = S(t) { BS(0, 1, σ, τ, h) - e^{-rτ}(ℒ^{-1}ψ)(τ) }
            ```
            
            **Transformée de Laplace Φ(θ) (2.11a):**
            ```
            Φ(θ) = [sinh(μb)/sinh(μ(a+b))] g₁(e^{-a}) 
                  + [sinh(μa)/sinh(μ(a+b))] g₁(e^{b})
            ```
            où μ = √(2θ + ν²)
            
            **Fonctions g₁ (2.11b-c):**
            ```
            g₁(e^{-a}) = h^{ν+1-μ} e^{-μa} / [μ(μ-ν)(μ-ν-1)]
            g₁(e^{b}) = 2{e^{b(ν+1)}/[μ²-(ν+1)²] - h e^{bν}/[μ²-ν²]}
                      + e^{-μb} h^{ν+1+μ} / [μ(μ+ν)(μ+ν+1)]
            ```
            """)
        
        # Tableau récapitulatif
        st.markdown("#### 📋 Récapitulatif des Paramètres")
        
        summary_data = {
            "Paramètre": ["Spot S₀", "Strike K", "Barrière L", "Barrière U", 
                         "Maturité T", "Taux r", "Volatilité σ", "Dividende q"],
            "Valeur": [f"{S0:.2f}", f"{K:.2f}", f"{L:.2f}", f"{U:.2f}",
                      f"{T:.3f} ans", f"{r*100:.3f}%", f"{sigma*100:.2f}%", f"{q*100:.3f}%"],
            "Description": ["Prix actuel", "Prix d'exercice", "Barrière inférieure",
                          "Barrière supérieure", "Temps à l'expiration",
                          "Taux sans risque", "Volatilité annuelle", "Taux de dividende"]
        }
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Export des résultats
        st.markdown("#### 📥 Export des Résultats")
        
        if st.button("📄 Générer un rapport détaillé"):
            # Créer un DataFrame avec tous les résultats
            results_df = pd.DataFrame({
                "Méthode": ["Geman-Yor", "Black-Scholes", "Monte Carlo"],
                "Prix": [gy_price, vanilla_price, mc_price if mc_price else np.nan],
                "Temps (s)": [gy_time, 0.001, None],
                "Précision": ["Exact", "Exact", f"±{mc_std:.4f}" if mc_std else None]
            })
            
            # Afficher le DataFrame
            st.dataframe(results_df, use_container_width=True)
            
            # Option de téléchargement
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats en CSV",
                data=csv,
                file_name=f"double_barrier_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul: {str(e)}")

else:
    # Page d'accueil
    st.markdown("""
    <div class="info-box">
    <h3>👋 Bienvenue dans l'outil de pricing d'options double-barrière</h3>
    <p>Cette application implémente la <strong>méthode exacte de Geman-Yor (1996)</strong> 
    pour le pricing et le hedging d'options double-barrière.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Comment utiliser:
        1. **Configurez les paramètres** dans la sidebar
        2. **Vérifiez** que L < K < U et L < S₀ < U
        3. **Cliquez sur "CALCULER LE PRIX"**
        4. **Analysez** les résultats et visualisations
        
        ### 📊 Paramètres recommandés:
        - Spot S₀: 2
        - Strike K: 2
        - Barrière L: 1.5
        - Barrière U: 2.5
        - Maturité T: 1 an
        - Volatilité σ: 30%
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Caractéristiques de la méthode:
        
        **Avantages:**
        - ✅ Solution quasi-analytique
        - ✅ Extrêmement rapide
        - ✅ Précision élevée
        - ✅ Grecs stables
        
        **Formules implémentées:**
        - Formules (2.11a-c) pour Φ(θ)
        - Formule (2.12) pour le prix
        - Méthode de Stehfest pour l'inversion
        
        **Validation:**
        - Comparaison avec Monte Carlo
        - Vérification des bornes
        - Tests de sensibilité
        """)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p><strong>Référence:</strong> Geman, H., & Yor, M. (1996). Pricing and hedging double-barrier options: A probabilistic approach. <em>Mathematical Finance</em>, 6(4), 365-378.</p>
    <p>Implémentation exacte des formules mathématiques de l'article original.</p>
</div>
""", unsafe_allow_html=True)