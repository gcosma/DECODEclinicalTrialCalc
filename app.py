import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Configure page and styling
st.set_page_config(page_title="Clinical Trial Calculator", layout="wide")

# Custom CSS to match Colab styling
st.markdown("""
    <style>
    .main { padding: 2rem; background-color: white; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d6efd;
        color: white;
    }
    .stButton>button {
        width: 200px;
        background-color: #0d6efd;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        border: none;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-left: 5px solid #0d6efd;
        margin: 1rem 0;
        border-radius: 0 0.3rem 0.3rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-left: 5px solid #ffeeba;
        margin: 1rem 0;
        border-radius: 0 0.3rem 0.3rem 0;
    }
    h1, h2, h3, h4 { color: #2c3e50; margin: 1rem 0; font-weight: 600; }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .plot-container {
        background-color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class EnhancedRecruitmentCalculator:
    def __init__(self):
        self.setup_distributions()
        
    def setup_distributions(self):
        self.severity_dist = {
            'Mild': 40,
            'Moderate': 35,
            'Severe': 25
        }
        self.living_dist = {
            'Family': 60,
            'Independent': 25,
            'Supported': 15
        }
        self.severity_mods = {
            'Mild': 1.2,
            'Moderate': 1.0,
            'Severe': 0.8
        }
        self.living_mods = {
            'Family': 1.1,
            'Independent': 1.0,
            'Supported': 0.9
        }

    def calculate_recruitment(self, params):
        results = []
        for severity, severity_pct in self.severity_dist.items():
            for living, living_pct in self.living_dist.items():
                stratum_size = params['total_patients'] * (severity_pct/100) * (living_pct/100)
                adjusted_recruitment = (params['recruitment_rate'] * 
                                     self.severity_mods[severity] * 
                                     self.living_mods[living])
                eligible = stratum_size * params['mltc_rate'] * params['ahc_rate']
                expected_recruits = eligible * adjusted_recruitment
                final_recruits = expected_recruits * (1 - params['attrition_rate'])
                
                results.append({
                    'severity': severity,
                    'living': living,
                    'stratum_size': stratum_size,
                    'eligible': eligible,
                    'expected_recruits': expected_recruits,
                    'final_recruits': final_recruits
                })
        return results

    def plot_recruitment_analysis(self, results, params):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 18))
        gs = plt.GridSpec(3, 1, height_ratios=[1.2, 1, 0.8], hspace=0.4)
        
        # Plot 1: Recruitment Heatmap
        ax1 = fig.add_subplot(gs[0])
        data = np.zeros((3, 3))
        severities = list(self.severity_dist.keys())
        living_situations = list(self.living_dist.keys())
        
        for result in results:
            i = severities.index(result['severity'])
            j = living_situations.index(result['living'])
            data[i, j] = result['final_recruits']
        
        sns.heatmap(data, annot=True, fmt='.1f',
                   xticklabels=living_situations,
                   yticklabels=severities,
                   cmap='YlOrRd', ax=ax1,
                   cbar_kws={'label': 'Expected Recruits'})
        ax1.set_title('Expected Recruitment by Patient Characteristics',
                     pad=20, fontsize=14, fontweight='bold')
        
        # Plot 2: Practice Distribution
        ax2 = fig.add_subplot(gs[1])
        total_per_practice = sum(r['final_recruits'] for r in results)
        practice_range = np.arange(24, 49, 2)
        avg_total = practice_range * total_per_practice
        percentage_above = ((avg_total - 204) / 204) * 100
        
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(practice_range)))
        bars = ax2.bar(practice_range, percentage_above, color=colors)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Target (204)')
        target_practices = 36
        target_idx = np.where(practice_range == target_practices)[0][0]
        ax2.plot(target_practices, percentage_above[target_idx], 'r*', 
                markersize=15, label=f'Target: {target_practices} practices')
        
        ax2.set_xlabel('Number of Practices', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Percentage Above Target (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Expected Recruitment by Practice Numbers',
                     fontsize=14, pad=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Success Probability
        ax3 = fig.add_subplot(gs[2])
        variability = params.get('practice_variability', 0.35)
        practice_performances = np.random.normal(
            total_per_practice,
            total_per_practice * variability,
            size=1000
        )
        practice_performances = np.clip(practice_performances, 0, 3 * total_per_practice)
        
        target_per_arm = 102
        simulated_totals = []
        min_practices = params.get('min_practices', 18)
        
        for _ in range(1000):
            practices = np.random.choice(practice_performances, size=min_practices)
            total_recruits = sum(practices)
            simulated_totals.append(total_recruits)
        
        prob_success = sum(np.array(simulated_totals) >= target_per_arm) / 1000
        
        ax3.hist(simulated_totals, bins=30, color='skyblue', alpha=0.7)
        ax3.axvline(target_per_arm, color='red', linestyle='--',
                   label=f'Target per arm ({target_per_arm})')
        ax3.set_title(f'Recruitment Success Probability Distribution\n'
                     f'Probability of Meeting Target: {prob_success:.1%}')
        ax3.set_xlabel('Total Recruits per Arm')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        plt.tight_layout()
        return fig, total_per_practice, np.ceil(102/total_per_practice), prob_success

class EnhancedPowerCalculator:
    def __init__(self):
        self.setup_defaults()

    def setup_defaults(self):
        self.power_levels = [0.80, 0.85, 0.90, 0.95]
        self.colors = ['#FF6B6B', '#45B7D1', '#4ECDC4', '#96CEB4']

    def calculate_power_analysis(self, params):
        results = []
        for effect_size in params['effect_sizes']:
            z_alpha = stats.norm.ppf(1 - params['alpha']/2)
            z_beta = stats.norm.ppf(params['power'])
            variance_adjusted = 2 * params['score_sd']**2 * (1 - params['baseline_correlation'])
            base_n = 2 * ((z_alpha + z_beta)**2 * variance_adjusted) / (effect_size**2)
            
            design_effect = 1 + ((params['cluster_size'] - 1) * params['icc'])
            adjusted_n = base_n * design_effect / (1 - params['dropout_rate'])
            practices_per_group = int(np.ceil(adjusted_n/params['cluster_size']))
            
            results.append({
                'effect_size': effect_size,
                'base_n': base_n,
                'adjusted_n': adjusted_n,
                'total_size': adjusted_n * 2,
                'practices_per_group': practices_per_group,
                'power': params['power']
            })
        return results

    def plot_power_analysis(self, params, results):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        powers = np.linspace(0.5, 0.99, 100)
        
        y_offsets = {0.80: -0.03, 0.85: -0.02, 0.90: 0.02, 0.95: 0.03}
        x_offsets = {0.80: 20, 0.85: 40, 0.90: 20, 0.95: 40}
        
        for i, effect_size in enumerate(params['effect_sizes']):
            sample_sizes = []
            for p in powers:
                z_alpha = stats.norm.ppf(1 - params['alpha']/2)
                z_beta = stats.norm.ppf(p)
                variance_adjusted = 2 * params['score_sd']**2 * (1 - params['baseline_correlation'])
                base_n = 2 * ((z_alpha + z_beta)**2 * variance_adjusted) / (effect_size**2)
                adjusted_n = base_n * (1 + ((params['cluster_size'] - 1) * params['icc'])) / (1 - params['dropout_rate'])
                sample_sizes.append(adjusted_n)
            
            plt.plot(sample_sizes, powers,
                    label=f'Effect size {effect_size:.3f}',
                    color=self.colors[i % len(self.colors)],
                    linewidth=2.5)
            
            for power_level in self.power_levels:
                idx = np.abs(powers - power_level).argmin()
                n = sample_sizes[idx]
                practices = np.ceil(n / params['cluster_size'])
                
                plt.plot(n, power_level, 'D',
                        color=self.colors[i % len(self.colors)],
                        markersize=8)
                
                plt.annotate(
                    f'{power_level:.0%} Power:\nN={n:,.0f}\nPractices={practices:.0f}',
                    xy=(n, power_level),
                    xytext=(n + x_offsets[power_level], power_level + y_offsets[power_level]),
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        fc="white",
                        ec=self.colors[i % len(self.colors)],
                        alpha=0.9
                    ),
                    arrowprops=dict(
                        arrowstyle='-',
                        color=self.colors[i % len(self.colors)],
                        alpha=0.5,
                        connectionstyle='arc3,rad=-0.2'
                    )
                )
        
        for power_level in self.power_levels:
            plt.axhline(y=power_level, color='#2c3e50', linestyle='--', alpha=0.2)
        
        plt.xlabel('Sample Size per Group', fontsize=12, fontweight='bold')
        plt.ylabel('Statistical Power', fontsize=12, fontweight='bold')
        plt.title(f'Power Analysis for EQ-5D Study\nICC = {params["icc"]:.2f}, α = {params["alpha"]}',
                 fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(title='Effect Sizes', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        return fig

def main():
    st.title("Clinical Trial Calculator")
    
    tab1, tab2 = st.tabs(["📊 Recruitment Calculator", "📈 Power Analysis"])
    
    with tab1:
        st.markdown("""
            <div class='info-box'>
            <h4>Enhanced Recruitment Calculator Features:</h4>
            <ul>
                <li>Accounts for attrition in sample size calculations</li>
                <li>Models practice-level recruitment variation</li>
                <li>Includes stratification by LD severity and living situation</li>
                <li>Projects recruitment success probability</li>
            </ul>
            <p><strong>Target:</strong> 204 total participants (accounting for attrition)</p>
            </div>
        """, unsafe_allow_html=True)
        
        calculator = EnhancedRecruitmentCalculator()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Base Parameters")
            total_patients = st.number_input(
                "Total LD Patients per Practice",
                value=40,
                min_value=1
            )
            mltc_rate = st.slider(
                "MLTC Rate (%)",
                0.0, 100.0, 66.67
            ) / 100
            ahc_rate = st.slider(
                "AHC Uptake Rate (%)",
                0.0, 100.0, 70.0
            ) / 100
        
        with col2:
            st.subheader("Response Parameters")
            recruitment_rate = st.slider(
                "Study Recruitment Rate (%)",
                0.0, 100.0, 40.0
            ) / 100
            attrition_rate = st.slider(
                "Expected Attrition Rate (%)",
                0.0, 50.0, 20.0
            ) / 100
        
        st.subheader("Practice Variation")
        col3, col4 = st.columns(2)
        with col3:
            practice_variability = st.slider(
                "Practice Performance Variability (%)",
                0.0, 100.0, 35.0
            ) / 100
            min_practices = st.number_input(
                "Minimum Practices per Arm",
                value=18,
                min_value=1
            )
        
        with col4:
            st.markdown("""
                <div class='info-box'>
                <h5>Stratification Settings</h5>
                <p><strong>LD Severity Distribution:</strong><br>
                Mild: 40%, Moderate: 35%, Severe: 25%</p>
                <p><strong>Living Situation Distribution:</strong><br>
                Family: 60%, Independent: 25%, Supported: 15%</p>
                </div>
            """, unsafe_allow_html=True)

        if st.button("Calculate Recruitment", key="recruit_btn"):
            params = {
                'total_patients': total_patients,
                'mltc_rate': mltc_rate,
                'ahc_rate': ahc_rate,
                'recruitment_rate': recruitment_rate,
                'attrition_rate': attrition_rate,
                'practice_variability': practice_variability,
                'min_practices': min_practices
            }
            
            results = calculator.calculate_recruitment(params)
            fig, total_recruits, required_practices, prob_success = calculator.plot_recruitment_analysis(
                results, params
            )
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric(
                    "Total Expected Recruits per Practice",
                    f"{total_recruits:.1f}",
                    delta=f"{total_recruits*2:.0f} total participants"
                )
            with col6:
                st.metric(
                    "Required Practices per Arm",
                    f"{required_practices:.0f}",
                    delta=f"{required_practices*2:.0f} total practices"
                )
            
            st.markdown(f"""
                <div class='metric-container'>
                <h4>Recruitment Success Probability</h4>
                <p style='font-size: 24px; font-weight: bold; color: {'green' if prob_success >= 0.8 else 'orange' if prob_success >= 0.6 else 'red'}'>
                {prob_success:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.pyplot(fig)
            
            st.subheader("Detailed Recruitment Projections")
            df = pd.DataFrame(results)
            styled_df = df.style.format({
                'stratum_size': '{:.1f}',
                'eligible': '{:.1f}',
                'expected_recruits': '{:.1f}',
                'final_recruits': '{:.1f}'
            }).background_gradient(
                cmap='YlOrRd',
                subset=['final_recruits']
            )
            st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        st.markdown("""
            <div class='info-box'>
            <h4>Power Analysis Calculator Features:</h4>
            <ul>
                <li>Multiple effect size analysis</li>
                <li>Clustering adjustment with ICC</li>
                <li>Baseline correlation adjustment</li>
                <li>Comprehensive power curves</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)
        
        power_calculator = EnhancedPowerCalculator()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Study Parameters")
            eq5d_version = st.selectbox(
                "EQ-5D Version",
                options=["EQ-5D-3L"],
                help="Selected 3L (simpler) for its suitability for the LD population"
            )
            power = st.slider(
                "Statistical Power",
                0.5, 0.99, 0.90,
                help="Probability of detecting a true effect"
            )
            alpha = st.slider(
                "Significance Level (α)",
                0.01, 0.10, 0.05,
                help="Probability of Type I error"
            )
            icc = st.slider(
                "Intraclass Correlation Coefficient",
                0.0, 1.0, 0.05,
                help="Measure of clustering effect"
            )
            cluster_size = st.number_input(
                "Average Patients per Practice",
                value=7,
                min_value=1
            )
            dropout_rate = st.slider(
                "Expected Dropout Rate (%)",
                0.0, 50.0, 20.0
            ) / 100
        
        with col2:
            st.subheader("EQ-5D Parameters")
            baseline_score = st.number_input(
                "Baseline EQ-5D Score",
                value=0.64,
                min_value=-0.594,
                max_value=1.0
            )
            score_sd = st.number_input(
                "EQ-5D Standard Deviation",
                value=0.26,
                min_value=0.0,
                max_value=1.0
            )
            baseline_correlation = st.slider(
                "Baseline-Outcome Correlation",
                0.0, 1.0, 0.5
            )
            mcid = st.number_input(
                "Minimal Clinically Important Difference",
                value=0.074,
                min_value=0.0,
                max_value=1.0
            )
        
        effect_sizes = st.multiselect(
            "Effect Sizes to Analyze",
            options=[0.074, 0.10, 0.15, 0.20],
            default=[0.074, 0.10, 0.15, 0.20]
        )
        
        if st.button("Calculate Power Analysis", key="power_btn"):
            params = {
                'power': power,
                'alpha': alpha,
                'icc': icc,
                'cluster_size': cluster_size,
                'dropout_rate': dropout_rate,
                'score_sd': score_sd,
                'baseline_correlation': baseline_correlation,
                'effect_sizes': effect_sizes
            }
            
            results = power_calculator.calculate_power_analysis(params)
            fig = power_calculator.plot_power_analysis(params, results)
            
            st.pyplot(fig)
            
            st.subheader("Power Analysis Results")
            df = pd.DataFrame(results)
            styled_df = df.style.format({
                'effect_size': '{:.3f}',
                'base_n': '{:.0f}',
                'adjusted_n': '{:.0f}',
                'total_size': '{:.0f}',
                'practices_per_group': '{:.0f}',
                'power': '{:.1%}'
            }).background_gradient(
                cmap='YlOrRd',
                subset=['adjusted_n', 'total_size']
            )
            st.dataframe(styled_df, use_container_width=True)

if __name__ == "__main__":
    main()
