import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Data classes for organization
@dataclass
class RecruitmentResult:
    severity: str
    living: str
    stratum_size: float
    eligible: float
    expected_recruits: float
    final_recruits: float

@dataclass
class PowerResult:
    effect_size: float
    base_n: float
    adjusted_n: float
    total_size: float
    practices_per_group: int
    power_achieved: float

class RecruitmentCalculator:
    def __init__(self):
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

    def calculate_stratified_recruitment(
            self, total_patients: int, mltc_rate: float, 
            ahc_rate: float, recruitment_rate: float, 
            attrition_rate: float) -> List[RecruitmentResult]:
        
        results = []
        for severity, severity_pct in self.severity_dist.items():
            for living, living_pct in self.living_dist.items():
                stratum_size = total_patients * (severity_pct/100) * (living_pct/100)
                adjusted_recruitment = (recruitment_rate * 
                                     self.severity_mods[severity] * 
                                     self.living_mods[living])
                eligible = stratum_size * mltc_rate * ahc_rate
                expected_recruits = eligible * adjusted_recruitment
                final_recruits = expected_recruits * (1 - attrition_rate)
                
                results.append(RecruitmentResult(
                    severity=severity,
                    living=living,
                    stratum_size=stratum_size,
                    eligible=eligible,
                    expected_recruits=expected_recruits,
                    final_recruits=final_recruits
                ))
        return results

    def create_recruitment_table(self, results: List[RecruitmentResult]) -> pd.DataFrame:
        data = []
        for result in results:
            data.append({
                'Severity': result.severity,
                'Living Situation': result.living,
                'Stratum Size': f"{result.stratum_size:.1f}",
                'Eligible Patients': f"{result.eligible:.1f}",
                'Expected Recruits': f"{result.expected_recruits:.1f}",
                'Final Recruits': f"{result.final_recruits:.1f}"
            })
        return pd.DataFrame(data)

class PowerCalculator:
    def calculate_power_analysis(
            self, effect_sizes: List[float], power: float, 
            alpha: float, score_sd: float, 
            baseline_correlation: float, cluster_size: int, 
            icc: float, dropout_rate: float) -> List[PowerResult]:
        
        results = []
        for effect_size in effect_sizes:
            # Calculate base sample size
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            variance_adjusted = 2 * score_sd**2 * (1 - baseline_correlation)
            base_n = 2 * ((z_alpha + z_beta)**2 * variance_adjusted) / (effect_size**2)
            
            # Apply adjustments
            design_effect = 1 + ((cluster_size - 1) * icc)
            adjusted_n = base_n * design_effect / (1 - dropout_rate)
            
            # Calculate achieved power
            achieved_power = stats.norm.cdf(
                np.sqrt((effect_size**2 * adjusted_n)/(2 * variance_adjusted)) - z_alpha
            )
            
            results.append(PowerResult(
                effect_size=effect_size,
                base_n=base_n,
                adjusted_n=adjusted_n,
                total_size=adjusted_n * 2,
                practices_per_group=int(np.ceil(adjusted_n/cluster_size)),
                power_achieved=achieved_power
            ))
        return results

    def create_power_table(self, results: List[PowerResult]) -> pd.DataFrame:
        data = []
        for result in results:
            data.append({
                'Effect Size': f"{result.effect_size:.3f}",
                'Base Sample Size': f"{result.base_n:.0f}",
                'Adjusted Sample Size': f"{result.adjusted_n:.0f}",
                'Total Study Size': f"{result.total_size:.0f}",
                'Practices per Group': result.practices_per_group,
                'Achieved Power': f"{result.power_achieved:.1%}"
            })
        return pd.DataFrame(data)

def main():
    st.set_page_config(page_title="Clinical Trial Calculator", layout="wide")
    
    st.title("Clinical Trial Calculator")
    st.sidebar.markdown("""
    ### About
    This calculator combines two essential tools for clinical trial planning:
    1. Recruitment Calculator
    2. Power Analysis Calculator
    
    Each calculator provides detailed visualizations and tables for comprehensive trial planning.
    """)
    
    calculator_type = st.sidebar.radio(
        "Select Calculator",
        ["Recruitment Calculator", "Power Analysis Calculator"]
    )
    
    if calculator_type == "Recruitment Calculator":
        recruitment_calculator = RecruitmentCalculator()
        
        st.header("Recruitment Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Base Parameters")
            total_patients = st.number_input(
                "Total LD Patients per Practice",
                value=40, min_value=1
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
            attrition_rate = st.slider(
                "Expected Attrition Rate (%)",
                0.0, 50.0, 20.0
            ) / 100
            recruitment_rate = st.slider(
                "Study Recruitment Rate (%)",
                0.0, 100.0, 40.0
            ) / 100
        
        if st.button("Calculate Recruitment"):
            results = recruitment_calculator.calculate_stratified_recruitment(
                total_patients, mltc_rate, ahc_rate,
                recruitment_rate, attrition_rate
            )
            
            # Create recruitment table
            df = recruitment_calculator.create_recruitment_table(results)
            st.subheader("Detailed Recruitment Projections")
            st.dataframe(df, use_container_width=True)
            
            # Calculate totals
            total_recruits = sum(r.final_recruits for r in results)
            required_practices = np.ceil(102/total_recruits)
            
            # Display summary metrics
            col3, col4 = st.columns(2)
            with col3:
                st.metric(
                    "Total Expected Recruits per Practice",
                    f"{total_recruits:.1f}"
                )
            with col4:
                st.metric(
                    "Required Practices per Arm",
                    f"{required_practices:.0f}"
                )
            
            # Create and display heatmap
            data = np.zeros((3, 3))
            severities = list(recruitment_calculator.severity_dist.keys())
            living_situations = list(recruitment_calculator.living_dist.keys())
            
            for result in results:
                i = severities.index(result.severity)
                j = living_situations.index(result.living)
                data[i, j] = result.final_recruits
            
            st.subheader("Recruitment Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                data,
                annot=True,
                fmt='.1f',
                xticklabels=living_situations,
                yticklabels=severities,
                cmap='YlOrRd'
            )
            plt.title('Expected Recruitment by Patient Characteristics')
            st.pyplot(fig)
    
    else:  # Power Analysis Calculator
        power_calculator = PowerCalculator()
        
        st.header("Power Analysis Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Study Parameters")
            power = st.slider("Statistical Power", 0.5, 0.99, 0.90)
            alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05)
            icc = st.slider("Intraclass Correlation Coefficient", 0.0, 1.0, 0.05)
            cluster_size = st.number_input(
                "Average Patients per Practice",
                value=7, min_value=1
            )
            dropout_rate = st.slider(
                "Expected Dropout Rate (%)",
                0.0, 50.0, 20.0
            ) / 100
        
        with col2:
            st.subheader("EQ-5D Parameters")
            baseline_score = st.number_input(
                "Baseline EQ-5D Score",
                value=0.64, min_value=-0.594, max_value=1.0
            )
            score_sd = st.number_input(
                "EQ-5D Standard Deviation",
                value=0.26, min_value=0.0, max_value=1.0
            )
            baseline_correlation = st.slider(
                "Baseline-Outcome Correlation",
                0.0, 1.0, 0.5
            )
            mcid = st.number_input(
                "Minimal Clinically Important Difference",
                value=0.074, min_value=0.0, max_value=1.0
            )
        
        effect_sizes = st.multiselect(
            "Effect Sizes to Analyze",
            options=[0.074, 0.10, 0.15, 0.20],
            default=[0.074, 0.10, 0.15, 0.20]
        )
        
        if st.button("Calculate Power Analysis"):
            results = power_calculator.calculate_power_analysis(
                effect_sizes, power, alpha, score_sd,
                baseline_correlation, cluster_size,
                icc, dropout_rate
            )
            
            # Create and display power table
            df = power_calculator.create_power_table(results)
            st.subheader("Power Analysis Results")
            st.dataframe(df, use_container_width=True)
            
            # Create power curve plot
            st.subheader("Power Curves")
            fig, ax = plt.subplots(figsize=(12, 8))
            powers = np.linspace(0.5, 0.99, 100)
            colors = ['#FF6B6B', '#45B7D1', '#4ECDC4', '#96CEB4']
            
            for i, effect_size in enumerate(effect_sizes):
                sample_sizes = []
                for p in powers:
                    z_alpha = stats.norm.ppf(1 - alpha/2)
                    z_beta = stats.norm.ppf(p)
                    variance_adjusted = 2 * score_sd**2 * (1 - baseline_correlation)
                    base_n = 2 * ((z_alpha + z_beta)**2 * variance_adjusted) / (effect_size**2)
                    adjusted_n = base_n * (1 + ((cluster_size - 1) * icc)) / (1 - dropout_rate)
                    sample_sizes.append(adjusted_n)
                
                plt.plot(
                    sample_sizes,
                    powers,
                    label=f'Effect size {effect_size:.3f}',
                    color=colors[i % len(colors)],
                    linewidth=2.5
                )
            
            plt.xlabel('Sample Size per Group')
            plt.ylabel('Statistical Power')
            plt.title(f'Power Analysis for EQ-5D Study\nICC = {icc:.2f}, α = {alpha}')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend(title='Effect Sizes')
            
            st.pyplot(fig)

if __name__ == "__main__":
    main()
