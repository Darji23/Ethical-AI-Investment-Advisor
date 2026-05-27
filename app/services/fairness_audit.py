"""
Fairness Audit System
Tests for demographic parity and bias across age groups, experience levels, and income brackets
"""
import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from app.schemas.schemas import InvestorProfile, Product, PortfolioRecommendation
from app.services.suitability_validator import SuitabilityValidator

class FairnessAuditor:
    """
    Enhanced auditor for bias detection across multiple demographic dimensions
    """
    
    def __init__(self, validator: SuitabilityValidator):
        self.validator = validator
    
    def run_comprehensive_simulation(
        self, 
        profiles: List[InvestorProfile], 
        test_recommendations: List[PortfolioRecommendation]
    ) -> Dict:
        """
        Run suitability checks across all profiles and recommendations
        """
        # Define demographic categories
        age_brackets = {
            'young_adult': (22, 35),
            'middle_aged': (36, 50),
            'pre_retirement': (51, 64),
            'retirement': (65, 100)
        }
        
        income_brackets = {
            'low': (0, 60000),
            'medium': (60001, 120000),
            'high': (120001, 250000),
            'very_high': (250001, 10000000)
        }
        
        experience_levels = ['beginner', 'intermediate', 'advanced']
        
        # Initialize result storage
        results = {
            'by_age': defaultdict(list),
            'by_experience': defaultdict(list),
            'by_income': defaultdict(list),
            'by_risk_tolerance': defaultdict(list),
            'raw_results': []
        }
        
        # Run validation for each profile-recommendation pair
        total_tests = 0
        for profile in profiles:
            age_group = self._get_age_group(profile.age, age_brackets)
            income_group = self._get_income_group(profile.annual_income, income_brackets)
            experience_group = profile.investment_experience
            risk_group = f"risk_{profile.risk_tolerance}"
            
            for rec in test_recommendations:
                result = self.validator.validate_recommendation(profile, rec)
                
                test_result = {
                    'investor_id': profile.investor_id,
                    'age': profile.age,
                    'age_group': age_group,
                    'income': profile.annual_income,
                    'income_group': income_group,
                    'experience': experience_group,
                    'risk_tolerance': profile.risk_tolerance,
                    'risk_group': risk_group,
                    'is_suitable': result.is_suitable,
                    'approved_count': len(result.approved_products),
                    'rejected_count': len(result.rejected_products),
                    'violation_count': len(result.risk_violations),
                    'portfolio_risk': result.overall_portfolio_risk,
                    'recommendation': rec.rationale
                }
                
                # Store in all relevant categories
                results['by_age'][age_group].append(test_result)
                results['by_experience'][experience_group].append(test_result)
                results['by_income'][income_group].append(test_result)
                results['by_risk_tolerance'][risk_group].append(test_result)
                results['raw_results'].append(test_result)
                
                total_tests += 1
        
        results['_metadata'] = {
            'total_profiles': len(profiles),
            'total_recommendations': len(test_recommendations),
            'total_tests': total_tests
        }
        
        return results
    
    def _get_age_group(self, age: int, brackets: Dict) -> str:
        """Categorize age into bracket"""
        for group, (min_age, max_age) in brackets.items():
            if min_age <= age <= max_age:
                return group
        return 'unknown'
    
    def _get_income_group(self, income: float, brackets: Dict) -> str:
        """Categorize income into bracket"""
        for group, (min_income, max_income) in brackets.items():
            if min_income <= income <= max_income:
                return group
        return 'unknown'
    
    def calculate_comprehensive_metrics(self, simulation_results: Dict) -> Dict:
        """
        Calculate detailed fairness metrics across all demographic dimensions
        """
        metrics = {
            'by_age': {},
            'by_experience': {},
            'by_income': {},
            'by_risk_tolerance': {},
            'overall_fairness': {}
        }
        
        for dimension in ['by_age', 'by_experience', 'by_income', 'by_risk_tolerance']:
            dimension_data = simulation_results[dimension]
            dimension_metrics = {}
            
            for group, results in dimension_data.items():
                total = len(results)
                approved = sum(1 for r in results if r['is_suitable'])
                
                dimension_metrics[group] = {
                    'total_tests': total,
                    'approval_rate': approved / total if total > 0 else 0,
                    'rejection_rate': (total - approved) / total if total > 0 else 0,
                    'avg_violations': np.mean([r['violation_count'] for r in results]),
                    'avg_approved_products': np.mean([r['approved_count'] for r in results]),
                    'avg_rejected_products': np.mean([r['rejected_count'] for r in results]),
                    'avg_portfolio_risk': np.mean([r['portfolio_risk'] for r in results if r['portfolio_risk'] > 0]),
                    'std_portfolio_risk': np.std([r['portfolio_risk'] for r in results if r['portfolio_risk'] > 0])
                }
            
            # Calculate demographic parity for this dimension
            approval_rates = [m['approval_rate'] for m in dimension_metrics.values()]
            if approval_rates:
                max_rate = max(approval_rates)
                min_rate = min(approval_rates)
                parity_gap = max_rate - min_rate
                
                dimension_metrics['_summary'] = {
                    'demographic_parity_gap': parity_gap,
                    'max_approval_rate': max_rate,
                    'min_approval_rate': min_rate,
                    'mean_approval_rate': np.mean(approval_rates),
                    'std_approval_rate': np.std(approval_rates),
                    'parity_threshold': 0.20,
                    'passes_fairness_test': parity_gap < 0.20
                }
            
            metrics[dimension] = dimension_metrics
        
        # Calculate overall fairness score
        parity_gaps = []
        for dimension in ['by_age', 'by_experience', 'by_income']:
            if '_summary' in metrics[dimension]:
                parity_gaps.append(metrics[dimension]['_summary']['demographic_parity_gap'])
        
        if parity_gaps:
            metrics['overall_fairness'] = {
                'average_parity_gap': np.mean(parity_gaps),
                'max_parity_gap': max(parity_gaps),
                'dimensions_passing': sum(1 for gap in parity_gaps if gap < 0.20),
                'total_dimensions': len(parity_gaps),
                'overall_pass': all(gap < 0.20 for gap in parity_gaps),
                'fairness_score': max(0, 100 - (np.mean(parity_gaps) * 100))
            }
        
        return metrics
    
    def generate_detailed_report(self, metrics: Dict, simulation_results: Dict) -> str:
        """Generate comprehensive human-readable fairness report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE FAIRNESS AUDIT REPORT")
        report.append("=" * 80)
        report.append("")
        
        metadata = simulation_results['_metadata']
        report.append(f"Dataset Size:")
        report.append(f"  • Investor Profiles: {metadata['total_profiles']}")
        report.append(f"  • Test Recommendations: {metadata['total_recommendations']}")
        report.append(f"  • Total Validation Tests: {metadata['total_tests']}")
        report.append("")
        
        report.append("─" * 80)
        report.append("1. AGE-BASED FAIRNESS ANALYSIS")
        report.append("─" * 80)
        self._add_dimension_report(report, metrics['by_age'], 'Age Group')
        
        report.append("─" * 80)
        report.append("2. EXPERIENCE-BASED FAIRNESS ANALYSIS")
        report.append("─" * 80)
        self._add_dimension_report(report, metrics['by_experience'], 'Experience Level')
        
        report.append("─" * 80)
        report.append("3. INCOME-BASED FAIRNESS ANALYSIS")
        report.append("─" * 80)
        self._add_dimension_report(report, metrics['by_income'], 'Income Bracket')
        
        report.append("─" * 80)
        report.append("4. RISK TOLERANCE ANALYSIS (Informational)")
        report.append("─" * 80)
        self._add_dimension_report(report, metrics['by_risk_tolerance'], 'Risk Level', is_fairness_dimension=False)
        
        report.append("=" * 80)
        report.append("OVERALL FAIRNESS VERDICT")
        report.append("=" * 80)
        
        overall = metrics['overall_fairness']
        report.append(f"Fairness Score: {overall['fairness_score']:.1f}/100")
        report.append(f"Average Parity Gap: {overall['average_parity_gap']:.1%}")
        report.append(f"Maximum Parity Gap: {overall['max_parity_gap']:.1%}")
        report.append(f"Dimensions Passing: {overall['dimensions_passing']}/{overall['total_dimensions']}")
        report.append("")
        
        if overall['overall_pass']:
            report.append("✅ RESULT: SYSTEM PASSES COMPREHENSIVE FAIRNESS TEST")
            report.append("")
            report.append("The validation system demonstrates demographic parity across all")
            report.append("protected dimensions (age, experience, income). Approval rate")
            report.append("differences are within acceptable industry standards (<20%).")
        else:
            report.append("❌ RESULT: FAIRNESS CONCERNS DETECTED")
            report.append("")
            report.append("The validation system shows significant disparities in approval")
            report.append("rates across demographic groups. Further investigation and")
            report.append("calibration recommended.")
            
            report.append("")
            report.append("Problematic Dimensions:")
            for dimension in ['by_age', 'by_experience', 'by_income']:
                if '_summary' in metrics[dimension]:
                    summary = metrics[dimension]['_summary']
                    if not summary['passes_fairness_test']:
                        gap = summary['demographic_parity_gap']
                        report.append(f"  • {dimension.replace('by_', '').title()}: {gap:.1%} gap (threshold: 20%)")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def _add_dimension_report(self, report: List[str], dimension_metrics: Dict, label: str, is_fairness_dimension: bool = True):
        """Helper to add dimension-specific metrics to report"""
        groups = {k: v for k, v in dimension_metrics.items() if not k.startswith('_')}
        
        for group, metrics in sorted(groups.items()):
            report.append(f"{label}: {group.upper().replace('_', ' ')}")
            report.append(f"  Total Tests: {metrics['total_tests']}")
            report.append(f"  Approval Rate: {metrics['approval_rate']:.1%}")
            report.append(f"  Rejection Rate: {metrics['rejection_rate']:.1%}")
            report.append(f"  Avg Violations per Test: {metrics['avg_violations']:.2f}")
            report.append(f"  Avg Approved Products: {metrics['avg_approved_products']:.1f}")
            report.append(f"  Avg Portfolio Risk: {metrics['avg_portfolio_risk']:.2f}/5.0")
            report.append("")
        
        if '_summary' in dimension_metrics:
            summary = dimension_metrics['_summary']
            report.append(f"DEMOGRAPHIC PARITY ANALYSIS:")
            report.append(f"  Parity Gap: {summary['demographic_parity_gap']:.1%}")
            report.append(f"  Range: {summary['min_approval_rate']:.1%} to {summary['max_approval_rate']:.1%}")
            report.append(f"  Mean: {summary['mean_approval_rate']:.1%} (±{summary['std_approval_rate']:.1%})")
            report.append(f"  Threshold: {summary['parity_threshold']:.1%}")
            
            if is_fairness_dimension:
                if summary['passes_fairness_test']:
                    report.append(f"  ✅ PASSES: Demographic parity maintained")
                else:
                    report.append(f"  ❌ FAILS: Significant disparity detected")
            else:
                report.append(f"  ℹ️  Note: Risk tolerance differences are expected and not a fairness concern")
        report.append("")
    
    def export_results(self, metrics: Dict, simulation_results: Dict, filename: str = 'data/fairness_audit_detailed.json'):
        """Export detailed results to JSON for further analysis"""
        export_data = {
            'metrics': metrics,
            'metadata': simulation_results['_metadata'],
            'summary': {
                'fairness_score': metrics['overall_fairness']['fairness_score'],
                'passes_all_tests': metrics['overall_fairness']['overall_pass'],
                'dimensions_tested': ['age', 'experience', 'income'],
                'recommendation': 'APPROVED' if metrics['overall_fairness']['overall_pass'] else 'NEEDS_REVIEW'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename


def generate_diverse_test_recommendations() -> List[PortfolioRecommendation]:
    """
    Generate 8 test portfolios covering full risk spectrum
    """
    return [
        # Ultra-conservative
        PortfolioRecommendation(
            recommended_products=["GOVT", "SHV"],
            allocation_percentages=[60.0, 40.0],
            rationale="Ultra-safe treasury focus for capital preservation"
        ),
        
        # Conservative
        PortfolioRecommendation(
            recommended_products=["GOVT", "VBTLX", "BND"],
            allocation_percentages=[40.0, 35.0, 25.0],
            rationale="Conservative fixed-income portfolio"
        ),
        
        # Moderate-conservative
        PortfolioRecommendation(
            recommended_products=["AGG", "VTI", "VBTLX"],
            allocation_percentages=[50.0, 30.0, 20.0],
            rationale="Balanced portfolio with bond focus"
        ),
        
        # Moderate
        PortfolioRecommendation(
            recommended_products=["VTI", "VWELX", "VBTLX"],
            allocation_percentages=[50.0, 30.0, 20.0],
            rationale="Balanced growth and income"
        ),
        
        # Moderate-aggressive
        PortfolioRecommendation(
            recommended_products=["VTI", "QQQ", "VGT", "VXUS"],
            allocation_percentages=[40.0, 30.0, 20.0, 10.0],
            rationale="Growth-focused with tech tilt"
        ),
        
        # Aggressive
        PortfolioRecommendation(
            recommended_products=["ARKK", "QQQ", "VTI", "SOXX"],
            allocation_percentages=[35.0, 30.0, 20.0, 15.0],
            rationale="High-growth technology focus"
        ),
        
        # Very aggressive
        PortfolioRecommendation(
            recommended_products=["ARKK", "ARKW", "QQQ", "BITW"],
            allocation_percentages=[30.0, 30.0, 25.0, 15.0],
            rationale="Aggressive innovation and crypto exposure"
        ),
        
        # Extremely aggressive (should fail for most)
        PortfolioRecommendation(
            recommended_products=["BITW", "SPXL", "ARKW"],
            allocation_percentages=[40.0, 35.0, 25.0],
            rationale="Maximum risk crypto and leveraged strategy"
        ),
    ]
