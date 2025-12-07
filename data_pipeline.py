"""
DATA PIPELINE: Manufacturer Trust Evaluation System
====================================================
Demonstrates complete data workflow:
- Multiple data source merging
- Synthetic data generation
- Data preprocessing
- Feature engineering
- Data validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MANUFACTURER TRUST EVALUATION SYSTEM - DATA PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: SIMULATE DATA FROM MULTIPLE SOURCES
# ============================================================================
print("\n📊 STEP 1: LOADING DATA FROM MULTIPLE SOURCES")
print("-" * 80)

# Source 1: Financial Database (SAP)
print("\n1️⃣  Loading from Financial Database (SAP)...")
np.random.seed(42)
n_records_financial = 8000

financial_data = pd.DataFrame({
    'manufacturer_id': range(1, n_records_financial + 1),
    'source': 'SAP_Financial_DB',
    'revenue_million': np.random.uniform(10, 500, n_records_financial),
    'profit_margin_percent': np.random.uniform(5, 35, n_records_financial),
    'debt_to_equity_ratio': np.random.uniform(0.1, 3.0, n_records_financial),
    'operational_efficiency': np.random.uniform(0.5, 1.0, n_records_financial),
})
print(f"   ✅ Loaded {len(financial_data):,} records from Financial DB")
print(f"   📌 Columns: {list(financial_data.columns)}")

# Source 2: Operational Database (Manufacturing ERP)
print("\n2️⃣  Loading from Operational Database (Manufacturing ERP)...")
n_records_operational = 7500

operational_data = pd.DataFrame({
    'manufacturer_id': np.random.choice(range(1, 8001), n_records_operational, replace=True),
    'source': 'Manufacturing_ERP',
    'defect_rate_percent': np.random.uniform(0.1, 10, n_records_operational),
    'on_time_delivery_rate': np.random.uniform(60, 100, n_records_operational),
    'capacity_utilization': np.random.uniform(40, 100, n_records_operational),
    'warranty_claim_rate': np.random.uniform(0.1, 5, n_records_operational),
})
print(f"   ✅ Loaded {len(operational_data):,} records from Manufacturing ERP")
print(f"   📌 Columns: {list(operational_data.columns)}")

# Source 3: Compliance Database
print("\n3️⃣  Loading from Compliance Database...")
n_records_compliance = 7000

compliance_data = pd.DataFrame({
    'manufacturer_id': np.random.choice(range(1, 8001), n_records_compliance, replace=True),
    'source': 'Compliance_Registry',
    'compliance_score': np.random.uniform(20, 100, n_records_compliance),
    'regulatory_violation_count': np.random.randint(0, 10, n_records_compliance),
    'certifications_count': np.random.randint(1, 20, n_records_compliance),
    'contract_breach_count': np.random.randint(0, 5, n_records_compliance),
})
print(f"   ✅ Loaded {len(compliance_data):,} records from Compliance DB")
print(f"   📌 Columns: {list(compliance_data.columns)}")

# Source 4: Customer Satisfaction Database
print("\n4️⃣  Loading from Customer Satisfaction Database...")
n_records_customer = 6500

customer_data = pd.DataFrame({
    'manufacturer_id': np.random.choice(range(1, 8001), n_records_customer, replace=True),
    'source': 'CRM_Customer_DB',
    'customer_rating': np.random.uniform(2, 5, n_records_customer),
    'after_sales_service_rating': np.random.uniform(2, 5, n_records_customer),
    'delivery_timeliness': np.random.uniform(0, 100, n_records_customer),
})
print(f"   ✅ Loaded {len(customer_data):,} records from Customer DB")
print(f"   📌 Columns: {list(customer_data.columns)}")

# Source 5: Supply Chain Database
print("\n5️⃣  Loading from Supply Chain Database...")
n_records_supply_chain = 7200

supply_chain_data = pd.DataFrame({
    'manufacturer_id': np.random.choice(range(1, 8001), n_records_supply_chain, replace=True),
    'source': 'Supply_Chain_DB',
    'market_share_percent': np.random.uniform(0.1, 15, n_records_supply_chain),
    'supplier_diversity': np.random.uniform(0, 100, n_records_supply_chain),
    'supply_chain_resilience': np.random.uniform(0, 100, n_records_supply_chain),
    'employee_count': np.random.randint(50, 5000, n_records_supply_chain),
})
print(f"   ✅ Loaded {len(supply_chain_data):,} records from Supply Chain DB")
print(f"   📌 Columns: {list(supply_chain_data.columns)}")

print(f"\n📈 TOTAL RECORDS LOADED: {len(financial_data) + len(operational_data) + len(compliance_data) + len(customer_data) + len(supply_chain_data):,}")

# ============================================================================
# STEP 2: MERGE DATA FROM MULTIPLE SOURCES
# ============================================================================
print("\n\n🔗 STEP 2: MERGING DATA FROM MULTIPLE SOURCES")
print("-" * 80)

print("\n   Merging Financial Database...")
merged_data = financial_data.copy()
print(f"   Shape after Financial: {merged_data.shape}")

print("   Merging Operational Database...")
merged_data = merged_data.merge(operational_data, on='manufacturer_id', how='left', suffixes=('', '_ops'))
print(f"   Shape after Operational: {merged_data.shape}")

print("   Merging Compliance Database...")
merged_data = merged_data.merge(compliance_data, on='manufacturer_id', how='left', suffixes=('', '_comp'))
print(f"   Shape after Compliance: {merged_data.shape}")

print("   Merging Customer Satisfaction Database...")
merged_data = merged_data.merge(customer_data, on='manufacturer_id', how='left', suffixes=('', '_cust'))
print(f"   Shape after Customer: {merged_data.shape}")

print("   Merging Supply Chain Database...")
merged_data = merged_data.merge(supply_chain_data, on='manufacturer_id', how='left', suffixes=('', '_sc'))
print(f"   Shape after Supply Chain: {merged_data.shape}")

print(f"\n✅ Merged Data Summary:")
print(f"   Total Records: {len(merged_data):,}")
print(f"   Total Columns: {len(merged_data.columns)}")
print(f"   Missing Values Before Handling: {merged_data.isnull().sum().sum():,}")

# ============================================================================
# STEP 3: ADD SYNTHETIC DATA FOR MISSING FEATURES
# ============================================================================
print("\n\n🤖 STEP 3: GENERATING SYNTHETIC DATA FOR MISSING FEATURES")
print("-" * 80)

synthetic_features = {
    'sustainability_index': np.random.uniform(0, 100, len(merged_data)),
    'innovation_index': np.random.uniform(0, 100, len(merged_data)),
    'it_system_maturity': np.random.uniform(1, 5, len(merged_data)),
    'training_hours_per_employee': np.random.uniform(10, 100, len(merged_data)),
    'carbon_intensity': np.random.uniform(10, 500, len(merged_data)),
    'renewable_energy_ratio': np.random.uniform(0, 100, len(merged_data)),
    'it_incident_rate': np.random.uniform(0, 10, len(merged_data)),
    'data_quality_score': np.random.uniform(50, 100, len(merged_data)),
    'digitization_level': np.random.uniform(0, 100, len(merged_data)),
    'average_lead_time_days': np.random.randint(5, 60, len(merged_data)),
    'currency_volatility_index': np.random.uniform(0, 100, len(merged_data)),
    'partnership_years': np.random.randint(1, 30, len(merged_data)),
    'legal_dispute_count': np.random.randint(0, 3, len(merged_data)),
    'region_risk_index': np.random.uniform(0, 100, len(merged_data)),
}

print(f"\n   Generated {len(synthetic_features)} synthetic features:")
for feature, values in synthetic_features.items():
    merged_data[feature] = values
    print(f"   ✅ {feature:35s} - Range: [{values.min():.2f}, {values.max():.2f}]")

# ============================================================================
# STEP 4: DATA CLEANING & PREPROCESSING
# ============================================================================
print("\n\n🧹 STEP 4: DATA CLEANING & PREPROCESSING")
print("-" * 80)

# 4.1 Remove duplicate rows
print("\n   4.1 Removing Duplicates...")
initial_rows = len(merged_data)
merged_data = merged_data.drop_duplicates(subset=['manufacturer_id'])
print(f"   ✅ Removed {initial_rows - len(merged_data)} duplicate rows")

# 4.2 Handle missing values
print("\n   4.2 Handling Missing Values...")
missing_before = merged_data.isnull().sum().sum()
print(f"   Missing values before: {missing_before}")

# Drop unnecessary 'source' columns
merged_data = merged_data.drop(['source', 'source_ops', 'source_comp', 'source_cust', 'source_sc'], 
                                 axis=1, errors='ignore')

# Forward fill for numerical columns
numerical_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numerical_cols] = merged_data[numerical_cols].fillna(merged_data[numerical_cols].mean())

missing_after = merged_data.isnull().sum().sum()
print(f"   Missing values after: {missing_after}")
print(f"   ✅ Imputation complete")

# 4.3 Remove outliers using IQR method
print("\n   4.3 Removing Outliers (IQR Method)...")
outliers_removed = 0
for col in numerical_cols:
    Q1 = merged_data[col].quantile(0.25)
    Q3 = merged_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before = len(merged_data)
    merged_data = merged_data[(merged_data[col] >= lower_bound) & (merged_data[col] <= upper_bound)]
    outliers_removed += before - len(merged_data)

print(f"   ✅ Removed {outliers_removed} outlier records")

# 4.4 Data type conversion
print("\n   4.4 Converting Data Types...")
integer_cols = ['manufacturer_id', 'certifications_count', 'regulatory_violation_count', 
                'contract_breach_count', 'legal_dispute_count', 'employee_count', 
                'average_lead_time_days', 'partnership_years']
for col in integer_cols:
    if col in merged_data.columns:
        merged_data[col] = merged_data[col].astype('int64')
        print(f"   ✅ {col:35s} → int64")

# ============================================================================
# STEP 5: FEATURE NORMALIZATION & SCALING
# ============================================================================
print("\n\n📊 STEP 5: FEATURE NORMALIZATION & SCALING")
print("-" * 80)

# 5.1 Min-Max Normalization for percentage/ratio features
print("\n   5.1 Min-Max Normalization (0-1 range)...")
percentage_cols = ['profit_margin_percent', 'on_time_delivery_rate', 'capacity_utilization',
                   'market_share_percent', 'renewable_energy_ratio', 'supplier_diversity',
                   'supply_chain_resilience', 'sustainability_index', 'innovation_index',
                   'delivery_timeliness', 'digitization_level', 'data_quality_score']

scaler_minmax = MinMaxScaler(feature_range=(0, 1))
for col in percentage_cols:
    if col in merged_data.columns:
        merged_data[col] = scaler_minmax.fit_transform(merged_data[[col]])

print(f"   ✅ Normalized {len(percentage_cols)} percentage features to [0, 1]")

# 5.2 Standardization for continuous features
print("\n   5.2 Standardization (Z-score)...")
continuous_cols = ['revenue_million', 'debt_to_equity_ratio', 'defect_rate_percent',
                   'warranty_claim_rate', 'carbon_intensity', 'it_incident_rate',
                   'currency_volatility_index', 'region_risk_index']

scaler_standard = StandardScaler()
for col in continuous_cols:
    if col in merged_data.columns:
        merged_data[col] = scaler_standard.fit_transform(merged_data[[col]])

print(f"   ✅ Standardized {len(continuous_cols)} continuous features (mean=0, std=1)")

# ============================================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================================
print("\n\n⚙️  STEP 6: FEATURE ENGINEERING")
print("-" * 80)

print("\n   Creating derived features...")

# Quality Score = combination of defect rate and warranty claims
merged_data['quality_score'] = (
    (1 - merged_data['defect_rate_percent'].clip(0, 1)) * 0.6 +
    (1 - merged_data['warranty_claim_rate'].clip(0, 1)) * 0.4
)
print("   ✅ quality_score: Composite of defect & warranty rates")

# Risk Score = combination of violations and disputes
merged_data['risk_score'] = (
    merged_data['regulatory_violation_count'].clip(0, 1) * 0.6 +
    merged_data['legal_dispute_count'].clip(0, 1) * 0.4
)
print("   ✅ risk_score: Composite of violations & disputes")

# Operational Excellence = combination of efficiency metrics
merged_data['operational_excellence'] = (
    merged_data['operational_efficiency'] * 0.3 +
    merged_data['on_time_delivery_rate'].clip(0, 1) * 0.3 +
    merged_data['capacity_utilization'].clip(0, 1) * 0.2 +
    merged_data['delivery_timeliness'].clip(0, 1) * 0.2
)
print("   ✅ operational_excellence: Combined efficiency score")

# Innovation & Sustainability Index
merged_data['innovation_sustainability'] = (
    merged_data['innovation_index'].clip(0, 1) * 0.5 +
    merged_data['sustainability_index'].clip(0, 1) * 0.5
)
print("   ✅ innovation_sustainability: Combined innovation & sustainability")

# ============================================================================
# STEP 7: TARGET VARIABLE CALCULATION
# ============================================================================
print("\n\n🎯 STEP 7: CALCULATING TARGET VARIABLE (TRUST SCORE)")
print("-" * 80)

# Trust Score = weighted combination of key factors
merged_data['trust_score'] = (
    merged_data['compliance_score'].clip(0, 1) * 0.20 +
    merged_data['operational_efficiency'] * 0.15 +
    merged_data['on_time_delivery_rate'].clip(0, 1) * 0.15 +
    merged_data['quality_score'] * 0.15 +
    (1 - merged_data['risk_score']) * 0.10 +
    merged_data['customer_rating'].clip(0, 1) * 0.10 +
    merged_data['operational_excellence'] * 0.10 +
    merged_data['digitization_level'].clip(0, 1) * 0.05
)

# Add slight noise to make it realistic
merged_data['trust_score'] = merged_data['trust_score'] + np.random.normal(0, 0.02, len(merged_data))
merged_data['trust_score'] = merged_data['trust_score'].clip(0, 1)

# Create trust status categories
merged_data['trust_status'] = pd.cut(merged_data['trust_score'],
                                     bins=[0, 0.3, 0.6, 0.7, 1.0],
                                     labels=['POOR', 'FAIR', 'GOOD', 'EXCELLENT'])

print(f"\n   Trust Score Distribution:")
print(f"   Mean: {merged_data['trust_score'].mean():.3f}")
print(f"   Median: {merged_data['trust_score'].median():.3f}")
print(f"   Std Dev: {merged_data['trust_score'].std():.3f}")
print(f"   Min: {merged_data['trust_score'].min():.3f}")
print(f"   Max: {merged_data['trust_score'].max():.3f}")

print(f"\n   Trust Status Distribution:")
print(merged_data['trust_status'].value_counts())

# ============================================================================
# STEP 8: DATA VALIDATION
# ============================================================================
print("\n\n✅ STEP 8: DATA VALIDATION")
print("-" * 80)

print("\n   8.1 Schema Validation...")
print(f"   Total Rows: {len(merged_data):,}")
print(f"   Total Columns: {len(merged_data.columns)}")
print(f"   Memory Usage: {merged_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\n   8.2 Data Quality Checks...")
print(f"   ✅ No duplicate IDs: {merged_data['manufacturer_id'].is_unique}")
print(f"   ✅ No missing values: {merged_data.isnull().sum().sum() == 0}")
print(f"   ✅ Trust score range [0,1]: {merged_data['trust_score'].min() >= 0 and merged_data['trust_score'].max() <= 1}")

print("\n   8.3 Column Information:")
for i, col in enumerate(merged_data.columns, 1):
    dtype = merged_data[col].dtype
    print(f"   {i:2d}. {col:35s} - {str(dtype):15s}")

# ============================================================================
# STEP 9: SAVE PROCESSED DATA
# ============================================================================
print("\n\n💾 STEP 9: SAVING PROCESSED DATA")
print("-" * 80)

# Save to CSV
output_filename = 'manufacturers_processed_data.csv'
merged_data.to_csv(output_filename, index=False)
print(f"\n   ✅ Saved to: {output_filename}")

# Save data summary
summary_filename = 'data_processing_summary.txt'
with open(summary_filename, 'w') as f:
    f.write("DATA PROCESSING SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Final Dataset Shape: {merged_data.shape}\n")
    f.write(f"Total Records: {len(merged_data):,}\n")
    f.write(f"Total Features: {len(merged_data.columns)}\n")
    f.write(f"\nFeatures:\n")
    for col in merged_data.columns:
        f.write(f"  - {col}\n")
    f.write(f"\nTrust Score Statistics:\n")
    f.write(f"  Mean: {merged_data['trust_score'].mean():.3f}\n")
    f.write(f"  Median: {merged_data['trust_score'].median():.3f}\n")
    f.write(f"  Std Dev: {merged_data['trust_score'].std():.3f}\n")

print(f"   ✅ Saved to: {summary_filename}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("PROCESSING COMPLETE!")
print("=" * 80)
print(f"\n📊 FINAL DATASET:")
print(f"   Records: {len(merged_data):,}")
print(f"   Features: {len(merged_data.columns)}")
print(f"   Files Created:")
print(f"     1. {output_filename}")
print(f"     2. {summary_filename}")
print(f"\n✨ Ready for model training and evaluation!")
print("=" * 80 + "\n")