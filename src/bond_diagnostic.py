import pandas as pd
import numpy as np
from pathlib import Path

def diagnose_bond_data():
    """
    Comprehensive bond data diagnostics
    """
    # Load data
    data_file = Path("data/processed/regime_labeled_dataset.csv")
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print("="*80)
    print("BOND DATA COMPREHENSIVE DIAGNOSTICS")
    print("="*80)
    
    # 1. Basic info
    print("\n1. DATASET INFO:")
    print(f"   Total rows: {len(df)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # 2. Bond return statistics (monthly)
    bond_data = df['bond_return'].dropna()
    print("\n2. MONTHLY BOND RETURNS:")
    print(f"   Count: {len(bond_data)}")
    print(f"   Mean: {bond_data.mean()*100:.4f}%")
    print(f"   Median: {bond_data.median()*100:.4f}%")
    print(f"   Std Dev: {bond_data.std()*100:.4f}%")
    print(f"   Min: {bond_data.min()*100:.2f}% (date: {df.loc[bond_data.idxmin(), 'date']})")
    print(f"   Max: {bond_data.max()*100:.2f}% (date: {df.loc[bond_data.idxmax(), 'date']})")
    print(f"   25th %ile: {bond_data.quantile(0.25)*100:.2f}%")
    print(f"   75th %ile: {bond_data.quantile(0.75)*100:.2f}%")
    
    # 3. Check for data issues
    print("\n3. DATA QUALITY CHECKS:")
    print(f"   Missing values: {df['bond_return'].isna().sum()}")
    print(f"   Zero values: {(df['bond_return'] == 0).sum()}")
    print(f"   Values > 10%: {(abs(df['bond_return']) > 0.10).sum()}")
    print(f"   Values > 20%: {(abs(df['bond_return']) > 0.20).sum()}")
    
    # 4. Extreme months
    print("\n4. TOP 10 EXTREME POSITIVE MONTHS:")
    extreme_positive = df.nlargest(10, 'bond_return')[['date', 'bond_return']]
    for idx, row in extreme_positive.iterrows():
        print(f"   {row['date'].strftime('%Y-%m')}: {row['bond_return']*100:+.2f}%")
    
    print("\n5. TOP 10 EXTREME NEGATIVE MONTHS:")
    extreme_negative = df.nsmallest(10, 'bond_return')[['date', 'bond_return']]
    for idx, row in extreme_negative.iterrows():
        print(f"   {row['date'].strftime('%Y-%m')}: {row['bond_return']*100:+.2f}%")
    
    # 5. Sample of actual data
    print("\n6. FIRST 20 ROWS (date and bond_return):")
    print(df[['date', 'bond_return']].head(20).to_string(index=False))
    
    print("\n7. LAST 20 ROWS (date and bond_return):")
    print(df[['date', 'bond_return']].tail(20).to_string(index=False))
    
    # 6. Annual volatility calculation
    print("\n8. ANNUAL STATISTICS:")
    df['year'] = df['date'].dt.year
    for year in sorted(df['year'].unique())[-10:]:  # Last 10 years
        year_data = df[df['year'] == year]['bond_return']
        annual_return = (1 + year_data).prod() - 1
        print(f"   {year}: Return={annual_return*100:+.2f}%, Monthly StdDev={year_data.std()*100:.2f}%")
    
    # 7. Check if it's decimal vs percentage issue
    print("\n9. DECIMAL FORMAT CHECK:")
    sample_values = bond_data.head(10).values
    print(f"   Raw values (first 10): {sample_values}")
    print(f"   If these are decimals (0.05 = 5%): Looks correct")
    print(f"   If these should be percentages (5.0 = 5%): DIVIDE BY 100")
    
    # 8. Compare to equity volatility
    equity_data = df['equity_return'].dropna()
    print("\n10. COMPARISON TO EQUITY:")
    print(f"   Bond monthly std: {bond_data.std()*100:.2f}%")
    print(f"   Equity monthly std: {equity_data.std()*100:.2f}%")
    print(f"   Ratio (Bond/Equity): {bond_data.std()/equity_data.std():.2f}")
    print(f"   Expected ratio: ~0.3-0.5 for normal bonds")
    
    # 9. Annualized volatility
    bond_annual_vol = bond_data.std() * np.sqrt(12) * 100
    equity_annual_vol = equity_data.std() * np.sqrt(12) * 100
    print("\n11. ANNUALIZED VOLATILITY:")
    print(f"   Bond: {bond_annual_vol:.2f}%")
    print(f"   Equity: {equity_annual_vol:.2f}%")
    
    # 10. Check metadata columns
    print("\n12. OTHER COLUMNS (might indicate data source):")
    for col in df.columns:
        if 'bond' in col.lower() or 'regime' in col.lower():
            print(f"   {col}: {df[col].dtype}")
            if df[col].dtype == 'object':
                print(f"      Sample values: {df[col].dropna().unique()[:5]}")

if __name__ == "__main__":
    diagnose_bond_data()