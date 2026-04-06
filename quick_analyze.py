import pandas as pd

df = pd.read_csv('/app/data/processed/epi_data.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

print("=" * 70)
print("📊 Analyse du fichier epi_data.csv")
print("=" * 70)

print(f"\nTotal enregistrements: {len(df):,}")
print(f"Colonnes: {', '.join(df.columns)}")

print("\n📈 Distribution par maladie:")
for disease in ['Paludisme', 'Choléra', 'Tuberculose', 'VIH/SIDA']:
    count = len(df[df['disease'] == disease])
    print(f"  {disease}: {count:,}")

print("\n📅 Période couverte:")
if df['date'].notna().any():
    print(f"  Début: {df['date'].min().date()}")
    print(f"  Fin: {df['date'].max().date()}")
    months = (df['date'].max() - df['date'].min()).days / 30.44
    print(f"  Durée: {months:.1f} mois")

# Vérifier colonne zone
zone_cols = [c for c in df.columns if 'zone' in c.lower() or 'health' in c.lower() or 'name' in c.lower()]
print(f"\n🗺️  Colonnes géographiques trouvées: {zone_cols}")

if 'health_zone' in df.columns:
    zones = df['health_zone'].nunique()
    print(f"  Zones uniques: {zones:,}")

# Afficher premières lignes pour référence
print("\n📋 Premières lignes:")
print(df.head(10).to_string())
