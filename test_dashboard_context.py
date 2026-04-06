import requests
import re

resp = requests.get('http://localhost:8000/analysis/dashboard/')
print(f"Status: {resp.status_code}")

# Chercher les labels de graphique
if 'chart_tier_labels' in resp.text:
    print("✅ chart_tier_labels trouvé dans HTML")
    # Extraire la ligne
    for line in resp.text.split('\n'):
        if 'chart_tier_labels' in line:
            print(f"  Ligne: {line.strip()[:100]}")
else:
    print("❌ chart_tier_labels NON trouvé")

# Chercher anomalies
if 'system_anomalies' in resp.text:
    print("✅ system_anomalies trouvé")
else:
    print("❌ system_anomalies NON trouvé")

# Sauvegarder un extrait
with open('/tmp/dashboard_test.html', 'w') as f:
    f.write(resp.text[:5000])
print("\n📄 HTML sauvé dans /tmp/dashboard_test.html")
