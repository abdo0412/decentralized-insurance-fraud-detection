
from pathlib import Path
import sys

# Trouver la racine du projet
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from agents.text_agent import TextAgent

# Chemin du modèle
MODEL_PATH = ROOT_DIR / "models" / "distilbert_insurance_fraud"

# Initialiser le Text Agent
print("Chargement du Text Agent...")

agent = TextAgent(model_path=MODEL_PATH)

print("Text Agent charge avec succes !")

# Déclaration d'assurance à analyser
narrative = """
The accident occurred at night.
No witnesses were present.
The driver reported the incident several days later.
The vehicle suffered severe damage.
"""

# Faire une prédiction
result = agent.predict(
    claim_id="TEST001",
    narrative=narrative
)

# Afficher le résultat
print("\n===== RESULTAT DU TEXT AGENT =====")
print(result.to_json())

# Tester l'explication
print("\n===== EXPLICATION =====")
print(agent.explain(narrative))
