
"""
Part B: Text Agent (DistilBERT).

Owner: Person B.
Scores how suspicious a written claim narrative is in [0, 1]
using a fine-tuned DistilBERT model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

from agents.base import AgentMessage, BaseAgent


class TextAgent(BaseAgent):
    """Specialist Text Agent using a fine-tuned DistilBERT model."""

    agent_name = "text"
    model_version = "distilbert-claim-fraud-v1"

    # Seuil sélectionné sur le jeu Validation
    THRESHOLD = 0.10

    # Même longueur maximale que pendant l'entraînement
    MAX_LENGTH = 128

    def __init__(
        self,
        model_path: str | Path = "models/distilbert_insurance_fraud",
        device: str = "auto",
    ) -> None:

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        resolved_path = Path(model_path)

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Model not found at {resolved_path}. "
                "Check that the trained model is saved in models/."
            )

        # Chargement du tokenizer et du modèle entraîné
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            str(resolved_path)
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(
          str(resolved_path),
          attn_implementation="eager"

        )

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        claim_id: str,
        narrative: str = "",
        **kwargs: Any,
    ) -> AgentMessage:

        if not isinstance(narrative, str):
            narrative = ""

        # 1. Tokenisation
        inputs = self.tokenizer(
            narrative,
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # 2. Prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)

            probabilities = F.softmax(outputs.logits, dim=-1)

            # Classe 1 = fraude
            suspicion_score = probabilities[0, 1].item()

        # 3. Score et décision
        label = (
            "suspicious"
            if suspicion_score >= self.THRESHOLD
            else "normal"
        )

        confidence = max(
            suspicion_score,
            1.0 - suspicion_score
        )

        # 4. Indices textuels simples
        suspicious_cues = [
            "no witnesses",
            "no police",
            "urgent",
            "cash",
            "total loss",
        ]

        triggered = [
            cue
            for cue in suspicious_cues
            if cue in narrative.lower()
        ]

        return AgentMessage(
            claim_id=str(claim_id),
            agent=self.agent_name,
            output={
                "suspicion_score": round(suspicion_score, 4),
                "label": label,
                "top_tokens": triggered[:5],
            },
            confidence=round(confidence, 4),
            model_version=self.model_version,
        )

    def explain(self, narrative: str) -> Dict[str, float]:
        """
        Returns approximate attention weights for the input tokens.
        These weights are indicative and are not a formal explanation
        of the model's decision.
        """

        inputs = self.tokenizer(
            narrative,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_LENGTH,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
            )

        # Shape par couche :
        # [batch, heads, sequence_length, sequence_length]
        attentions = torch.stack(outputs.attentions)

        # Moyenne sur les couches et les têtes
        mean_attention = attentions.mean(dim=0).mean(dim=1)

        # Moyenne sur les positions des tokens qui observent
        weights = mean_attention[0].mean(dim=0)

        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        # Exclure les tokens spéciaux
        special_tokens = {
            self.tokenizer.cls_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
        }

        token_weights = [
            (token, weight.item())
            for token, weight in zip(tokens, weights)
            if token not in special_tokens
        ]

        total = sum(weight for _, weight in token_weights)

        if total == 0:
            return {}

        return {
            token: round(weight / total, 4)
            for token, weight in token_weights
        }
