import os
import yaml

from ..models.intent import Intent, IntentTaxonomy
from ..core.config import get_settings


def load_intent_taxonomy(taxonomy_path: str = None) -> IntentTaxonomy:
    if taxonomy_path is None:
        taxonomy_path = get_settings().INTENT_TAXONOMY_PATH

    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"Intent taxonomy not found at {taxonomy_path}")

    try:
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        intents = []
        for item in data.get("intents", []):
            intents.append(Intent(
                id=item.get("id", ""),
                description=item.get("description", ""),
                target_pipeline=item.get("target_pipeline", ""),
                keywords=[str(kw) for kw in (item.get("keywords") or [])],
                examples=[str(ex) for ex in (item.get("examples") or [])],
            ))

        return IntentTaxonomy(intents=intents)

    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse intent taxonomy YAML: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading taxonomy: {e}")
