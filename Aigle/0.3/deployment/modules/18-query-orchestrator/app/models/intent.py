from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Intent:
    id: str = field(default_factory=str)
    description: str = ""
    target_pipeline: str = ""
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class IntentTaxonomy:
    intents: List[Intent]

    def get_intent(self, intent_id: str) -> Optional[Intent]:
        for intent in self.intents:
            if intent.id == intent_id:
                return intent
        return None
