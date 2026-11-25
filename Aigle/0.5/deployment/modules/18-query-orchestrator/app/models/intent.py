from dataclasses import dataclass, field


@dataclass
class Intent:
    id: str = field(default_factory=str)
    description: str = ""
    target_pipeline: str = ""
    keywords: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class IntentTaxonomy:
    intents: list[Intent]

    def get_intent(self, intent_id: str) -> Intent | None:
        for intent in self.intents:
            if intent.id == intent_id:
                return intent
        return None
