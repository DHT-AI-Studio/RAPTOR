"""The default Standard Policy Format policy — the 13 official Llama Guard 3
hazard categories (S1-S13; S14 "Code Interpreter Abuse" is a later addition,
not part of Llama Guard 3's original taxonomy, so it's excluded here).

Single source of truth, reused by several call sites so they can never drift
apart: `GET /guardrail/policies/template?format=standard`
(app/routers/guardrail_policies.py), the "no active policy" fallback
(app/engine/policy_store.py::ensure_default_policy_active(), called both at
startup and after a runtime deactivate leaves nothing active), and the
standalone `scripts/seed_default_policy.py`.

DEFAULT_POLICY_NAME/DEFAULT_POLICY_VERSION identify this specific policy row
(by name, regardless of version) so it can be found again and protected from
deletion — see policy_store.py's delete_policy().
"""
from __future__ import annotations

DEFAULT_POLICY_NAME = "default-standard-policy"
DEFAULT_POLICY_VERSION = "1.0"

DEFAULT_STANDARD_POLICY: list[dict] = [
    {
        "id": "S1", "name": "violent_crimes",
        "description": "Content that enables, encourages, or endorses the commission of violent crimes, "
                        "including content that provides instructions on how to commit such crimes.",
        "severity": "critical", "decision": "block",
        "criteria": ["Requests or provides instructions for committing violence against a person or group.",
                     "Praises, glorifies, or expresses intent to commit a violent crime."],
        "exceptions": [("Fictional, historical, or news-reporting discussion of violence with no actionable "
                       "instructions or endorsement.")],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S2", "name": "non_violent_crimes",
        "description": "Content that enables, encourages, or endorses the commission of non-violent crimes, "
                        "such as fraud, theft, drug trafficking, or cybercrime.",
        "severity": "high", "decision": "block",
        "criteria": ["Requests or provides instructions for committing a non-violent crime.",
                     "Praises or expresses intent to commit a non-violent crime."],
        "exceptions": ["Academic, journalistic, or general-knowledge discussion with no actionable instructions."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S3", "name": "sex_crimes",
        "description": "Content that enables, encourages, or endorses the commission of sex crimes, "
                        "including sexual assault and trafficking.",
        "severity": "critical", "decision": "block",
        "criteria": ["Requests or provides instructions for committing a sexual crime.",
                     "Praises or expresses intent to commit a sexual crime."],
        "exceptions": [],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S4", "name": "child_exploitation",
        "description": "Content that depicts, endorses, or enables the sexual abuse or exploitation of minors.",
        "severity": "critical", "decision": "block",
        "criteria": ["Sexualizes a minor in any way.",
                     "Requests, provides, or facilitates content depicting child sexual abuse."],
        "exceptions": [],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S5", "name": "defamation",
        "description": "Content that publishes verifiably false statements about a real, identifiable "
                        "individual likely to injure their reputation.",
        "severity": "medium", "decision": "block",
        "criteria": ["States a specific false factual claim about a named, real person as though true.",
                     "The claim would materially harm that person's reputation."],
        "exceptions": ["Clearly labeled satire, opinion, or fiction not presented as fact."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S6", "name": "specialized_advice",
        "description": "Content that provides dangerous professional (medical, legal, or financial) advice "
                        "without appropriate qualification, where following it could cause real harm.",
        "severity": "medium", "decision": "block",
        "criteria": [("Gives specific medical, legal, or financial advice presented as safe/reliable "
                     "without appropriate caveats, where acting on it could cause harm.")],
        "exceptions": ["General educational information clearly framed as not a substitute for professional advice."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S7", "name": "privacy",
        "description": "Content that exposes or facilitates the exposure of a private individual's "
                        "sensitive personal information without consent.",
        "severity": "high", "decision": "block",
        "criteria": [("Requests or discloses non-public personal information (address, phone, SSN, "
                     "financial/medical records) about a specific private individual.")],
        "exceptions": ["Information about public figures already legitimately public, or the user's own information."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S8", "name": "intellectual_property",
        "description": "Content that violates the intellectual property rights of a third party.",
        "severity": "medium", "decision": "block",
        "criteria": ["Reproduces substantial copyrighted material verbatim without authorization.",
                     "Facilitates circumventing IP protections (e.g. DRM removal, piracy)."],
        "exceptions": ["Short quotations, fair-use commentary, or discussion about IP law itself."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S9", "name": "indiscriminate_weapons",
        "description": "Content that enables the creation of weapons with the potential for mass casualties "
                        "(chemical, biological, radiological, nuclear, or high-yield explosives).",
        "severity": "critical", "decision": "block",
        "criteria": ["Provides actionable instructions for creating or deploying a CBRNE weapon.",
                     "Requests such instructions with apparent intent to act on them."],
        "exceptions": ["General historical or policy discussion with no actionable technical detail."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S10", "name": "hate",
        "description": "Content that demeans or dehumanizes people on the basis of a protected attribute "
                        "such as race, religion, gender, or sexual orientation.",
        "severity": "high", "decision": "block",
        "criteria": ["Dehumanizes, demeans, or incites hatred against a group based on a protected attribute.",
                     "Uses slurs or hateful stereotypes to attack such a group."],
        "exceptions": ["Academic or critical discussion of hate speech that does not itself demean anyone."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S11", "name": "self_harm",
        "description": "Content that enables, encourages, or endorses acts of intentional self-harm, "
                        "including suicide and eating disorders.",
        "severity": "critical", "decision": "block",
        "criteria": ["Provides instructions or encouragement for self-harm or suicide.",
                     "Praises or romanticizes self-harm."],
        "exceptions": ["Supportive, resource-directing responses to someone expressing distress."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S12", "name": "sexual_content",
        "description": "Content depicting sexually explicit acts, intended to arouse sexual excitement.",
        "severity": "medium", "decision": "block",
        "criteria": ["Depicts sexually explicit acts in graphic detail for the purpose of arousal."],
        "exceptions": ["Clinical, educational, or artistic content without explicit graphic intent."],
        "examples": {"violation": [], "allowed": []},
    },
    {
        "id": "S13", "name": "elections",
        "description": "Content that provides factually incorrect information about the administration or "
                        "outcome of a civic election, e.g. false voting procedures or dates.",
        "severity": "medium", "decision": "block",
        "criteria": [("States false factual information about voting procedures, eligibility, dates, or "
                     "official election results as though true.")],
        "exceptions": ["Opinion/commentary about elections that doesn't assert false facts."],
        "examples": {"violation": [], "allowed": []},
    },
]
