# Bundled Benchmark Corpus

Offline-capable benchmark data for the Guardian SDK benchmark suite.
No internet connection or HuggingFace login required.

## Files

### `benign_prompts.jsonl`
275 curated benign prompts across 11 real-world domains (25 each):

| Category | Count | Description |
|---|---|---|
| `customer_service` | 25 | Account, billing, shipping queries |
| `coding` | 25 | Programming questions, code requests |
| `health` | 25 | General health and wellness questions |
| `general_knowledge` | 25 | Facts, geography, history |
| `creative_writing` | 25 | Writing requests, copywriting |
| `math` | 25 | Calculations, word problems |
| `technology` | 25 | Tech concepts, system design |
| `cooking` | 25 | Recipes, cooking techniques |
| `travel` | 25 | Trip planning, destination questions |
| `finance` | 25 | Financial concepts (non-sensitive) |
| `science` | 25 | Scientific concepts and explanations |

All prompts are legitimate and should be classified as **benign (label=0)**.

### `attack_prompts.jsonl`
141 curated attack prompts covering 12 attack categories used in published benchmark studies:

| Category | Description |
|---|---|
| `direct_injection` | Explicit override/ignore instructions |
| `system_prompt_extraction` | Attempts to reveal the system prompt |
| `jailbreak_roleplay` | Roleplay-based filter bypass (DAN, AIM, etc.) |
| `jailbreak_hypothetical` | Hypothetical framing to bypass restrictions |
| `social_engineering` | Authority/urgency/empathy manipulation |
| `indirect_injection` | Injections hidden in document/email content |
| `identity_manipulation` | Attempts to alter the AI's identity |
| `goal_hijacking` | Persistent goal/priority overrides |
| `obfuscation` | Encoded, rotated, or disguised instructions |
| `token_manipulation` | Special tokens, XML tags, prompt delimiters |
| `adversarial_probing` | Mapping vulnerabilities and failure modes |
| `data_exfiltration` | Attempts to extract internal data/credentials |

All prompts are real-world attack patterns and should be classified as **attacks (label=1)**.

## Format

Each line is a JSON object:
```json
{"text": "...", "category": "category_name"}
```

## Sourcing Notes

Benign prompts were manually curated to represent genuine user queries across
domains commonly seen in production LLM deployments (customer service bots,
coding assistants, general-purpose chatbots).

Attack prompts were collected and synthesized from:
- Published academic literature (Perez & Ribeiro 2022; Greshake et al. 2023)
- Public jailbreak databases (JailbreakBench, PromptInject)
- Red-team exercises and adversarial ML research

No benign prompts contain harmful content. No attack prompts contain working
instructions for producing harmful real-world outcomes — they test the injection/
jailbreak detection capability of Guardian, not the LLM's harmfulness defenses.
