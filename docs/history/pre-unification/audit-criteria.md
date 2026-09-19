# Audit Criteria

## Scope
- Targets: Chatbot_v1, Chatbot_v2, Chatbot_v3, Chatbot_v4, Chatbot_v5.final, Chatbot_v6, onTology_system_v9, ontology_system_v11, ontology_system_v12, ontology_system_v13
- Focus: folder structure and file layout only (names and placement)
- Exclusions: .git metadata, venv, __pycache__, .pytest_cache, node_modules, build artifacts

## Evidence Requirements
- Use Get-ChildItem to capture full directory lists for all targets (with exclusions applied).
- Use Get-ChildItem to capture top-level file lists for each target.
- No assumptions beyond observed paths and names.

## Analysis Checklist
- For each version: core problem, structural traits, meaningful change vs previous, rationale for the choice.
- Across versions: maturity direction, complexity inflection points, structure-first improvements.
- Strengths: interview-ready statements.
- Improvements: growth-oriented critique.
- Hiring summary: one-line summary, developer traits, hire rationale.

## Constraints
- Do not infer runtime behavior without file evidence.
- Flag uncertainty explicitly.
- Stay within the provided scope.