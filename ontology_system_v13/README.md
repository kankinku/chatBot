# Ontology System (Core)

Minimal, domain-focused ontology pipeline for extracting, validating, updating, and reasoning over relations.

## Overview

Pipeline:

1. Extraction: fragments -> entities -> relations
2. Validation: schema/sign/semantic checks
3. Domain Update: static guard + dynamic update
4. Reasoning: graph retrieval -> path fusion -> conclusion

## Structure

```
src/
  domain/        # domain ontology update logic
  extraction/    # fragment/entity/relation extraction
  validation/    # schema/sign/semantic validation
  reasoning/     # query parsing and reasoning
  storage/       # graph repository backends
  llm/           # optional LLM adapters
  shared/        # shared models/exceptions
config/          # ontology schema/config
main.py          # entry point
```

## Configuration

- `config/entity_types.yaml`
- `config/relation_types.yaml`
- `config/alias_dictionary.yaml`
- `config/validation_schema.yaml`
- `config/static_domain.yaml`
- `config/infrastructure.yaml` (storage/LLM backends)

## Data

- `data/domain/entities.json` (canonical entities)
- `data/domain/relations.json` (domain relations)
- `data/samples/sample_documents.json` (optional sample texts)

## Run

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

If you do not want to use an LLM:

```bash
python main.py --no_llm
```
