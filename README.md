# NEXUS - Autonomous AI Model Development Platform

NEXUS is a powerful, modular, autonomous AI model development system that can understand
natural language requests, plan work, discover models and datasets, allocate cloud compute,
train/fine-tune models, evaluate results, improve them and export the finished model.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Documentation](#documentation)
- [Examples](#examples)
- [Security](#security)
- [Development](#development)
- [License](#license)

## Overview

NEXUS handles the complete model development workflow:

1. **Understand** - Parse natural language requests into structured training objectives
2. **Discover** - Find suitable base models and datasets from Hugging Face
3. **Plan** - Create reproducible execution plans with resource estimation
4. **Train** - Fine-tune models on cloud GPUs (Kaggle, Colab)
5. **Evaluate** - Benchmark before/after performance with safety testing
6. **Improve** - Identify weaknesses and perform additional training
7. **Export** - Save adapters, tokenizers, and training metadata
8. **Publish** - Upload to Hugging Face Hub (with explicit approval)

### Example User Request

> "Create a language model under 1 billion parameters that is good at Python, JavaScript and conversational chat."

NEXUS autonomously handles all steps from this single request.

## Key Features

| Feature | Description |
|---------|-------------|
| **Autonomous Orchestration** | Complete workflow from request to deployed model |
| **Multi-Cloud Support** | Kaggle, Colab, Hugging Face providers |
| **Free-First** | No paid services by default |
| **Modular Architecture** | Easy to extend and maintain |
| **Security-First** | Token protection, approval gates, audit logging |
| **Resumable Jobs** | Checkpointing and job persistence |
| **Sandboxed Evaluation** | Restricted code execution for safety |

## Architecture

```
NEXUS/
├── nexus/
│   ├── core/           # Config, engine, logger, state, exceptions
│   ├── providers/      # Kaggle, Colab, Hugging Face, fake providers
│   ├── models/         # Discovery, selection, registry, validator, downloader
│   ├── datasets/       # Discovery, validator, cleaner, formatter, registry
│   ├── training/       # Preflight, LoRA, QLoRA, trainer, checkpoint, notebook generator
│   ├── evaluation/     # Sandbox, code/chat/reasoning/safety benchmarks, regression, reports
│   ├── orchestration/  # Planner, scheduler, job manager, workflow executor, approval gate
│   └── utils/          # Files, hardware, security, validation
├── config/             # settings.yaml, providers.yaml, training.yaml
├── workspace/          # models, datasets, jobs, outputs
├── docs/               # Documentation (guide.md, example.md)
├── tests/              # 150+ unit and integration tests
├── logs/               # Application logs
└── .env                # Environment variables (HF_TOKEN, etc.)
```

## Installation

### Prerequisites

- Python 3.8+
- Hugging Face account (for model/dataset access)
- Kaggle account (optional, for GPU compute)
- Google account (optional, for Colab GPU)

### Install

```bash
# Clone the repository
cd NEXUS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install NEXUS
pip install -e .

# Install ML dependencies (optional, for local development)
pip install -e .[ml]
```

### Configure

```bash
# Copy environment template
cp .env.example .env

# Add your Hugging Face token
echo "HF_TOKEN=hf_your_token_here" >> .env
```

## Quick Start

```bash
# 1. Check system status
nexus doctor

# 2. List available providers
nexus providers

# 3. Plan a workflow
nexus plan "Create a small conversational programming assistant under 1B parameters, specializing in Python and JavaScript"

# 4. Create a training job
nexus create "Python/JavaScript Coding Assistant" \
  --description "Small conversational programming assistant" \
  --provider kaggle

# 5. List jobs
nexus jobs

# 6. View logs
nexus logs <JOB_ID>

# 7. Check provider quotas
nexus quota

# 8. Train (requires cloud credentials)
nexus train <JOB_ID>

# 9. Evaluate
nexus evaluate <JOB_ID>

# 10. Export
nexus export <JOB_ID>

# 11. Publish (requires explicit approval)
nexus publish <JOB_ID>
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `nexus doctor` | Check system status and configuration |
| `nexus status` | Display system status (alias for doctor) |
| `nexus config` | Show effective configuration |
| `nexus providers` | List configured providers |
| `nexus jobs` | List all training jobs |
| `nexus create NAME` | Create a new training job |
| `nexus plan OBJECTIVE` | Plan a workflow for an objective |
| `nexus train JOB_ID` | Train a model for a job |
| `nexus evaluate JOB_ID` | Evaluate a trained model |
| `nexus export JOB_ID` | Export a trained model |
| `nexus publish JOB_ID` | Publish to Hugging Face (requires approval) |
| `nexus approve REQ_ID APPROVER` | Approve a pending request |
| `nexus approvals` | List pending approval requests |
| `nexus logs JOB_ID` | View job logs and events |
| `nexus quota` | Show provider quota information |

## Documentation

- **[User Guide](docs/guide.md)** - Complete usage guide
- **[Example Project](docs/example.md)** - Step-by-step walkthrough

## Examples

See [docs/example.md](docs/example.md) for a complete walkthrough of creating a
Python/JavaScript coding assistant under 1 billion parameters.

### Example: Create a Coding Assistant

```bash
# Plan
nexus plan "Create a small conversational programming assistant under 1B parameters"

# Create job
nexus create "Code Assistant" \
  --description "Python/JavaScript coding assistant" \
  --provider kaggle

# Check status
nexus jobs

# View logs
nexus logs <JOB_ID>
```

## Security

### Token Protection

- Hugging Face tokens stored in `.env` only
- Tokens never logged, printed, or committed to version control
- `.env` is in `.gitignore`
- Token redaction in all log outputs

### Approval Gates

The following operations require explicit user approval:

- Publishing models to Hugging Face Hub
- Accessing private datasets
- Destructive operations (delete models/datasets)
- Using paid cloud resources
- Changing account settings

### Sandbox Execution

All untrusted code evaluation runs in a restricted sandbox:
- No filesystem access outside temp directory
- No network access
- No subprocess spawning
- Import restrictions (only safe modules allowed)
- Timeout limits

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_e2e.py

# With coverage
pytest --cov=nexus --cov-report=html
```

### Project Structure

```
NEXUS/
├── nexus/              # Main package
│   ├── core/           # Config, engine, logger, state
│   ├── providers/      # Cloud provider integrations
│   ├── models/         # Model management
│   ├── datasets/       # Dataset management
│   ├── training/       # Training orchestration
│   ├── evaluation/     # Model evaluation
│   ├── orchestration/  # Workflow management
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── docs/               # Documentation
├── tests/              # Test suite (150+ tests)
├── workspace/          # Working directory
├── logs/               # Application logs
├── .env                # Environment variables
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
├── pyproject.toml      # Package configuration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Support

For issues, feature requests, or questions:
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and community

---

Built with ❤️ for autonomous AI model development.