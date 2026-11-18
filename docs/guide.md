# NEXUS Documentation

NEXUS is an autonomous AI model development platform that handles the complete
model development workflow from user request to deployed model.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Architecture](#architecture)
6. [Providers](#providers)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Security](#security)
10. [Examples](#examples)

## Overview

NEXUS autonomously:

1. Understands user requests in natural language
2. Converts them into structured training objectives
3. Researches and selects suitable base models
4. Discovers and validates datasets
5. Plans and executes training on cloud GPUs
6. Evaluates models before and after training
7. Exports and optionally publishes models

### Key Features

- **Autonomous orchestration**: From request to deployed model
- **Multi-cloud support**: Kaggle, Colab, Hugging Face
- **Free-first infrastructure**: No paid services by default
- **Modular architecture**: Easy to extend and maintain
- **Security-focused**: Token protection, approval gates, audit logging
- **Resumable jobs**: Checkpointing and job persistence

## Installation

### Prerequisites

- Python 3.8+
- Hugging Face account (for model/dataset access)
- Kaggle account (for GPU compute, optional)
- Google account (for Colab GPU, optional)

### Install NEXUS

```bash
# Clone the repository
git clone <repository-url>
cd NEXUS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install NEXUS
pip install -e .

# Install ML dependencies (optional, for local development)
pip install -e .[ml]
```

### Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# At minimum, set HF_TOKEN
echo "HF_TOKEN=your_huggingface_token" >> .env
```

## Quick Start

### 1. Check System Status

```bash
nexus doctor
```

### 2. List Available Providers

```bash
nexus providers
```

### 3. Create a Training Job

```bash
nexus create "Python and JavaScript coding assistant under 1B parameters" \
  --description "Small conversational programming assistant" \
  --provider kaggle
```

### 4. Plan a Workflow

```bash
nexus plan "Create a small conversational programming assistant under 1B parameters, specializing in Python and JavaScript"
```

### 5. List Jobs

```bash
nexus jobs
```

## CLI Reference

### `nexus doctor`

Check system status and configuration.

### `nexus status`

Display NEXUS system status (alias for `doctor`).

### `nexus config`

Display current configuration.

### `nexus providers`

Display provider information and capabilities.

### `nexus jobs`

List all training jobs with their status.

### `nexus create NAME`

Create a new training job.

Options:
- `--description`, `-d`: Job description
- `--provider`, `-p`: Provider to use (kaggle, colab, huggingface)
- `--config`, `-c`: Configuration file path

### `nexus plan OBJECTIVE`

Plan a workflow for the given objective.

Example:
```bash
nexus plan "Create a conversational model under 1B parameters"
```

### `nexus train JOB_ID`

Train a model for the given job.

### `nexus evaluate JOB_ID`

Evaluate a trained model.

### `nexus export JOB_ID`

Export a trained model (adapter, tokenizer, metadata).

### `nexus publish JOB_ID`

Publish a trained model to Hugging Face (requires approval).

### `nexus approve REQUEST_ID APPROVER`

Approve a pending approval request.

### `nexus approvals`

List pending approval requests.

### `nexus logs JOB_ID`

Display job logs and events.

### `nexus quota`

Display provider quota information.

## Architecture

```
NEXUS/
├── nexus/
│   ├── core/           # Core infrastructure (config, engine, logger, state)
│   ├── providers/      # Cloud provider integrations
│   ├── models/         # Model discovery & management
│   ├── datasets/       # Dataset discovery & processing
│   ├── training/       # Training orchestration
│   ├── evaluation/     # Model evaluation
│   ├── orchestration/  # Workflow management
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── workspace/          # Working directory
├── tests/              # Test suite
└── logs/               # Application logs
```

### Core Components

- **Config**: YAML + environment variable loading with precedence
- **Engine**: Central orchestrator wiring all components
- **Logger**: Structured JSON/text logging with token redaction
- **State**: Job persistence with state machine and event tracking

### Provider Layer

- **ComputeProvider**: Abstract base class for cloud providers
- **KaggleProvider**: Kaggle GPU compute via notebooks
- **ColabProvider**: Google Colab notebook generation
- **HuggingFaceProvider**: Model/dataset hub integration
- **FakeProvider**: Mock implementations for testing

### Training Layer

- **PreflightChecker**: GPU feasibility and resource estimation
- **LoRATrainer**: LoRA fine-tuning configuration
- **QLoRATrainer**: 4-bit quantized LoRA configuration
- **BaseTrainer**: Training loop with checkpointing
- **CheckpointManager**: Checkpoint save/load/resume/sync
- **NotebookGenerator**: Colab/Kaggle notebook generation

### Evaluation Layer

- **SandboxExecutor**: Restricted code execution (no host access)
- **CodeEvaluator**: Code generation benchmarks
- **ChatEvaluator**: Conversational quality evaluation
- **ReasoningEvaluator**: Logical reasoning evaluation
- **SafetyEvaluator**: Toxicity, bias, safety testing
- **RegressionTester**: Baseline vs fine-tuned comparison
- **EvaluationReporter**: Markdown/JSON report generation

### Orchestration Layer

- **WorkflowPlanner**: Intent parsing and execution planning
- **JobScheduler**: Cross-provider job scheduling
- **OrchestrationJobManager**: Job lifecycle management
- **WorkflowExecutor**: End-to-end workflow execution
- **ApprovalGate**: User approval gates for sensitive operations

## Providers

### Kaggle

- GPU: Tesla T4 (14.56 GB VRAM)
- Max runtime: 168 hours/week
- Authentication: Kaggle API credentials
- Notebook-based execution

### Google Colab

- GPU: Tesla T4 (~12 GB VRAM)
- Max runtime: 12 hours/session
- Authentication: Google OAuth
- Manual notebook execution required

### Hugging Face

- Model/dataset hub
- No GPU compute
- Authentication: HF API token
- Model/dataset upload/download

## Training

### Supported Methods

- **SFT**: Supervised Fine-tuning (full parameters)
- **LoRA**: Low-rank Adaptation (parameter-efficient)
- **QLoRA**: Quantized LoRA (4-bit, memory-efficient)
- **PEFT**: Parameter-efficient Fine-tuning

### Training Configuration

Training configurations are specified in `config/training.yaml`:

```yaml
training:
  default_method: qlora
  default_batch_size: 2
  default_learning_rate: 0.0001
  default_epochs: 3
  default_max_sequence_length: 2048
```

### Preflight Checks

Before training, NEXUS performs preflight checks:

- GPU memory feasibility
- Method compatibility
- Quota availability
- Resource estimation

## Evaluation

### Benchmarks

- **Code generation**: Python and JavaScript code correctness
- **Chat quality**: Conversational coherence and relevance
- **Safety**: Toxicity, bias, harmful content detection
- **Reasoning**: Logical reasoning, math word problems
- **Regression**: Baseline vs fine-tuned comparison

### Sandbox Execution

All code execution happens in a sandbox with:
- No filesystem access outside temp directory
- No network access
- No subprocess spawning
- Import restrictions
- Timeout limits

## Security

### Token Protection

- Tokens stored in environment variables only
- Never logged or printed
- Redacted in all outputs
- Never committed to version control

### Approval Gates

The following operations require explicit user approval:

- Publishing models to Hugging Face
- Accessing private datasets
- Destructive operations
- Using paid resources
- Changing account settings

### Sandbox

All untrusted code execution happens in a restricted sandbox.

## Examples

### Example 1: Create a Coding Assistant

```bash
# Plan the workflow
nexus plan "Create a Python coding assistant under 1B parameters"

# Create a job
nexus create "Python coding assistant" \
  --description "Small model for Python code generation" \
  --provider kaggle

# Check job status
nexus jobs

# View logs
nexus logs <JOB_ID>
```

### Example 2: Full Workflow

```bash
# 1. Check status
nexus doctor

# 2. Check providers
nexus providers

# 3. Plan a workflow
nexus plan "Conversational programming assistant under 1B parameters"

# 4. Create a job
nexus create "Code assistant" --provider kaggle

# 5. Train (requires cloud credentials)
nexus train <JOB_ID>

# 6. Evaluate
nexus evaluate <JOB_ID>

# 7. Export
nexus export <JOB_ID>

# 8. Publish (requires approval)
nexus publish <JOB_ID>
```

### Example 3: Approval Workflow

```bash
# Try to publish (will request approval)
nexus publish <JOB_ID>
# Output: Approval required. Request ID: approval_123456

# Approve the request
nexus approve approval_123456 your_username

# Try again
nexus publish <JOB_ID>
```

## License

MIT License. See LICENSE file for details.

## Support

For support, issues, or feature requests:
- GitHub Issues
- Documentation: README updates
- Feature requests: GitHub Discussions