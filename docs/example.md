# NEXUS Example Training Project

This example demonstrates the complete NEXUS workflow for creating a
small conversational programming assistant under 1 billion parameters.

## Example: Python/JavaScript Coding Assistant

### Objective

"Create a small conversational programming assistant under 1 billion parameters,
specializing in Python and JavaScript."

### Step-by-Step Walkthrough

#### 1. Check System Status

```bash
$ nexus doctor

NEXUS Status
==================================================
Configuration sources: 3
  - file:settings
  - file:providers
  - file:training

Configured providers: 3
  - kaggle: Enabled
  - colab: Enabled
  - huggingface: Enabled

Workspace: /home/user/NEXUS/workspace
  Exists: True

Configuration: Valid
```

#### 2. Check Provider Availability

```bash
$ nexus providers

NEXUS Providers
==================================================

KAGGLE:
  Description: Kaggle GPU Compute Platform
  Priority: 1
  Timeout: 30 seconds

COLAB:
  Description: Google Colab GPU Platform
  Priority: 2
  Timeout: 30 seconds

HUGGINGFACE:
  Description: Hugging Face Model & Dataset Hub
  Priority: 3
  Timeout: 30 seconds
```

#### 3. Plan the Workflow

```bash
$ nexus plan "Create a small conversational programming assistant under 1 billion parameters, specializing in Python and JavaScript"

Workflow planned successfully!
Objective: Create a small conversational programming assistant under 1 billion parameters, specializing in Python and JavaScript.
Plan ID: plan_1794573924_d776b1f3
Estimated duration: 97 minutes
Estimated GPU hours: 3.5
Provider sequence: kaggle, colab, huggingface
Steps: 15

Execution steps:
  1. Intent parsed into structured objective
  2. Discover suitable base models from Hugging Face
  3. Select best model based on requirements
  4. Discover suitable datasets
  5. Validate dataset licenses and usability
  6. Clean, format, and split training data
  7. Verify GPU feasibility and resource requirements
  8. Select optimal cloud provider
  9. Submit training job to cloud provider
  10. Monitor training execution and checkpoints
  11. Evaluate baseline and fine-tuned model
  12. Additional training if justified
  13. Export trained model and adapter
  14. Generate complete training report
  15. Publish to Hugging Face (requires approval)
```

#### 4. Create a Training Job

```bash
$ nexus create "Python/JavaScript Coding Assistant" \
  --description "Small conversational programming assistant under 1B parameters" \
  --provider kaggle

Job created successfully!
Job ID: job_1794573924_a1b2c3d4
Name: Python/JavaScript Coding Assistant
Description: Small conversational programming assistant under 1B parameters
Provider: kaggle
```

#### 5. Check Job Status

```bash
$ nexus jobs

NEXUS Jobs
==================================================

Total jobs: 1

PENDING (1):
  - Python/JavaScript Coding Assistant (ID: job_1794573924_a1b2c3d4)
    Provider: kaggle
    Progress: 0.0%
```

#### 6. View Logs

```bash
$ nexus logs job_1794573924_a1b2c3d4

Job logs: job_1794573924_a1b2c3d4
==================================================
[2026-11-13 12:00:00] job_created
    job_id: job_1794573924_a1b2c3d4
    name: Python/JavaScript Coding Assistant
    description: Small conversational programming assistant under 1B parameters
    provider: kaggle
```

#### 7. Check Provider Quotas

```bash
$ nexus quota

NEXUS Provider Quotas
==================================================

KAGGLE:
  Type: gpu_hours
  Used: 0.5
  Limit: 30.0
  Remaining: 29.5
  Refresh: 2026-11-07 00:00:00

COLAB:
  Quota information not available

HUGGINGFACE:
  Type: api_calls
  Used: 10.0
  Limit: 60.0
  Remaining: 50.0
  Refresh: 2026-11-13 12:01:00
```

### Complete Workflow Script

Save this as `run_example.sh`:

```bash
#!/bin/bash
set -e

echo "=== NEXUS Example: Python/JavaScript Coding Assistant ==="
echo ""

# Step 1: Check status
echo "[1/7] Checking system status..."
nexus doctor

# Step 2: Plan workflow
echo ""
echo "[2/7] Planning workflow..."
nexus plan "Create a small conversational programming assistant under 1B parameters, specializing in Python and JavaScript"

# Step 3: Create job
echo ""
echo "[3/7] Creating training job..."
JOB_ID=$(nexus create "Python/JavaScript Coding Assistant" \
  --description "Small conversational programming assistant under 1B parameters" \
  --provider kaggle | grep "Job ID:" | awk '{print $3}')
echo "Job ID: $JOB_ID"

# Step 4: Check jobs
echo ""
echo "[4/7] Listing jobs..."
nexus jobs

# Step 5: Check quotas
echo ""
echo "[5/7] Checking provider quotas..."
nexus quota

echo ""
echo "=== Example complete ==="
echo "Job ID: $JOB_ID"
echo "Use 'nexus logs $JOB_ID' to view progress"
echo "Use 'nexus train $JOB_ID' to start training (requires cloud credentials)"
```

Run it:
```bash
chmod +x run_example.sh
./run_example.sh
```

## Expected Output Files

After a complete training run, the workspace will contain:

```
workspace/
├── jobs/
│   └── job_1794573924_a1b2c3d4/
│       ├── index.json
│       └── job_1794573924_a1b2c3d4.json
├── models/
│   └── Qwen2.5-0.5B-Instruct/
│       ├── config.json
│       └── model.safetensors
├── datasets/
│   └── python-code-instructions/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── outputs/
│   └── evaluations/
│       └── evaluation_20261113_120000.json
└── logs/
    └── nexus.log
```

## Next Steps

1. Set up your Hugging Face token in `.env`
2. Configure Kaggle credentials for GPU access
3. Run the example workflow
4. Monitor training progress with `nexus logs`
5. Evaluate the model with `nexus evaluate`
6. Export and optionally publish with approval