# Git Hooks

These git hooks ensure clean commits without specific vendor references or Co-Authored-By lines.

## Setup

Run the setup script to install the hooks:

```bash
./setup-hooks.sh
```

Or manually:

```bash
git config core.hooksPath .githooks
```

## Hooks Included

### pre-commit
- Checks for prohibited vendor references in code
- Ensures generic provider naming (provider_a, provider_b)
- Validates no vendor-specific terms are committed

### commit-msg
- Validates commit message format
- Removes vendor-related references from commit messages
- Ensures no Co-Authored-By lines except for Christopher Bratkovics

### prepare-commit-msg
- Automatically removes any Co-Authored-By lines before commit
- Prevents assistant attribution

## Bypass Hooks (Emergency Only)

If you need to bypass hooks temporarily:

```bash
git commit --no-verify -m "your message"
```

## Prohibited Terms

The hooks check for:
- Specific vendor names (use provider_a, provider_b instead)
- Specific model names (use model-3.5-turbo format)
- Vendor-specific terminology (use "Chatbot System")
- Co-Authored-By lines (except Christopher Bratkovics)