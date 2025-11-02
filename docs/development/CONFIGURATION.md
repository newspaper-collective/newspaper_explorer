# Configuration Philosophy

## Principle: Environment Variables vs Runtime Flags

This project follows a clear separation between **environment configuration** and **runtime behavior**.

### Environment Variables (`.env` file)

**Purpose**: Configure WHERE and WHAT across different environments (dev/staging/prod)

✅ **Use for:**

- Directory paths (`DATA_DIR`, `DOWNLOAD_DIR`, `EXTRACTED_DIR`)
- Source locations (`SOURCES_DIR`)
- Logging configuration (`LOG_LEVEL`)
- Credentials (if needed in future)
- Database connections (if needed in future)
- API endpoints (if needed in future)

❌ **Don't use for:**

- Runtime behaviors that change per-command
- User preferences that should be explicit
- Actions that should be visible in command history

### CLI Flags

**Purpose**: Control HOW operations are performed at runtime

✅ **Use for:**

- Optional features (`--no-extract`, `--no-fix`)
- Operation modes (`--force`, `--verbose`)
- Output formats (if added in future)
- Behavior toggles that users choose per-command

❌ **Don't use for:**

- Configuration that rarely changes
- System-level settings
- Credentials or secrets

## Current Configuration

### In `.env`:

```bash
# Paths
DATA_DIR=data
DOWNLOAD_DIR=data/downloads
EXTRACTED_DIR=data/extracted
SOURCES_DIR=data/sources

# System settings
LOG_LEVEL=INFO
```

### As CLI flags:

```bash
# Download behavior
--no-extract     # Skip extraction
--no-fix         # Skip error corrections
--force          # Force re-download
--all            # Download all parts

# Display options
--verbose        # Detailed output
```

## Benefits of This Approach

1. **Predictability**: Commands show exactly what they'll do
2. **Auditability**: Command history shows what was run and how
3. **Flexibility**: Different behavior per invocation
4. **Standards**: Follows Unix/CLI best practices
5. **Debugging**: Easier to reproduce issues with explicit flags

## Examples

### ✅ Good: Environment for paths

```bash
# .env
DATA_DIR=/mnt/large-storage/newspapers

# Command remains simple
newspaper-explorer data download --part dertag_1900-1902
```

### ✅ Good: Flags for behavior

```bash
# Different behaviors for different needs
newspaper-explorer data download --part dertag_1900-1902 --no-extract
newspaper-explorer data download --part dertag_1903-1905  # with extraction
```

### ❌ Bad: Environment for behavior

```bash
# .env
AUTO_EXTRACT=false  # Hidden, not visible in command

# Command doesn't show what will happen
newspaper-explorer data download --part dertag_1900-1902
# Will it extract? Have to check .env file!
```

## When to Add New Configuration

### Add to `.env` if:

- [ ] It's about WHERE (paths, URLs)
- [ ] It changes between environments
- [ ] It's rarely modified during development
- [ ] It applies to ALL commands globally

### Add as CLI flag if:

- [ ] It's about HOW (behavior, options)
- [ ] Users might want different values per-command
- [ ] It should be visible in command history
- [ ] It affects only specific operations

## Summary

**Think of it this way:**

- `.env` = Your workshop's location and tools
- CLI flags = How you use those tools for each job

This keeps configuration predictable, explicit, and follows the principle of least surprise.
