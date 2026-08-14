## Bitwarden Integration for Secure Credential Management

## Overview

Agentkit now supports **Bitwarden** for secure API key and credential management. Store your secrets in an encrypted Bitwarden vault instead of `.env` files or hardcoded strings.

### Why Use Bitwarden?

✅ **Encrypted storage** - AES-256 encryption at rest  
✅ **Cross-device sync** - Access credentials anywhere  
✅ **Team sharing** - Securely share with organization members  
✅ **Audit logs** - Track who accessed what  
✅ **Automatic rotation** - Update once, propagates everywhere  
✅ **No accidental commits** - No `.env` files to leak  
✅ **Centralized management** - All credentials in one place  

### vs. Traditional `.env` Files

| Feature | `.env` Files | Bitwarden |
|---------|-------------|-----------|
| Encryption | ❌ Plain text | ✅ AES-256 |
| Sync | ❌ Manual copy | ✅ Automatic |
| Team sharing | ❌ Insecure | ✅ Encrypted |
| Audit trail | ❌ None | ✅ Full logs |
| Rotation | ❌ Manual everywhere | ✅ Update once |
| Git leaks | ⚠️ Easy to commit | ✅ Never in repo |

---

## Installation

### 1. Install Bitwarden CLI

**Linux/Mac:**
```bash
npm install -g @bitwarden/cli
# or
brew install bitwarden-cli
# or
snap install bw
```

**Windows:**
```bash
choco install bitwarden-cli
# or
scoop install bitwarden-cli
```

**Verify installation:**
```bash
bw --version
```

### 2. No Python Dependencies Required

The agentkit Bitwarden integration uses the Bitwarden CLI (already installed above). No additional Python packages needed!

---

## Quick Start

### Step 1: Login to Bitwarden

```bash
bw login your-email@example.com
```

Enter your master password when prompted.

### Step 2: Unlock Your Vault

```bash
bw unlock
```

This outputs a command like:
```bash
export BW_SESSION="ABC123XYZ789..."
```

**Copy and run that export command in your shell.**

### Step 3: Create Items for Your API Keys

**Option A: Using Web/Desktop App (Easier)**

1. Open Bitwarden app or https://vault.bitwarden.com
2. Click "New Item"
3. Fill in:
   - **Name:** `OPENAI_API_KEY`
   - **Type:** Login
   - **Password:** `sk-your-openai-key-here`
4. Save

Repeat for other API keys:
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- etc.

**Option B: Using CLI**

```bash
bw create item \
  --name "OPENAI_API_KEY" \
  --username "" \
  --password "sk-your-openai-key"
```

### Step 4: Use in Your Code

```python
from agentkit import BitwardenSecrets, OpenAIModel, create_agent

# Initialize Bitwarden secrets manager
secrets = BitwardenSecrets()

# Get API key
api_key = secrets.get("OPENAI_API_KEY")

# Use with model
model = OpenAIModel(api_key=api_key)

# Create agent
agent = create_agent(model=model, tools=[...])
```

That's it! Your API keys are now securely retrieved from Bitwarden.

---

## Usage Examples

### Example 1: Basic Secret Retrieval

```python
from agentkit import BitwardenSecrets

secrets = BitwardenSecrets()

# Get a secret by item name
api_key = secrets.get("OPENAI_API_KEY")

# With a default fallback
api_key = secrets.get("OPENAI_API_KEY", default="fallback-key")

# Quick access (creates instance for you)
from agentkit import get_secret
api_key = get_secret("OPENAI_API_KEY")
```

### Example 2: Retrieve Specific Fields

```python
secrets = BitwardenSecrets()

# Get a specific field from an item
api_key = secrets.get_field("Anthropic API", "api_key")
endpoint = secrets.get_field("Custom LLM", "endpoint_url")
```

### Example 3: Structured Credentials (AWS, Azure, etc.)

```python
secrets = BitwardenSecrets()

# Get all fields from an item as a dictionary
aws = secrets.get_item("AWS Credentials")
# Returns: {
#   "access_key_id": "AKIA...",
#   "secret_access_key": "...",
#   "region": "us-east-1"
# }

# Use with AWS
import os
os.environ["AWS_ACCESS_KEY_ID"] = aws["access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = aws["secret_access_key"]
os.environ["AWS_REGION_NAME"] = aws["region"]
```

### Example 4: OpenAI Agent

```python
from agentkit import BitwardenSecrets, OpenAIModel, create_agent, tool

secrets = BitwardenSecrets()

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

model = OpenAIModel(
    model="gpt-4o-mini",
    api_key=secrets.get("OPENAI_API_KEY"),
)

agent = create_agent(model=model, tools=[search])
```

### Example 5: Anthropic Agent

```python
from agentkit import get_secret, AnthropicModel, create_agent

model = AnthropicModel(
    model="claude-sonnet-4-5",
    api_key=get_secret("ANTHROPIC_API_KEY"),
)

agent = create_agent(model=model, tools=[...])
```

### Example 6: LiteLLM with Multiple Providers

```python
from agentkit import BitwardenSecrets, LiteLLMModel

secrets = BitwardenSecrets()

# OpenAI
model = LiteLLMModel(
    model="gpt-4o-mini",
    api_key=secrets.get("OPENAI_API_KEY"),
)

# Anthropic
model = LiteLLMModel(
    model="claude-sonnet-4-5",
    api_key=secrets.get("ANTHROPIC_API_KEY"),
)

# Groq (fast inference)
model = LiteLLMModel(
    model="groq/llama3-70b",
    api_key=secrets.get("GROQ_API_KEY"),
)
```

### Example 7: Azure OpenAI with Structured Credentials

```python
from agentkit import BitwardenSecrets, LiteLLMModel

secrets = BitwardenSecrets()

# Get all Azure credentials at once
azure = secrets.get_item("Azure OpenAI")
# Item in Bitwarden has custom fields:
#   - api_key: your-azure-key
#   - endpoint: https://your-resource.openai.azure.com/

model = LiteLLMModel(
    model="azure/gpt-4",
    api_key=azure["api_key"],
    api_base=azure["endpoint"],
)
```

### Example 8: List Available Secrets

```python
from agentkit import BitwardenSecrets

secrets = BitwardenSecrets()

# List all items in your vault
items = secrets.list_items()
print("Available secrets:", items)
# ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'AWS Credentials', ...]
```

---

## Bitwarden Item Structure

### Simple API Key (Recommended)

**Item Name:** `OPENAI_API_KEY`  
**Type:** Login  
**Password:** `sk-your-actual-key`

The password field is retrieved with `secrets.get("OPENAI_API_KEY")`.

### Multi-Field Credentials

**Item Name:** `AWS Credentials`  
**Type:** Login  
**Custom Fields:**
- `access_key_id`: `AKIA...`
- `secret_access_key`: `wJalr...`
- `region`: `us-east-1`

Retrieved with:
```python
aws = secrets.get_item("AWS Credentials")
access_key = aws["access_key_id"]
secret_key = aws["secret_access_key"]
region = aws["region"]
```

### Best Practices for Naming

✅ **Good:**
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `AWS Credentials`
- `Database Password`

❌ **Avoid:**
- `my key` (spaces, lowercase)
- `temp123` (not descriptive)
- `sk-abc...` (the key itself as name)

---

## API Reference

### `BitwardenSecrets`

Main class for retrieving secrets from Bitwarden.

```python
from agentkit import BitwardenSecrets

secrets = BitwardenSecrets(
    session_key=None,  # Optional: BW_SESSION override
    cache=True,        # Cache vault items (default: True)
)
```

**Parameters:**
- `session_key` (str, optional): Bitwarden session key. Defaults to `BW_SESSION` env var.
- `cache` (bool): Cache vault items to reduce CLI calls. Default: `True`.

#### Methods

##### `get(name, default=None)`

Get a secret by item name.

```python
api_key = secrets.get("OPENAI_API_KEY")
api_key = secrets.get("OPENAI_API_KEY", default="fallback")
```

**Returns:** Secret value from password field, or custom field named `api_key`/`value`/`secret`.

**Raises:** `ConfigurationError` if not found and no default.

##### `get_field(item_name, field_name, default=None)`

Get a specific field from an item.

```python
endpoint = secrets.get_field("Custom LLM", "endpoint_url")
```

**Returns:** Field value.

**Raises:** `ConfigurationError` if item or field not found and no default.

##### `get_item(item_name)`

Get all fields from an item as a dictionary.

```python
aws = secrets.get_item("AWS Credentials")
# Returns: {"access_key_id": "...", "secret_access_key": "...", "region": "..."}
```

**Returns:** Dictionary of field_name → value.

**Raises:** `ConfigurationError` if item not found.

##### `list_items()`

List all item names in the vault.

```python
items = secrets.list_items()
# Returns: ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', ...]
```

##### `clear_cache()`

Clear the cached vault items.

```python
secrets.clear_cache()  # Force reload on next access
```

### `get_secret(name, default=None)`

Convenience function for quick access.

```python
from agentkit import get_secret

api_key = get_secret("OPENAI_API_KEY")
```

Equivalent to:
```python
secrets = BitwardenSecrets()
api_key = secrets.get("OPENAI_API_KEY")
```

---

## Session Management

### Environment Variable (Recommended)

```bash
# Unlock vault
bw unlock

# Copy and run the export command:
export BW_SESSION="ABC123..."

# Now run your Python code
python my_agent.py
```

### Shell Profile (Persistent)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Bitwarden session
export BW_SESSION="your-session-key"
```

**⚠️ Security Note:** Session keys should be kept private. Don't commit them to git.

### Session in Code (Not Recommended)

```python
secrets = BitwardenSecrets(session_key="your-session-key")
```

**Better:** Use environment variable so keys never appear in code.

### Locking When Done

```bash
# Lock your vault when finished working
bw lock
```

This invalidates the session key. Re-run `bw unlock` to get a new one.

---

## Advanced Usage

### Caching

By default, vault items are cached in memory to reduce CLI calls:

```python
secrets = BitwardenSecrets(cache=True)  # Default

# First access: calls bw CLI
api_key = secrets.get("OPENAI_API_KEY")

# Second access: uses cache (fast)
api_key = secrets.get("OPENAI_API_KEY")

# Force reload
secrets.clear_cache()
api_key = secrets.get("OPENAI_API_KEY")  # Calls CLI again
```

### Error Handling

```python
from agentkit import BitwardenSecrets
from agentkit.errors import ConfigurationError

try:
    secrets = BitwardenSecrets()
    api_key = secrets.get("OPENAI_API_KEY")
except ConfigurationError as e:
    print(f"Bitwarden error: {e}")
    # Fallback to environment variable
    api_key = os.getenv("OPENAI_API_KEY")
```

### Fallback to Environment Variables

```python
def get_api_key(name: str) -> str:
    """Try Bitwarden first, fallback to environment."""
    try:
        return get_secret(name)
    except:
        return os.getenv(name)

api_key = get_api_key("OPENAI_API_KEY")
```

### Multiple Vaults/Accounts

```python
# Personal vault
personal = BitwardenSecrets(session_key=os.getenv("BW_SESSION_PERSONAL"))

# Work vault
work = BitwardenSecrets(session_key=os.getenv("BW_SESSION_WORK"))

personal_key = personal.get("Personal OpenAI Key")
work_key = work.get("Work OpenAI Key")
```

---

## Troubleshooting

### Error: "Bitwarden CLI not found"

**Solution:**
```bash
npm install -g @bitwarden/cli
# or
brew install bitwarden-cli
```

Verify: `bw --version`

### Error: "Bitwarden session key not found"

**Solution:**
```bash
bw unlock
# Copy and run the export command it outputs:
export BW_SESSION="ABC123..."
```

### Error: "Secret 'X' not found in Bitwarden vault"

**Solution:**
1. Check item name matches exactly (case-sensitive)
2. Create the item in Bitwarden:
   ```bash
   bw create item --name "OPENAI_API_KEY" --password "sk-..."
   ```
3. Verify it exists:
   ```bash
   bw list items | grep "OPENAI_API_KEY"
   ```

### Error: "You are not logged in"

**Solution:**
```bash
bw login your-email@example.com
```

### Session Expired

Symptoms: CLI commands fail with "Invalid session"

**Solution:**
```bash
bw unlock
# Export the new session key
```

### Permission Denied on CLI

**Solution:**
```bash
# Linux/Mac
chmod +x $(which bw)

# Or reinstall
npm install -g @bitwarden/cli
```

---

## Security Best Practices

### ✅ DO:

- Use `BW_SESSION` environment variable
- Lock your vault when done: `bw lock`
- Use strong master password
- Enable 2FA on your Bitwarden account
- Regularly rotate API keys
- Use Bitwarden organizations for team sharing
- Audit access logs regularly

### ❌ DON'T:

- Commit `BW_SESSION` to git
- Share session keys
- Use weak master password
- Leave vault unlocked on shared machines
- Store master password in plaintext
- Use same password for vault and services

---

## Comparison with Other Solutions

| Feature | Bitwarden | AWS Secrets Manager | HashiCorp Vault | .env Files |
|---------|-----------|--------------------|-----------------| -----------|
| **Cost** | Free/Paid | Pay per secret | Free/Enterprise | Free |
| **Ease of Setup** | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐ Medium | ⭐⭐ Complex | ⭐⭐⭐⭐⭐ Easy |
| **Cross-Platform** | ✅ Yes | ⚠️ AWS only | ✅ Yes | ✅ Yes |
| **Team Sharing** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ Manual |
| **Encryption** | ✅ AES-256 | ✅ AWS KMS | ✅ AES-256 | ❌ Plaintext |
| **Audit Logs** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Local Dev** | ✅ Works offline | ❌ Needs AWS | ⚠️ Self-host | ✅ Works offline |
| **Mobile Access** | ✅ Apps | ⚠️ Console | ⚠️ Limited | ❌ No |
| **Git Safety** | ✅ Never in repo | ✅ Never in repo | ✅ Never in repo | ⚠️ Easy to commit |

**Bitwarden is best for:**
- Individual developers
- Small-medium teams
- Multi-cloud/hybrid environments
- Need mobile access
- Want simple setup

**AWS Secrets Manager is best for:**
- AWS-native applications
- Large enterprises on AWS
- Need AWS IAM integration

**HashiCorp Vault is best for:**
- Large enterprises
- Complex security requirements
- Dynamic secret generation
- Self-hosted infrastructure

---

## Example: Complete Production Setup

```python
"""
Production agent with Bitwarden-managed secrets.
"""

from agentkit import (
    BitwardenSecrets,
    LiteLLMModel,
    create_agent,
    tool,
    ConsoleTracer,
    BudgetMiddleware,
)

# Initialize secrets manager
secrets = BitwardenSecrets()

# Get API keys from Bitwarden
api_key = secrets.get("OPENAI_API_KEY")

# Optional: Get database credentials
db_creds = secrets.get_item("Database Credentials")

@tool
def query_database(sql: str) -> list:
    """Execute SQL query."""
    # Use db_creds from Bitwarden
    conn = connect(
        host=db_creds["host"],
        user=db_creds["username"],
        password=db_creds["password"],
    )
    # ...
    return results

# Create model with Bitwarden-managed API key
model = LiteLLMModel(
    model="gpt-4o-mini",
    api_key=api_key,
)

# Create production agent
agent = create_agent(
    model=model,
    tools=[query_database],
    system_prompt="You are a data analyst.",
    middleware=[
        ConsoleTracer(),
        BudgetMiddleware(max_tokens=100000),
    ],
)

# Run
result = agent.invoke({"messages": [...]})
```

---

## Resources

- [Bitwarden CLI Documentation](https://bitwarden.com/help/cli/)
- [Bitwarden Security Whitepaper](https://bitwarden.com/help/security-whitepaper/)
- [Example 09: Bitwarden Secrets Demo](../examples/09_bitwarden_secrets.py)
- [Agentkit Bitwarden Module](../agentkit/bitwarden.py)

---

## Summary

**Bitwarden integration brings you:**
- ✅ Encrypted credential storage (AES-256)
- ✅ Cross-device synchronization
- ✅ Team sharing capabilities
- ✅ Full audit trail
- ✅ No `.env` files to leak
- ✅ Centralized secret management
- ✅ Works with all agentkit providers

**Get started:**
```bash
npm install -g @bitwarden/cli
bw login
bw unlock
export BW_SESSION="..."
python examples/09_bitwarden_secrets.py
```

**Security upgrade complete!** 🔒
