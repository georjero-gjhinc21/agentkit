# API Key Migration to Bitwarden

## Current Status

❌ **Your API keys are NOT yet in Bitwarden** - they're still in `.env` files.

I created the Bitwarden integration code, but you need to **manually migrate** your actual API keys.

## What I Found

**API Keys in your .env files:**

| Location | API Key | Status |
|----------|---------|--------|
| `/home/gvincent/.env` | `ANTHROPIC_API_KEY` | ⚠️ In plaintext .env file |

**Bitwarden Status:**
- ✅ Bitwarden CLI installed
- ✅ Logged in as: george@gjh-inc.com
- ❌ Vault is **locked** (needs unlock)

---

## Migration Steps

### Option 1: Automated Migration (Recommended)

I created a migration script that will move your keys to Bitwarden:

```bash
# Step 1: Unlock your Bitwarden vault
bw unlock

# Step 2: Copy and run the export command it gives you
export BW_SESSION="paste-the-session-key-here"

# Step 3: Run the migration script
cd ~/agentkit
./migrate_to_bitwarden.sh
```

The script will:
- ✅ Create Bitwarden items for each API key
- ✅ Preserve the key values
- ✅ Add helpful notes
- ✅ Sync to Bitwarden servers

### Option 2: Manual Migration (Web/Desktop App)

1. **Open Bitwarden:**
   - Go to https://vault.bitwarden.com
   - Or open Bitwarden desktop app

2. **Create New Item:**
   - Click "New Item"
   - Fill in:
     - **Name:** `ANTHROPIC_API_KEY`
     - **Type:** Login
     - **Username:** (leave empty)
     - **Password:** (your full API key from ~/.env file)
   - Click "Save"

3. **Verify:**
   ```bash
   export BW_SESSION="your-session-key"
   python -c "from agentkit import get_secret; print(get_secret('ANTHROPIC_API_KEY')[:20])"
   ```

### Option 3: CLI Manual Migration

```bash
# Unlock vault
bw unlock
export BW_SESSION="your-session-key"

# Create item for ANTHROPIC_API_KEY
# Get your actual key from ~/.env first
ANTHROPIC_KEY=$(grep "^ANTHROPIC_API_KEY=" ~/.env | cut -d'"' -f2)

bw create item \
  --name "ANTHROPIC_API_KEY" \
  --username "" \
  --password "$ANTHROPIC_KEY"

# Sync to cloud
bw sync
```

---

## After Migration

### 1. Test Retrieval

```bash
export BW_SESSION="your-session-key"

# Test in Python
python -c "
from agentkit import get_secret
key = get_secret('ANTHROPIC_API_KEY')
print(f'Retrieved: {key[:10]}...{key[-4:]}')
"
```

### 2. Update Your Code

**Old way (with .env):**
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
```

**New way (with Bitwarden):**
```python
from agentkit import get_secret

api_key = get_secret("ANTHROPIC_API_KEY")
```

### 3. Update agentkit Examples

The agentkit code can now use Bitwarden automatically:

```python
from agentkit import BitwardenSecrets, AnthropicModel, create_agent

secrets = BitwardenSecrets()

model = AnthropicModel(
    model="claude-sonnet-4-5",
    api_key=secrets.get("ANTHROPIC_API_KEY"),
)

agent = create_agent(model=model, tools=[...])
```

### 4. Backup and Remove .env Files (Optional)

Once everything works with Bitwarden:

```bash
# Make backups first!
cp ~/.env ~/.env.backup
cp ~/langchain-academy/.env ~/langchain-academy/.env.backup
# ... etc for other .env files

# Add .env to .gitignore everywhere (if not already)
echo ".env" >> .gitignore
echo ".env.backup" >> .gitignore

# After confirming Bitwarden works, you can remove .env files
# (Keep backups for a while!)
```

---

## Security Benefits After Migration

✅ **Encrypted** - Keys stored with AES-256 encryption  
✅ **Synced** - Access from any device securely  
✅ **Auditable** - Know when/where keys were accessed  
✅ **No git leaks** - Never risk committing .env  
✅ **Team sharing** - Securely share with teammates  
✅ **Auto-rotate** - Update once, applies everywhere  

---

## Troubleshooting

### "Bitwarden session key not found"

```bash
bw unlock
export BW_SESSION="the-session-key-it-gives-you"
```

### "Item already exists"

Your key is already in Bitwarden! Just verify you can retrieve it:

```bash
python -c "from agentkit import get_secret; print(get_secret('ANTHROPIC_API_KEY')[:20])"
```

### "CLI not found"

```bash
npm install -g @bitwarden/cli
# or
brew install bitwarden-cli
```

---

## Next Steps

1. ✅ **Unlock Bitwarden:** `bw unlock`
2. ✅ **Export session:** `export BW_SESSION="..."`
3. ✅ **Run migration:** `./migrate_to_bitwarden.sh`
4. ✅ **Test retrieval:** Verify keys work
5. ✅ **Update code:** Use `get_secret()` instead of `os.getenv()`
6. ✅ **Backup .env files:** Keep backups temporarily
7. ✅ **Remove .env files:** After confirming everything works

---

## Summary

**Status:** Integration code ready ✅ | Keys migration pending ⏳

**Your API keys are currently:**
- ❌ In plaintext .env files
- ❌ At risk of git leaks
- ❌ Not encrypted
- ❌ Hard to share with team

**After migration, your keys will be:**
- ✅ Encrypted in Bitwarden vault
- ✅ Protected from git leaks
- ✅ Synced across devices
- ✅ Easy to share securely
- ✅ Fully auditable

**To migrate now:**
```bash
bw unlock
export BW_SESSION="your-session-key"
cd ~/agentkit
./migrate_to_bitwarden.sh
```
