#!/bin/bash
# migrate_to_bitwarden.sh
# Securely migrate API keys from .env files to Bitwarden vault

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Migrate API Keys from .env to Bitwarden Vault           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if BW_SESSION is set
if [ -z "$BW_SESSION" ]; then
    echo "❌ BW_SESSION not set. Please unlock Bitwarden:"
    echo ""
    echo "  bw unlock"
    echo ""
    echo "Then run the export command it gives you, and run this script again."
    exit 1
fi

echo "✅ Bitwarden session active"
echo ""

# Found API keys in /home/gvincent/.env:
# - ANTHROPIC_API_KEY

echo "📋 API Keys found in your .env files:"
echo "  1. ANTHROPIC_API_KEY (in ~/.env)"
echo ""

# Function to create or update a Bitwarden item
create_or_update_item() {
    local name="$1"
    local password="$2"

    # Check if item already exists
    existing=$(bw list items --search "$name" 2>/dev/null | jq -r ".[0].id // empty")

    if [ -n "$existing" ]; then
        echo "  ⚠️  Item '$name' already exists in vault (skipping)"
        return
    fi

    # Create new item
    echo "  ➕ Creating '$name' in Bitwarden..."

    # Create item JSON
    cat > /tmp/bw_item_$$.json <<EOF
{
  "organizationId": null,
  "folderId": null,
  "type": 1,
  "name": "$name",
  "notes": "API key migrated from .env file",
  "favorite": false,
  "login": {
    "username": "",
    "password": "$password",
    "totp": null
  }
}
EOF

    # Encode and create
    encoded=$(bw encode < /tmp/bw_item_$$.json)
    bw create item "$encoded" > /dev/null

    rm /tmp/bw_item_$$.json
    echo "  ✅ Created '$name'"
}

# Migrate ANTHROPIC_API_KEY
echo "🔐 Migrating API keys to Bitwarden..."
echo ""

if [ -f /home/gvincent/.env ]; then
    # Read ANTHROPIC_API_KEY from .env
    ANTHROPIC_KEY=$(grep "^ANTHROPIC_API_KEY=" /home/gvincent/.env | cut -d'"' -f2)

    if [ -n "$ANTHROPIC_KEY" ]; then
        create_or_update_item "ANTHROPIC_API_KEY" "$ANTHROPIC_KEY"
    fi
fi

echo ""
echo "🔄 Syncing vault to Bitwarden servers..."
bw sync > /dev/null 2>&1 || echo "  ⚠️  Sync may have failed (offline?)"

echo ""
echo "✅ Migration complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Verify your keys in Bitwarden:"
echo "   bw list items | jq '.[] | {name: .name, id: .id}'"
echo ""
echo "2. Test retrieval:"
echo "   python -c 'from agentkit import get_secret; print(get_secret(\"ANTHROPIC_API_KEY\")[:20])'"
echo ""
echo "3. Update your code to use Bitwarden:"
echo "   # OLD:"
echo "   from dotenv import load_dotenv"
echo "   load_dotenv()"
echo "   api_key = os.getenv(\"ANTHROPIC_API_KEY\")"
echo ""
echo "   # NEW:"
echo "   from agentkit import get_secret"
echo "   api_key = get_secret(\"ANTHROPIC_API_KEY\")"
echo ""
echo "4. (Optional) Remove .env files after confirming everything works:"
echo "   # Keep as backup first!"
echo "   mv ~/.env ~/.env.backup"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔒 Your API keys are now securely stored in Bitwarden!"
