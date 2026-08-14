#!/bin/bash
# migrate_all_keys_to_bitwarden.sh
# Migrate all API keys from various .env files to Bitwarden vault
# Organized by provider/service for easy management

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Migrate ALL API Keys to Bitwarden (Organized)           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if BW_SESSION is set
if [ -z "$BW_SESSION" ]; then
    echo "❌ BW_SESSION not set. Please unlock Bitwarden:"
    echo "  bw unlock"
    echo "  export BW_SESSION='your-session-key'"
    exit 1
fi

echo "✅ Bitwarden session active"
echo ""

# Function to create or update a Bitwarden item
create_item() {
    local name="$1"
    local password="$2"
    local notes="$3"

    # Check if item already exists
    existing=$(bw list items --search "$name" 2>/dev/null | jq -r ".[0].id // empty")

    if [ -n "$existing" ]; then
        echo "  ⚠️  '$name' already exists (skipping)"
        return
    fi

    # Create new item
    echo "  ➕ Creating '$name'..."

    # Create item JSON
    cat > /tmp/bw_item_$$.json <<EOF
{
  "organizationId": null,
  "folderId": null,
  "type": 1,
  "name": "$name",
  "notes": "$notes",
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
    bw create item "$encoded" > /dev/null 2>&1 && echo "     ✅ Created" || echo "     ❌ Failed"

    rm /tmp/bw_item_$$.json
}

# Function to read env var from file
get_env_var() {
    local file="$1"
    local var="$2"

    if [ -f "$file" ]; then
        grep "^${var}=" "$file" 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'"
    fi
}

echo "🔐 Migrating API keys by category..."
echo ""

# ============================================================================
# LLM PROVIDERS
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 LLM Provider API Keys"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# OpenAI
OPENAI_KEY=$(get_env_var "/home/gvincent/litellm/.env" "OPENAI_API_KEY")
if [ -n "$OPENAI_KEY" ]; then
    create_item "OPENAI_API_KEY" "$OPENAI_KEY" "OpenAI API key - Used across multiple projects (litellm, langchain-academy, deep-agents, etc.)"
fi

# Anthropic (already migrated, but verify)
ANTHROPIC_KEY=$(get_env_var "/home/gvincent/.env" "ANTHROPIC_API_KEY")
if [ -n "$ANTHROPIC_KEY" ]; then
    create_item "ANTHROPIC_API_KEY" "$ANTHROPIC_KEY" "Anthropic Claude API key - Primary LLM provider"
fi

# Google Gemini
GEMINI_KEY=$(get_env_var "/home/gvincent/litellm/.env" "GEMINI_API_KEY")
if [ -n "$GEMINI_KEY" ]; then
    create_item "GEMINI_API_KEY" "$GEMINI_KEY" "Google Gemini API key - Used in gjh-dmarc-agent, gjh-partner-engine"
fi

# Groq
GROQ_KEY=$(get_env_var "/home/gvincent/litellm/.env" "GROQ_API_KEY")
if [ -n "$GROQ_KEY" ]; then
    create_item "GROQ_API_KEY" "$GROQ_KEY" "Groq API key - Fast inference provider"
fi

# DeepSeek
DEEPSEEK_KEY=$(get_env_var "/home/gvincent/litellm/.env" "DEEPSEEK_API_KEY")
if [ -n "$DEEPSEEK_KEY" ]; then
    create_item "DEEPSEEK_API_KEY" "$DEEPSEEK_KEY" "DeepSeek API key"
fi

echo ""

# ============================================================================
# AI SERVICES
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 AI Services & Tools"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Tavily (Web Search)
TAVILY_KEY=$(get_env_var "/home/gvincent/litellm/.env" "TAVILY_API_KEY")
if [ -n "$TAVILY_KEY" ]; then
    create_item "TAVILY_API_KEY" "$TAVILY_KEY" "Tavily API - Web search for AI agents"
fi

# LangSmith (Tracing)
LANGSMITH_KEY=$(get_env_var "/home/gvincent/langchain-academy/.env" "LANGSMITH_API_KEY")
if [ -n "$LANGSMITH_KEY" ]; then
    create_item "LANGSMITH_API_KEY" "$LANGSMITH_KEY" "LangSmith - LLM tracing and observability"
fi

# OpenCode
OPENCODE_KEY=$(get_env_var "/home/gvincent/litellm/.env" "OPENCODE_API_KEY")
if [ -n "$OPENCODE_KEY" ]; then
    create_item "OPENCODE_API_KEY" "$OPENCODE_KEY" "OpenCode API key"
fi

# AbuseIPDB
ABUSEIPDB_KEY=$(get_env_var "/home/gvincent/gjh-dmarc-agent/.env" "ABUSEIPDB_API_KEY")
if [ -n "$ABUSEIPDB_KEY" ]; then
    create_item "ABUSEIPDB_API_KEY" "$ABUSEIPDB_KEY" "AbuseIPDB - IP reputation checking for DMARC agent"
fi

echo ""

# ============================================================================
# LINKEDIN CREDENTIALS
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💼 LinkedIn API Credentials"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LINKEDIN_CLIENT_ID=$(get_env_var "/home/gvincent/linkedin-autoposter/.env" "LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET=$(get_env_var "/home/gvincent/linkedin-autoposter/.env" "LINKEDIN_CLIENT_SECRET")
LINKEDIN_ACCESS_TOKEN=$(get_env_var "/home/gvincent/linkedin-autoposter/.env" "LINKEDIN_ACCESS_TOKEN")
LINKEDIN_REFRESH_TOKEN=$(get_env_var "/home/gvincent/linkedin-autoposter/.env" "LINKEDIN_REFRESH_TOKEN")
LINKEDIN_PERSON_URN=$(get_env_var "/home/gvincent/linkedin-autoposter/.env" "LINKEDIN_PERSON_URN")

if [ -n "$LINKEDIN_CLIENT_ID" ]; then
    # Create structured LinkedIn credentials item
    cat > /tmp/linkedin_creds.json <<EOF
{
  "organizationId": null,
  "folderId": null,
  "type": 1,
  "name": "LinkedIn API Credentials",
  "notes": "LinkedIn API credentials for autoposter and security-sentinel projects",
  "favorite": false,
  "login": {
    "username": "$LINKEDIN_PERSON_URN",
    "password": "$LINKEDIN_ACCESS_TOKEN"
  },
  "fields": [
    {"name": "client_id", "value": "$LINKEDIN_CLIENT_ID", "type": 0},
    {"name": "client_secret", "value": "$LINKEDIN_CLIENT_SECRET", "type": 1},
    {"name": "refresh_token", "value": "$LINKEDIN_REFRESH_TOKEN", "type": 1},
    {"name": "person_urn", "value": "$LINKEDIN_PERSON_URN", "type": 0}
  ]
}
EOF

    existing=$(bw list items --search "LinkedIn API Credentials" 2>/dev/null | jq -r ".[0].id // empty")
    if [ -z "$existing" ]; then
        echo "  ➕ Creating 'LinkedIn API Credentials'..."
        encoded=$(bw encode < /tmp/linkedin_creds.json)
        bw create item "$encoded" > /dev/null 2>&1 && echo "     ✅ Created" || echo "     ❌ Failed"
    else
        echo "  ⚠️  'LinkedIn API Credentials' already exists (skipping)"
    fi
    rm /tmp/linkedin_creds.json
fi

echo ""

# ============================================================================
# EMAIL SERVICES
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📧 Email Service Credentials"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Resend
RESEND_KEY=$(get_env_var "/home/gvincent/linkedin-autoposter/.env" "RESEND_API_KEY")
if [ -n "$RESEND_KEY" ]; then
    create_item "RESEND_API_KEY" "$RESEND_KEY" "Resend - Email delivery service"
fi

# SMTP Credentials
SMTP_HOST=$(get_env_var "/home/gvincent/gjh-dmarc-agent/.env" "SMTP_HOST")
SMTP_PORT=$(get_env_var "/home/gvincent/gjh-dmarc-agent/.env" "SMTP_PORT")
SMTP_USER=$(get_env_var "/home/gvincent/gjh-dmarc-agent/.env" "SMTP_USER")
SMTP_PASSWORD=$(get_env_var "/home/gvincent/gjh-dmarc-agent/.env" "SMTP_PASSWORD")

if [ -n "$SMTP_USER" ] && [ -n "$SMTP_PASSWORD" ]; then
    cat > /tmp/smtp_creds.json <<EOF
{
  "organizationId": null,
  "folderId": null,
  "type": 1,
  "name": "SMTP Credentials",
  "notes": "SMTP server credentials for email notifications",
  "favorite": false,
  "login": {
    "username": "$SMTP_USER",
    "password": "$SMTP_PASSWORD",
    "uris": [{"uri": "$SMTP_HOST:$SMTP_PORT"}]
  },
  "fields": [
    {"name": "host", "value": "$SMTP_HOST", "type": 0},
    {"name": "port", "value": "$SMTP_PORT", "type": 0}
  ]
}
EOF

    existing=$(bw list items --search "SMTP Credentials" 2>/dev/null | jq -r ".[0].id // empty")
    if [ -z "$existing" ]; then
        echo "  ➕ Creating 'SMTP Credentials'..."
        encoded=$(bw encode < /tmp/smtp_creds.json)
        bw create item "$encoded" > /dev/null 2>&1 && echo "     ✅ Created" || echo "     ❌ Failed"
    else
        echo "  ⚠️  'SMTP Credentials' already exists (skipping)"
    fi
    rm /tmp/smtp_creds.json
fi

echo ""

# ============================================================================
# OTHER SERVICES
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Other Services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Slack Webhook
SLACK_WEBHOOK=$(get_env_var "/home/gvincent/gjh-dmarc-agent/.env" "SLACK_WEBHOOK_URL")
if [ -n "$SLACK_WEBHOOK" ]; then
    create_item "SLACK_WEBHOOK_URL" "$SLACK_WEBHOOK" "Slack webhook for notifications (DMARC agent)"
fi

# Database URLs
DATABASE_URL_DMARC=$(get_env_var "/home/gvincent/gjh-dmarc-agent/.env" "DATABASE_URL")
if [ -n "$DATABASE_URL_DMARC" ]; then
    create_item "DATABASE_URL_DMARC_AGENT" "$DATABASE_URL_DMARC" "PostgreSQL database URL for gjh-dmarc-agent"
fi

DATABASE_URL_PARTNER=$(get_env_var "/home/gvincent/gjh-partner-engine/.env" "DATABASE_URL")
if [ -n "$DATABASE_URL_PARTNER" ]; then
    create_item "DATABASE_URL_PARTNER_ENGINE" "$DATABASE_URL_PARTNER" "PostgreSQL database URL for gjh-partner-engine"
fi

echo ""
echo "🔄 Syncing vault to Bitwarden servers..."
bw sync > /dev/null 2>&1 || echo "  ⚠️  Sync may have failed (offline?)"

echo ""
echo "✅ Migration complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Items created in Bitwarden:"
bw list items | jq -r '.[] | "  ✓ \(.name)"'
echo ""
echo "Total items: $(bw list items | jq length)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. View in Bitwarden:"
echo "   https://vault.bitwarden.com"
echo ""
echo "2. Test retrieval:"
echo "   python -c 'from agentkit import get_secret; print(get_secret(\"OPENAI_API_KEY\")[:20])'"
echo ""
echo "3. Update your code to use Bitwarden:"
echo "   from agentkit import get_secret"
echo "   api_key = get_secret(\"OPENAI_API_KEY\")"
echo ""
echo "4. (Optional) Backup and archive .env files:"
echo "   # After testing thoroughly:"
echo "   mkdir -p ~/.env_backups"
echo "   cp /home/gvincent/*/.env ~/.env_backups/"
echo ""
echo "🔒 All your API keys are now securely stored in Bitwarden!"
