"""
Example 09 — Using Bitwarden for secure credential management

Store API keys and secrets in Bitwarden vault instead of .env files or
hardcoded strings. More secure, better for teams, easier to rotate.

SETUP:
    1. Install Bitwarden CLI:
       https://bitwarden.com/help/cli/

    2. Login to Bitwarden:
       bw login your-email@example.com

    3. Unlock your vault and get session key:
       bw unlock
       (Copy the export BW_SESSION="..." command it outputs)

    4. Export the session key:
       export BW_SESSION="your-session-key-here"

    5. Create items in Bitwarden for your API keys:
       - Item name: "OPENAI_API_KEY"
         Password field: sk-...

       - Item name: "ANTHROPIC_API_KEY"
         Password field: sk-ant-...

       - Item name: "AWS Credentials"
         Custom fields:
           - access_key_id: AKIA...
           - secret_access_key: ...
           - region: us-east-1

USAGE:
    export BW_SESSION="your-session-key"
    python examples/09_bitwarden_secrets.py
"""

from __future__ import annotations

import os
import sys

from agentkit import (
    AnthropicModel,
    BitwardenSecrets,
    ConsoleTracer,
    FakeModel,
    LiteLLMModel,
    Message,
    OpenAIModel,
    ToolCall,
    create_agent,
    get_secret,
    tool,
)


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return {
        "city": city,
        "temperature": 24,
        "conditions": "sunny",
    }


def demo_basic_usage():
    """Demo 1: Basic secret retrieval."""
    print("="*60)
    print("DEMO 1: Basic Secret Retrieval from Bitwarden")
    print("="*60)

    try:
        secrets = BitwardenSecrets()

        # List available items
        print("\n📋 Items in your Bitwarden vault:")
        items = secrets.list_items()
        for i, item in enumerate(items[:10], 1):  # Show first 10
            print(f"  {i}. {item}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")

        print("\n✅ Bitwarden connection working!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you:")
        print("  1. Have Bitwarden CLI installed: bw --version")
        print("  2. Are logged in: bw login")
        print("  3. Have unlocked vault: bw unlock")
        print("  4. Have exported session: export BW_SESSION='...'")


def demo_api_key_retrieval():
    """Demo 2: Retrieve API keys from Bitwarden."""
    print("\n" + "="*60)
    print("DEMO 2: Retrieve API Keys from Bitwarden")
    print("="*60)

    try:
        secrets = BitwardenSecrets()

        # Try to get OpenAI API key
        print("\n🔑 Retrieving OPENAI_API_KEY from Bitwarden...")
        openai_key = secrets.get("OPENAI_API_KEY", default=None)

        if openai_key:
            # Mask the key for display
            masked = openai_key[:7] + "..." + openai_key[-4:] if len(openai_key) > 11 else "***"
            print(f"✅ Found: {masked}")
        else:
            print("⚠️  Not found - you can create it in Bitwarden:")
            print("   Item name: OPENAI_API_KEY")
            print("   Password field: sk-...")

        # Try to get Anthropic API key
        print("\n🔑 Retrieving ANTHROPIC_API_KEY from Bitwarden...")
        anthropic_key = secrets.get("ANTHROPIC_API_KEY", default=None)

        if anthropic_key:
            masked = anthropic_key[:7] + "..." + anthropic_key[-4:] if len(anthropic_key) > 11 else "***"
            print(f"✅ Found: {masked}")
        else:
            print("⚠️  Not found")

    except Exception as e:
        print(f"❌ Error: {e}")


def demo_structured_credentials():
    """Demo 3: Retrieve structured credentials (multiple fields)."""
    print("\n" + "="*60)
    print("DEMO 3: Structured Credentials (AWS Example)")
    print("="*60)

    try:
        secrets = BitwardenSecrets()

        print("\n📦 Retrieving AWS Credentials...")
        print("   Looking for item: 'AWS Credentials'")

        # Get all fields from AWS Credentials item
        aws = secrets.get_item("AWS Credentials")

        print(f"✅ Found {len(aws)} fields:")
        for field_name, value in aws.items():
            # Mask secrets
            if len(value) > 10:
                masked = value[:4] + "..." + value[-4:]
            else:
                masked = "***"
            print(f"   - {field_name}: {masked}")

    except Exception as e:
        print(f"⚠️  Item not found or error: {e}")
        print("\nTo create AWS credentials in Bitwarden:")
        print("  1. Create new item named 'AWS Credentials'")
        print("  2. Add custom fields:")
        print("     - access_key_id: AKIA...")
        print("     - secret_access_key: ...")
        print("     - region: us-east-1")


def demo_agent_with_bitwarden():
    """Demo 4: Use Bitwarden secrets with an agent."""
    print("\n" + "="*60)
    print("DEMO 4: Agent with Bitwarden-Managed API Keys")
    print("="*60)

    try:
        # Check if we have session
        if not os.getenv("BW_SESSION"):
            print("⚠️  BW_SESSION not set - using FakeModel for demo")
            model = FakeModel(responses=[
                Message.assistant("", tool_calls=[
                    ToolCall(name="get_weather", args={"city": "London"})
                ]),
                Message.assistant("It's 24°C and sunny in London."),
            ])
        else:
            # Try to get API key from Bitwarden
            print("\n🔑 Getting API key from Bitwarden...")
            api_key = get_secret("OPENAI_API_KEY", default=None)

            if api_key:
                print("✅ API key retrieved from Bitwarden")
                model = OpenAIModel(model="gpt-4o-mini", api_key=api_key)
                print("🤖 Using OpenAI model")
            else:
                print("⚠️  OPENAI_API_KEY not in Bitwarden - using FakeModel")
                model = FakeModel(responses=[
                    Message.assistant("", tool_calls=[
                        ToolCall(name="get_weather", args={"city": "London"})
                    ]),
                    Message.assistant("It's 24°C and sunny in London."),
                ])

        # Create agent
        agent = create_agent(
            model=model,
            tools=[get_weather],
            system_prompt="You are a helpful assistant.",
            middleware=[ConsoleTracer()],
        )

        # Run agent
        print("\n🚀 Running agent...")
        result = agent.invoke({
            "messages": [Message.user("What's the weather in London?")]
        })

        print("\n📝 Final answer:", result["messages"][-1].content)

    except Exception as e:
        print(f"❌ Error: {e}")


def demo_all_providers():
    """Demo 5: Show how to use Bitwarden with all model providers."""
    print("\n" + "="*60)
    print("DEMO 5: All Providers with Bitwarden")
    print("="*60)

    print("""
Using Bitwarden with different model providers:

1. OpenAI:
   --------
   secrets = BitwardenSecrets()
   api_key = secrets.get("OPENAI_API_KEY")
   model = OpenAIModel(api_key=api_key)

2. Anthropic:
   ----------
   api_key = get_secret("ANTHROPIC_API_KEY")
   model = AnthropicModel(api_key=api_key)

3. LiteLLM (any provider):
   -----------------------
   openai_key = secrets.get("OPENAI_API_KEY")
   model = LiteLLMModel("gpt-4o-mini", api_key=openai_key)

   anthropic_key = secrets.get("ANTHROPIC_API_KEY")
   model = LiteLLMModel("claude-sonnet-4-5", api_key=anthropic_key)

4. AWS Bedrock via LiteLLM:
   ------------------------
   aws_creds = secrets.get_item("AWS Credentials")
   os.environ["AWS_ACCESS_KEY_ID"] = aws_creds["access_key_id"]
   os.environ["AWS_SECRET_ACCESS_KEY"] = aws_creds["secret_access_key"]
   os.environ["AWS_REGION_NAME"] = aws_creds["region"]
   model = LiteLLMModel("bedrock/anthropic.claude-v2")

5. Azure OpenAI:
   -------------
   azure = secrets.get_item("Azure OpenAI")
   model = LiteLLMModel(
       "azure/gpt-4",
       api_key=azure["api_key"],
       api_base=azure["endpoint"],
   )
""")


def show_bitwarden_setup_guide():
    """Show setup instructions."""
    print("\n" + "="*60)
    print("BITWARDEN SETUP GUIDE")
    print("="*60)
    print("""
Step 1: Install Bitwarden CLI
------------------------------
Linux/Mac:
  npm install -g @bitwarden/cli
  # or
  brew install bitwarden-cli

Windows:
  choco install bitwarden-cli

Verify:
  bw --version

Step 2: Login
-------------
  bw login your-email@example.com

Step 3: Unlock Vault
--------------------
  bw unlock

Copy the export command it outputs, e.g.:
  export BW_SESSION="AbCdEf123456..."

Run that export command in your shell.

Step 4: Create Items for Your Secrets
--------------------------------------
Option A: Via Web/Desktop App (easier)
  1. Open Bitwarden app or vault.bitwarden.com
  2. Create new items for each API key:

     Name: OPENAI_API_KEY
     Type: Login
     Password: sk-your-openai-key

     Name: ANTHROPIC_API_KEY
     Type: Login
     Password: sk-ant-your-anthropic-key

Option B: Via CLI
  bw create item --name "OPENAI_API_KEY" \\
     --username "" \\
     --password "sk-your-key-here"

Step 5: Use in Your Code
-------------------------
  from agentkit import BitwardenSecrets, OpenAIModel

  secrets = BitwardenSecrets()
  api_key = secrets.get("OPENAI_API_KEY")
  model = OpenAIModel(api_key=api_key)

Step 6 (Optional): Keep Session Active
---------------------------------------
Add to your shell profile (~/.bashrc, ~/.zshrc):

  # Bitwarden session
  export BW_SESSION="your-session-key"

Or use `bw unlock` when starting work and paste the export.

SECURITY NOTES:
--------------
✓ Never commit BW_SESSION to git
✓ Session keys expire - re-unlock if needed
✓ Use `bw lock` when done working
✓ Bitwarden CLI stores secrets encrypted at rest
✓ Much safer than .env files in your project
""")


if __name__ == "__main__":
    print("╔" + "="*58 + "╗")
    print("║        AGENTKIT + BITWARDEN: SECURE SECRETS MANAGEMENT     ║")
    print("╚" + "="*58 + "╝")

    # Check if BW_SESSION is set
    if not os.getenv("BW_SESSION"):
        print("\n⚠️  BW_SESSION environment variable not set")
        print("   Some demos will run in limited mode.\n")
        show_bitwarden_setup_guide()
    else:
        print("\n✅ BW_SESSION detected - ready to use Bitwarden!\n")

    # Run demos
    demo_basic_usage()
    demo_api_key_retrieval()
    demo_structured_credentials()
    demo_agent_with_bitwarden()
    demo_all_providers()

    print("\n" + "="*60)
    print("KEY BENEFITS")
    print("="*60)
    print("""
✓ Encrypted vault storage (AES-256)
✓ Cross-device sync
✓ Team sharing (for organizations)
✓ Audit logs
✓ Automatic secret rotation
✓ No .env files to accidentally commit
✓ Centralized credential management
✓ Works with all agentkit model providers
""")

    if os.getenv("BW_SESSION"):
        print("\n✅ All demos completed successfully!")
    else:
        print("\n💡 Set up Bitwarden to unlock full functionality:")
        print("   1. Install Bitwarden CLI: npm install -g @bitwarden/cli")
        print("   2. Run: bw login")
        print("   3. Run: bw unlock")
        print("   4. Export the BW_SESSION it gives you")
        print("   5. Run this example again!")
