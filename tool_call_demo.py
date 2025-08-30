"""
LLM Tool Calling Demo - Professional Implementation
This script demonstrates LLM tool calling using a Jinja template system
with improved maintainability through configuration management and modular design.
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from openai import OpenAI
import jinja2
from dotenv import load_dotenv  # For environment variable support


# ------------------------------
# 1. Configuration System
# ------------------------------

@dataclass
class ToolCallConfig:
    """Configuration class for tool calling parameters."""
    
    # API Configuration
    api_key: str = "your-api-key-here"
    base_url: str = "http://localhost:8080/v1"
    model_name: str = "Seed-OSS"
    
    # Template Configuration
    template_path: str = "seed_oss_chat_template.jinja"
    use_json_tooldef: bool = False
    add_generation_prompt: bool = True
    show_thinking_tokens: bool = False  # Whether to SHOW thinking tokens in LLM responses (not just prompts)
    
    # Token Configuration
    max_tokens: int = 65536
    thinking_budget: int = -1  # -1 means no limit
    
    # Tool Configuration
    tools: List[Dict[str, Any]] = field(default_factory=list)


def load_config(config_path: Optional[str] = None) -> ToolCallConfig:
    """
    Load configuration from file or environment variables.
    Priority: File config > Environment variables > Defaults
    """
    # Load environment variables first
    load_dotenv()
    
    # Start with default configuration
    config = ToolCallConfig()
    
# Override with environment variables if available
    if os.getenv("LLM_API_KEY"):
        config.api_key = os.getenv("LLM_API_KEY")
    if os.getenv("LLM_BASE_URL"):
        config.base_url = os.getenv("LLM_BASE_URL")
    if os.getenv("LLM_MODEL_NAME"):
        config.model_name = os.getenv("LLM_MODEL_NAME")
    if os.getenv("TEMPLATE_PATH"):
        config.template_path = os.getenv("TEMPLATE_PATH")
    if os.getenv("MAX_TOKENS"):
        config.max_tokens = int(os.getenv("MAX_TOKENS"))
    if os.getenv("SHOW_THINKING_TOKENS"):
        config.show_thinking_tokens = os.getenv("SHOW_THINKING_TOKENS").lower() in ["true", "1", "yes"]
    
    # Override with config file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            
            # Update configuration from file (only known fields)
            for field_name, field_value in file_config.items():
                if hasattr(config, field_name):
                    setattr(config, field_name, field_value)
                    
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {str(e)}")
    
    # Add default tools
    config.tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_local_time",
                "description": "Retrieves the current local date and time. Use when the user asks for the current time.",
                "parameters": {
                    "type": "object",
                    "properties": {},  # No parameters needed
                    "required": []
                }
            }
        }
    ]
    
    return config


# ------------------------------
# 2. Core Functionality Classes
# ------------------------------

class LLMToolClient:
    """Client for interacting with LLM and managing tool calls."""
    
    # Thinking tokens from the Jinja template
    THINK_BEGIN_TOKEN = '<seed:think>'
    THINK_END_TOKEN = '</seed:think>'
    
    def __init__(self, config: ToolCallConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.template = self._load_jinja_template()

    def _load_jinja_template(self) -> jinja2.Template:
        """Load and compile the Jinja template from file."""
        try:
            with open(self.config.template_path, "r", encoding="utf-8") as f:
                source = f.read()
            return jinja2.Template(source)
        except FileNotFoundError:
            raise RuntimeError(f"Template file not found: {self.config.template_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load template: {str(e)}")

    def render_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Render a prompt using the Jinja template."""
        try:
            return self.template.render(
                messages=messages,
                tools=self.config.tools,
                use_json_tooldef=self.config.use_json_tooldef,
                thinking_budget=self.config.thinking_budget,
                add_generation_prompt=self.config.add_generation_prompt,
                show_thinking_tokens=self.config.show_thinking_tokens
            )
        except Exception as e:
            raise RuntimeError(f"Failed to render prompt: {str(e)}")


    def _process_llm_response(self, llm_output: str) -> str:
        """
        Process LLM response to handle thinking tokens based on configuration.
        
        If show_thinking_tokens is False (default), removes thinking content wrapped in 
        <seed:think> ... </seed:think> tokens while preserving the rest of the response.
        
        Returns:
            Processed response string with or without thinking content
        """
        if self.config.show_thinking_tokens:
            # Return response as-is when thinking tokens are enabled
            return llm_output
            
        # When thinking tokens are disabled, remove all <seed:think> ... </seed:think> blocks
        try:
            # Pattern to match <seed:think> ... </seed:think> blocks (non-greedy)
            pattern = re.compile(f"{re.escape(self.THINK_BEGIN_TOKEN)}(.*?){re.escape(self.THINK_END_TOKEN)}", 
                              re.DOTALL)
            
            # Remove all thinking blocks while preserving the rest of the text
            processed_output = pattern.sub("", llm_output)
            
            # Clean up any extra whitespace from removed blocks
            processed_output = re.sub(r'\n\s*\n', '\n\n', processed_output).strip()
            
            return processed_output
            
        except Exception as e:
            # Fall back to original output if processing fails
            print(f"Warning: Failed to process thinking tokens: {str(e)}")
            return llm_output

    def call_llm(self, prompt: str) -> str:
        """Send a prompt to the LLM and get the processed response."""
        try:
            response = self.client.completions.create(
                model=self.config.model_name,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
            )
            
            raw_output = response.choices[0].text.strip() if response.choices else ""
            return self._process_llm_response(raw_output)
            
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

# ------------------------------
# 3. Tool Implementations
# ------------------------------

class ToolExecutor:
    """Executes tool functions and manages results."""
    
    @staticmethod
    def get_current_local_time() -> str:
        """Get current local date/time in human-readable format."""
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")  # e.g., "2024-05-20 16:30:00 EDT-0400"


    @staticmethod
    def execute_tool_call(tool_call: Dict[str, Any]) -> str:
        """Execute a single tool call and return the result."""
        func_name = tool_call["function"]["name"]
        func_args = tool_call["function"]["arguments"]
        
        # Execute the requested tool
        if func_name == "get_current_local_time":
            return ToolExecutor.get_current_local_time()
        else:
            raise ValueError(f"Unknown function '{func_name}'")


    @staticmethod
    def parse_tool_calls(llm_output: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM output using the template's <function=...> delimiters."""
        tool_calls = []
        
        # Regex to match function blocks (supports multi-line parameters)
        function_pattern = re.compile(
            r"<function=(?P<name>[^>]+)>(?P<params>.*?)</function>",
            re.DOTALL | re.IGNORECASE
        )
        
        for match in function_pattern.finditer(llm_output):
            func_name = match.group("name").strip()
            params_text = match.group("params").strip()
            
            # Parse parameters (if any) using <parameter=...> tags
            params = {}
            if params_text:
                param_pattern = re.compile(
                    r"<parameter=(?P<name>[^>]+)>(?P<value>.*?)</parameter>",
                    re.DOTALL | re.IGNORECASE
                )
                for param_match in param_pattern.finditer(params_text):
                    param_name = param_match.group("name").strip()
                    param_value = param_match.group("value").strip()
                    
                    # Convert string values to basic types (int/float/bool)
                    try:
                        param_value = json.loads(param_value)
                    except:
                        pass  # Keep as string if JSON parsing fails
                    
                    params[param_name] = param_value
            
            tool_calls.append({
                "function": {
                    "name": func_name,
                    "arguments": params
                }
            })
        
        return tool_calls


# ------------------------------
# 4. Utilities
# ------------------------------

class Logger:
    """Simple logging utility with consistent formatting."""
    
    @staticmethod
    def log_step(step: str, details: str = "", max_length: int = 5000) -> None:
        """Log workflow steps with clear formatting and truncation for long outputs."""
        print(f"┌{'─' * 60}┐")
        print(f"│ {step}")
        if details:
            truncated_details = details[:max_length] + "..." if len(details) > max_length else details
            print(f"│   Details: {truncated_details}")
        print(f"└{'─' * 60}┘")


# ------------------------------
# 5. Main Workflow
# ------------------------------

def run_tool_calling_demo(config_path: Optional[str] = None):
    """Run the complete tool calling demo workflow."""
    Logger.log_step("🚀 STARTING LLM TOOL CALLING DEMO (PROFESSIONAL VERSION)")
    
    try:
        # Load configuration
        config = load_config(config_path)
        Logger.log_step("⚙️ LOADED CONFIGURATION", f"Using template: {config.template_path}")
        
        # Initialize clients
        llm_client = LLMToolClient(config)
        tool_executor = ToolExecutor()
        
        # Initialize conversation
        conversation = []
        user_query = "What's the current local time right now?"
        conversation.append({"role": "user", "content": user_query})
        Logger.log_step("💬 USER INPUT", user_query)

        # Step 1: Render initial prompt
        Logger.log_step("🎨 RENDERING INITIAL PROMPT")
        rendered_prompt = llm_client.render_prompt(conversation)
        Logger.log_step("📝 RENDERED PROMPT PREVIEW", rendered_prompt)

        # Step 2: Send prompt to LLM
        Logger.log_step("📤 SENDING PROMPT TO LLM")
        llm_output = llm_client.call_llm(rendered_prompt)
        Logger.log_step("📥 LLM RESPONSE RECEIVED", llm_output)

        # Step 3: Check for tool calls
        tool_calls = tool_executor.parse_tool_calls(llm_output)
        
        if not tool_calls:
            Logger.log_step("⚠️ NO TOOL CALLS", "LLM responded directly")
            Logger.log_step("📢 DIRECT ANSWER", llm_output if llm_output else "Empty response")
            return

        # Step 4: Execute tool calls
        Logger.log_step("🔧 EXECUTING TOOL CALLS", f"Found {len(tool_calls)} tool(s)")
        tool_results = []

        for idx, tc in enumerate(tool_calls):
            try:
                result = tool_executor.execute_tool_call(tc)
                tool_results.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": f"call_{idx}"  # Simplified ID for tracking
                })
                Logger.log_step(f"✅ TOOL RESULT: {tc['function']['name']}", result)
            except Exception as e:
                error_msg = f"❌ ERROR in tool '{tc['function']['name']}': {str(e)}"
                tool_results.append({
                    "role": "tool",
                    "content": error_msg,
                    "tool_call_id": f"call_{idx}"
                })
                Logger.log_step(f"❌ TOOL ERROR: {tc['function']['name']}", str(e))

        # Step 5: Add results to conversation & re-render prompt
        for result in tool_results:
            conversation.append({"role": "tool", "content": result["content"]})
        
        Logger.log_step("📋 ADDING TOOL RESULTS TO CONVERSATION")

        # Step 6: Get final response from LLM
        Logger.log_step(f"🎨 RENDERING FINAL PROMPT (WITH TOOL RESULTS): {conversation}")
        final_prompt = llm_client.render_prompt(conversation)
        
        Logger.log_step(f"📤 SENDING FINAL PROMPT FOR ANSWER: {final_prompt}")
        final_answer = llm_client.call_llm(final_prompt)
        
        Logger.log_step("🎉 FINAL LLM RESPONSE", final_answer if final_answer else "No answer generated")

    except Exception as e:
        Logger.log_step("❌ CRITICAL ERROR OCCURRED", f"{type(e).__name__}: {str(e)}")
        import traceback
        Logger.log_step("📋 FULL TRACEBACK", traceback.format_exc())


# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    # Specify a config path here, or use environment variables
    # Example: run_tool_calling_demo(config_path="config.example.json")
    run_tool_calling_demo(config_path="config.json")
    
    print("\n" + "="*60)
    print("Demo completed. To customize, create a config.json file or set environment variables.")
    print("Available env vars: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, TEMPLATE_PATH, MAX_TOKENS, SHOW_THINKING_TOKENS")
    print("Config options: show_thinking_tokens (whether to SHOW thinking tokens <seed:think> ... </seed:think> in LLM responses)")
