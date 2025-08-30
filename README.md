# Seed-OSS: LLM Tool Calling System Demo

A professional implementation of LLM tool calling with configurable parameters, Jinja2 templating, and modular architecture for building AI assistant applications.

## Overview

Seed-OSS provides a complete framework for integrating Large Language Models (LLMs) with external tools, featuring:
- Configurable LLM interactions via JSON configuration
- Jinja2 template system for prompt engineering
- Tool calling capabilities with result processing
- Environment variable support
- Human-readable logging and debugging

## Project Structure

```
Seed-OSS/
├── config.example.json      # Template configuration file
├── config.json             # Runtime configuration (example values)
├── seed_oss_chat_template.jinja  # Jinja2 prompt template
├── tool_call_demo.py       # Complete implementation with demo workflow
└── README.md               # This documentation
```

## Quick Start

### 1. Configuration

Copy the example configuration to create your own:
```bash
cp config.example.json config.json
```

Edit `config.json` to set your LLM API parameters:
- `api_key`: Your LLM service API key
- `base_url`: LLM API endpoint (e.g., `http://localhost:8080/v1`)
- `model_name`: LLM model name (default: "Seed-OSS")
- `template_path`: Path to Jinja2 template (default: "seed_oss_chat_template.jinja")

### 2. Environment Variables

Alternatively, configure using environment variables:
```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="http://localhost:8080/v1"
export LLM_MODEL_NAME="Seed-OSS"
export TEMPLATE_PATH="seed_oss_chat_template.jinja"
export MAX_TOKENS=65536
export SHOW_THINKING_TOKENS=false
```

### 3. Run the Demo

Execute the tool calling demo:
```bash
python tool_call_demo.py
```

The demo will:
1. Load configuration from file or environment variables
2. Render a prompt using the Jinja2 template
3. Send the prompt to the LLM
4. Process any tool calls (e.g., `get_current_local_time`)
5. Return a final response incorporating tool results

## Configuration Options

### Core Settings (`config.json`)

| Parameter               | Type    | Default Value                          | Description                                                                 |
|-------------------------|---------|----------------------------------------|-----------------------------------------------------------------------------|
| `api_key`               | string  | "your-api-key-here"                    | LLM API authentication key                                                  |
| `base_url`              | string  | "http://localhost:8080/v1"             | LLM API base URL                                                            |
| `model_name`            | string  | "Seed-OSS"                             | LLM model identifier                                                        |
| `template_path`         | string  | "seed_oss_chat_template.jinja"         | Path to Jinja2 prompt template                                              |
| `use_json_tooldef`      | boolean | false                                  | Use JSON schema for tool definitions instead of Python function syntax      |
| `add_generation_prompt` | boolean | true                                   | Add assistant role prompt to start generation                               |
| `show_thinking_tokens`  | boolean | false                                  | Show thinking tokens in LLM responses                                       |
| `max_tokens`            | integer | 65536                                  | Maximum tokens for LLM completion                                           |
| `thinking_budget`       | integer | -1 (no limit)                          | Token budget for use in reflection intervals                                |

### Tool Configuration

The system includes a default tool:
- `get_current_local_time`: Retrieves current local date/time in `YYYY-MM-DD HH:MM:SS ZZZ±HHMM` format

Add custom tools by extending the `tools` array in your configuration or modifying the demo script.

## Template System

Seed-OSS uses Jinja2 templates for prompt engineering with support for:
- Special token variables (bos_token, eos_token, toolcall tokens)
- Dynamic thinking budget calculation
- System message preprocessing
- Tool definition rendering (JSON or Python syntax)
- Multi-turn conversation history

## Implementation Details

### Key Components

1. **`ToolCallConfig` Class**: Manages configuration with hierarchical loading (defaults → env vars → config file)
2. **`LLMToolClient` Class**: Handles LLM interactions, template rendering, and response processing
3. **`ToolExecutor` Class**: Executes tool calls and parses LLM output
4. **`Logger` Class**: Provides structured logging for workflow debugging

### Response Processing

- **Thinking Tokens**: Can be hidden (`show_thinking_tokens: false`) or shown in responses
- **Tool Calls**: Extracted using regex patterns matching `<function=...>` delimiters
- **Parameter Parsing**: Automatic conversion of string values to basic types (int/float/bool)

## Customization

### Adding New Tools

1. Add tool definition to `config.json` or `load_config()` method:
   ```json
   {
     "type": "function",
     "function": {
       "name": "your_tool_name",
       "description": "Tool description",
       "parameters": {
         "type": "object",
         "properties": {
           "param1": {"type": "string"},
           "param2": {"type": "integer"}
         },
         "required": ["param1"]
       }
     }
   }
   ```

2. Implement the tool function in `ToolExecutor` class:
   ```python
   @staticmethod
   def your_tool_name(param1: str, param2: int = None) -> str:
       # Your implementation here
   ```

### Modifying Templates

Customize the Jinja2 template (`seed_oss_chat_template.jinja`) to adjust:
- Special token definitions
- System message content
- Tool definition formatting
- Conversation history presentation
- Thinking budget behavior

## Error Handling

The system includes comprehensive error handling for:
- Configuration loading failures
- Template rendering errors
- LLM API call failures
- Tool execution errors
- Response processing issues

All errors are logged with context information for debugging.

## License

This project is provided as-is for educational and demonstration purposes. Modify and extend it according to your needs.

## Contributing

Feel free to submit issues or pull requests to improve the implementation. Key areas for enhancement:
- Additional tool examples
- Better parameter validation
- More template customization options
- Extended error recovery mechanisms
- Performance optimizations
