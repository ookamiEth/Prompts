# Cursor Notebook MCP Server: Complete Setup and Usage Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Available Tools](#available-tools)
6. [Best Practices & Workflows](#best-practices--workflows)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

## Overview

The Cursor Notebook MCP Server is a powerful tool that enables AI agents within Cursor IDE to interact with Jupyter Notebook (.ipynb) files. This server overcomes Cursor's limitation of not directly supporting notebook editing, providing over 20 specialized tools for notebook manipulation.

### Key Benefits
- **Direct notebook manipulation**: Create, read, edit, and manage Jupyter notebooks
- **Cost-efficient operations**: Optimized to minimize API calls
- **Remote editing**: Support for SFTP/SSH connections
- **Multiple transport modes**: HTTP, SSE, and stdio support
- **Comprehensive toolset**: Over 20 specialized notebook operations

## Prerequisites

Before setting up the Cursor Notebook MCP Server, ensure you have:

1. **Python 3.10 or higher with conda**
   ```bash
   conda --version  # Should show conda installed
   python --version  # Should show Python 3.10+
   ```

2. **Cursor IDE installed**
   - Download from [cursor.sh](https://cursor.sh) if not already installed

3. **Basic understanding of**:
   - Jupyter Notebooks
   - Command line operations
   - JSON configuration files

## Installation

### Step 1: Create Conda Environment and Install the MCP Server

```bash
# Create a dedicated conda environment
conda create -n cursor_mcp python=3.10 -y

# Activate the environment
conda activate cursor_mcp

# Install the MCP server
pip install cursor-notebook-mcp
```

### Alternative: Install in existing conda environment
```bash
# Activate your existing environment
conda activate your_env_name

# Install the package
pip install cursor-notebook-mcp
```

This will install the server along with core dependencies:
- `mcp`: Model Context Protocol framework
- `nbformat`: Jupyter notebook format library
- `nbconvert`: Notebook conversion utilities
- `paramiko`: SSH/SFTP support for remote operations

### Step 2: Install Optional Dependencies

For advanced export formats, install Pandoc:

**macOS:**
```bash
brew install pandoc
```

**Ubuntu/Debian:**
```bash
sudo apt-get install pandoc
```

**Windows:**
Download from [pandoc.org](https://pandoc.org/installing.html)

### Step 3: Verify Installation

```bash
cursor-notebook-mcp --version
```

## Configuration

### Step 1: Start the Server

First, ensure your conda environment is activated:
```bash
conda activate cursor_mcp
```

Then choose one of the following methods:

**Method 1: Streamable HTTP (Recommended for Cursor)**
```bash
cursor-notebook-mcp --transport streamable-http --allow-root /path/to/notebooks
```

**Method 2: Server-Sent Events (SSE)**
```bash
cursor-notebook-mcp --transport sse --port 8081 --allow-root /path/to/notebooks
```

**Method 3: Standard I/O**
```bash
cursor-notebook-mcp --transport stdio --allow-root /path/to/notebooks
```

### Step 2: Configure Cursor

1. **Locate Cursor's configuration directory**:
   - macOS: `~/Library/Application Support/Cursor/`
   - Windows: `%APPDATA%\Cursor\`
   - Linux: `~/.config/Cursor/`

2. **Create or edit `mcp.json`** in the configuration directory:

```json
{
  "mcpServers": {
    "notebook_mcp": {
      "url": "http://127.0.0.1:8080/mcp",
      "transport": "http",
      "settings": {
        "allow_root": "/path/to/notebooks",
        "cost_efficient": true
      }
    }
  }
}
```

3. **For project-specific configuration**, create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "notebook_mcp": {
      "url": "http://127.0.0.1:8080/mcp",
      "settings": {
        "allow_root": "./notebooks",
        "cost_efficient": true
      }
    }
  }
}
```

### Step 3: Restart Cursor

After configuring, restart Cursor IDE to load the MCP server configuration.

## Available Tools

The Cursor Notebook MCP Server provides the following tools:

### Core Operations
- `notebook_create`: Create a new notebook
- `notebook_read`: Read entire notebook or specific cells
- `notebook_list`: List notebooks in a directory
- `notebook_delete`: Delete a notebook

### Cell Management
- `notebook_add_cell`: Add new cells (code/markdown)
- `notebook_edit_cell`: Modify existing cell content
- `notebook_delete_cell`: Remove cells
- `notebook_move_cell`: Reorder cells
- `notebook_copy_cell`: Duplicate cells

### Execution & Output
- `notebook_execute`: Run notebook cells
- `notebook_clear_output`: Clear cell outputs
- `notebook_get_output`: Retrieve cell execution results

### Export & Conversion
- `notebook_export`: Export to various formats (HTML, PDF, Markdown, etc.)
- `notebook_convert`: Convert between notebook formats

### Metadata & Search
- `notebook_get_metadata`: Access notebook metadata
- `notebook_set_metadata`: Modify metadata
- `notebook_search`: Search within notebooks

### Remote Operations
- `notebook_remote_*`: All above operations via SFTP/SSH

## Best Practices & Workflows

### 1. Initial Setup Workflow

```markdown
1. Start your notebook server with appropriate permissions
2. Test connection by asking Cursor to list notebooks:
   "Can you list all notebooks in my project?"
3. Verify tools are accessible:
   "What notebook tools are available?"
```

### 2. Development Workflow

#### Creating a New Analysis Notebook
```markdown
"Create a new notebook called 'data_analysis.ipynb' with:
- A markdown cell explaining the purpose
- A code cell importing pandas and numpy
- A code cell for loading data"
```

#### Iterative Development
```markdown
1. "Read the current state of analysis.ipynb"
2. "Add a new cell after cell 3 that performs data cleaning"
3. "Execute cells 1-4 and show me the output"
4. "If there are errors, edit cell 4 to fix them"
```

### 3. Cost-Efficient Mode Best Practices

When working with large notebooks:
- Enable cost-efficient mode in configuration
- Use specific cell references: "Edit cell 5" instead of "Edit the data processing cell"
- Request targeted reads: "Read cells 10-15" instead of entire notebook

### 4. Collaboration Workflow

```markdown
1. Use project-scoped configuration (.mcp.json) for team consistency
2. Set standard notebook directories for team access
3. Use metadata to track notebook purposes and ownership
```

### 5. Remote Notebook Editing

For working with notebooks on remote servers:

```json
{
  "mcpServers": {
    "remote_notebook": {
      "url": "http://127.0.0.1:8080/mcp",
      "settings": {
        "sftp_host": "remote-server.com",
        "sftp_username": "your-username",
        "sftp_key_path": "~/.ssh/id_rsa",
        "allow_root": "/home/user/notebooks"
      }
    }
  }
}
```

## Advanced Usage

### 1. Batch Operations

```markdown
"For all notebooks in the 'experiments' folder:
1. Add a markdown cell at the top with the creation date
2. Clear all outputs
3. Export to HTML for documentation"
```

### 2. Template-Based Development

Create notebook templates and use them:
```markdown
"Create a new notebook using the template at templates/ml_experiment.ipynb
Name it 'experiment_2024_01.ipynb' and update the title cell"
```

### 3. Automated Reporting

```markdown
"Every Friday:
1. Execute the weekly_report.ipynb
2. Export it to PDF
3. Save in reports/ directory with date stamp"
```

### 4. Integration with Version Control

```markdown
"Before committing:
1. Clear all outputs from notebooks in this project
2. Export executed versions to archive/
3. Update the notebook index in README.md"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Server Connection Failed
```bash
# Check if server is running
curl http://127.0.0.1:8080/health

# Verify port is not in use
lsof -i :8080
```

#### 2. Permission Denied Errors
- Ensure `--allow-root` points to correct directory
- Check file permissions: `ls -la /path/to/notebooks`
- Run server with appropriate user permissions

#### 3. Tool Not Found in Cursor
- Restart Cursor after configuration changes
- Verify MCP configuration syntax
- Check Cursor's developer console for errors

#### 4. Notebook Execution Failures
- Ensure all required packages are installed in the Python environment
- Check kernel specifications in notebook metadata
- Verify Jupyter kernels are properly installed

### Debug Mode

Enable verbose logging:
```bash
cursor-notebook-mcp --transport streamable-http --allow-root /path --log-level DEBUG
```

### Performance Optimization

1. **Use cost-efficient mode** for large notebooks
2. **Limit cell reads** to specific ranges when possible
3. **Cache frequently accessed notebooks** locally
4. **Use batch operations** instead of individual cell edits

## Security Considerations

1. **Restrict root directory access** to prevent unauthorized file access
2. **Use SSH keys** instead of passwords for remote connections
3. **Run server on localhost** unless specifically needed for remote access
4. **Regularly update** the MCP server package for security patches

## Conclusion

The Cursor Notebook MCP Server bridges the gap between Cursor IDE and Jupyter Notebooks, enabling powerful AI-assisted notebook development. By following this guide and best practices, you can create an efficient workflow for notebook-based data science and analysis projects.

For more information and updates:
- GitHub: [jbeno/cursor-notebook-mcp](https://github.com/jbeno/cursor-notebook-mcp)
- MCP Documentation: [docs.anthropic.com/mcp](https://docs.anthropic.com/en/docs/claude-code/mcp)