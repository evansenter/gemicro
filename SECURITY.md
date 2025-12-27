# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please report it privately.

**Do not file a public issue.**

Instead, please report it via GitHub's private vulnerability reporting:
1. Navigate to **Security** > **Advisories** > **Report a vulnerability**
2. Provide a detailed description of the issue

## Security Best Practices

### API Keys
- Never commit API keys to version control.
- Use environment variables (e.g., `GEMINI_API_KEY`) to provide credentials.
- The `Debug` implementation for `LlmClient` hides the underlying API client entirely to prevent accidental exposure of credentials in logs.
- Check generated logs to ensure no sensitive data is leaked before sharing them.

### Input Validation
- This tool uses LLMs which can be susceptible to prompt injection. While we structure prompts carefully to separate instructions from user data, treat LLM outputs as untrusted.
- When using the `eval` suite, ensure input datasets are from trusted sources.

### Dependencies
- Dependencies are automatically audited on Cargo.toml/Cargo.lock changes and weekly via GitHub Actions (see `.github/workflows/audit.yml`).
- Known issues:
    - `number_prefix` (via `indicatif`) is unmaintained (RUSTSEC-2025-0119). This is a display-only dependency and considered low risk.
