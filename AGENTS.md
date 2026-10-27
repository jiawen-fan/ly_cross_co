# Research Pipeline Rules

## Environment
- Scientific code executes on SMU M3 HPC.
- GitHub hosts source code and lightweight results.
- Large simulation datasets remain on HPC.

## HPC Security Restrictions
- Never run AI agents on M3.
- Never make AI API calls from M3.
- Never install AI API clients on M3.
- Never store AI API credentials on M3.
- AI-assisted code development happens externally.
- Git push/pull and ordinary scientific software are allowed.

## Scientific Requirements
- Preserve physical units and coordinate conventions.
- Document Fourier normalization.
- Test changes to power-spectrum estimators.
- Record experiment configurations and random seeds.
- Do not commit simulation datasets or sensitive files.

## Development
- Prefer feature branches and pull requests.
- Test code before merging.
- Do not modify validated results without explanation.
