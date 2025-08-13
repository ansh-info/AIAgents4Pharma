# GitHub Workflows Documentation

This document provides a comprehensive overview of all GitHub Actions workflows in the AIAgents4Pharma repository, detailing their purpose, triggers, and functionality.

## Overview

Our CI/CD pipeline uses **UV** for fast, reliable dependency management across all workflows. All workflows are designed to be efficient, secure, and provide comprehensive quality assurance.

## Workflow Categories

### 🧪 Testing Workflows
- [`tests_talk2aiagents4pharma.yml`](#tests-talk2aiagents4pharma)
- [`tests_talk2biomodels.yml`](#tests-talk2biomodels)
- [`tests_talk2knowledgegraphs.yml`](#tests-talk2knowledgegraphs)
- [`tests_talk2scholars.yml`](#tests-talk2scholars)
- [`tests_talk2cells.yml`](#tests-talk2cells)

### 🔒 Security & Quality
- [`security_audit.yml`](#security-audit)
- [`sonarcloud.yml`](#sonarcloud-analysis)
- [`pre_commit.yml`](#pre-commit)

### 🐳 Build & Deploy
- [`docker_build.yml`](#docker-build)
- [`docker_compose_release.yml`](#docker-compose-release)
- [`package_build.yml`](#package-build)
- [`release.yml`](#release)

### 📚 Documentation
- [`mkdocs-deploy.yml`](#mkdocs-deploy)

---

## Testing Workflows

### Tests Talk2Scholars

**File:** `tests_talk2scholars.yml`

**Purpose:** Comprehensive testing and quality checks for the Talk2Scholars component

**Triggers:**
- Pull requests to `main` with changes to:
  - `aiagents4pharma/talk2scholars/**`
  - `pyproject.toml`
  - `uv.lock`
- Manual workflow dispatch

**Jobs:**

#### 1. Code Quality Checks
- **Runner:** Ubuntu Latest
- **Dependencies:** UV sync with frozen lockfile
- **Checks:**
  - Pylint analysis with specific disabled rules
  - Ruff linting for code style
  - Bandit security scanning

#### 2. Cross-Platform Testing Matrix
- **Strategy:** Fail-fast disabled for comprehensive testing
- **Matrix:**
  - OS: Ubuntu Latest, macOS 15, Windows Latest
  - Python: 3.12
- **Steps:**
  - UV dependency installation
  - Test execution with coverage
  - Coverage reporting and XML generation
  - **100% coverage requirement** (builds fail if not met)
  - Codecov upload (Ubuntu only)

**Environment Variables:**
```yaml
OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
ZOTERO_API_KEY: ${{ secrets.ZOTERO_API_KEY }}
ZOTERO_USER_ID: ${{ secrets.ZOTERO_USER_ID }}
```

---

## Security & Quality Workflows

### Security Audit

**File:** `security_audit.yml`

**Purpose:** Comprehensive security scanning and vulnerability detection

**Triggers:**
- Weekly schedule (Sundays at 2 AM UTC)
- Push to `main` affecting `pyproject.toml` or `uv.lock`
- Pull requests to `main` affecting dependency files
- Manual workflow dispatch

**Jobs:**

#### 1. Dependency Security Scan
- **Tools:** pip-audit, safety
- **Outputs:** JSON and Markdown reports
- **Features:** Continues on error to allow other jobs

#### 2. Code Security Scan (Bandit)
- **Tool:** Bandit static analysis
- **Outputs:** JSON and TXT reports
- **Configuration:** Uses `pyproject.toml` settings

#### 3. SARIF Upload
- **Purpose:** Integration with GitHub Security tab
- **Process:**
  - Downloads all security reports
  - Converts Bandit JSON to SARIF format
  - Uploads to GitHub Security dashboard
- **Limitation:** Only runs on push events (not PRs)

#### 4. Security Summary Generation
- **Output:** Markdown summary with vulnerability counts
- **Includes:** Dependency and code security issue counts by severity

#### 5. SonarCloud Security Integration
- **Trigger:** Push to main branch only
- **Features:**
  - Downloads security reports
  - Generates coverage for SonarCloud
  - Performs comprehensive security analysis

**Key Features:**
- Artifact upload for all reports
- Error tolerance to prevent blocking development
- Integration with external security tools

### SonarCloud Analysis

**File:** `sonarcloud.yml`

**Purpose:** Advanced code quality analysis and technical debt tracking

**Triggers:**
- Push to `main` branch
- Pull requests to `main`
- Manual workflow dispatch

**Process:**
1. **Setup:** UV dependency installation
2. **Testing:** Full test suite with coverage generation
3. **Analysis:** Pylint JSON output generation
4. **Security:** Bandit security scan
5. **Upload:** SonarCloud analysis with all reports

**Artifacts:**
- Coverage XML
- Pylint JSON report
- Bandit security report
- 30-day retention period

### Pre-Commit

**File:** `pre_commit.yml`

**Purpose:** Automated code quality enforcement using pre-commit hooks

**Triggers:**
- All pull requests
- Manual workflow dispatch

**Features:**
- Uses UV for dependency management
- Runs all configured pre-commit hooks
- Ensures consistent code formatting and quality

---

## Build & Deploy Workflows

### Docker Build

**File:** `docker_build.yml`

**Purpose:** Build and push Docker images for all agents

**Features:**
- Multi-stage builds for optimized image sizes
- Pinned base images (no `latest` tags)
- Separate CPU and GPU variants
- Health check implementations
- Push to Docker Hub registry

### Docker Compose Release

**File:** `docker_compose_release.yml`

**Purpose:** Release management for Docker Compose configurations

**Features:**
- Separate compose files for CPU and GPU deployments
- Production-ready configurations
- Version tagging and release automation

### Package Build

**File:** `package_build.yml`

**Purpose:** Python package building and distribution

**Features:**
- Hatchling build backend
- PyPI distribution preparation
- Version management from `release_version.txt`

### Release

**File:** `release.yml`

**Purpose:** Automated release management

**Features:**
- Version tagging
- Release notes generation
- Asset uploads
- Distribution to PyPI

---

## Documentation Workflows

### MkDocs Deploy

**File:** `mkdocs-deploy.yml`

**Purpose:** Automated documentation deployment

**Features:**
- Jupyter notebook integration
- Material theme with modern styling
- GitHub Pages deployment
- Automatic updates on documentation changes

---

## Workflow Architecture Principles

### 1. **UV-First Approach**
All workflows use UV for dependency management:
```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v4
  with:
    enable-cache: true
    cache-dependency-glob: "uv.lock"

- name: Set up Python
  run: uv python install 3.12

- name: Install dependencies
  run: uv sync --frozen
```

### 2. **Security-First Design**
- Comprehensive security scanning in every workflow
- Artifact uploads for security reports
- Integration with GitHub Security dashboard
- SARIF format support for standardized security reporting

### 3. **Quality Assurance**
- Multi-platform testing matrices
- 100% code coverage requirements
- Multiple linting and formatting tools
- Pre-commit hook enforcement

### 4. **Efficiency & Reliability**
- Dependency caching with UV
- Fail-fast disabled for comprehensive testing
- Error tolerance where appropriate
- Frozen lockfile usage for reproducible builds

### 5. **Integration & Reporting**
- SonarCloud integration for advanced analysis
- Codecov for coverage tracking
- GitHub Security tab integration
- Automated artifact management

## Environment Variables & Secrets

### Required Secrets
```yaml
OPENAI_API_KEY          # OpenAI API access
ZOTERO_API_KEY          # Zotero integration
ZOTERO_USER_ID          # Zotero user identification
CODECOV_TOKEN           # Coverage reporting
SONAR_TOKEN             # SonarCloud analysis
GITHUB_TOKEN            # GitHub API access (auto-provided)
```

### Security Best Practices
- All secrets managed through GitHub Secrets
- No hardcoded credentials in workflows
- Minimal permission scopes
- Secure artifact handling

## Monitoring & Troubleshooting

### Workflow Status Badges
Add these badges to your README for real-time status monitoring:

```markdown
[![Security Audit](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/security_audit.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/security_audit.yml)
[![SonarCloud](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/VirtualPatientEngine/AIAgents4Pharma/actions/workflows/sonarcloud.yml)
```

### Common Issues & Solutions

#### UV Cache Issues
If you encounter UV cache problems:
```yaml
- name: Clear UV cache
  run: uv cache clean
```

#### Coverage Threshold Failures
The workflows enforce 100% coverage. To handle this:
1. Add proper tests for uncovered code
2. Use coverage exclusions in `pyproject.toml` for legitimate cases
3. Temporarily adjust threshold if needed for development

#### Security Scan Failures
Security workflows continue on error by design. Check artifacts for:
- Vulnerability reports from pip-audit and safety
- Security issues from Bandit
- SARIF uploads in GitHub Security tab

## Maintenance

### Regular Updates
- **Dependencies:** Dependabot manages updates automatically
- **Actions:** Update action versions quarterly
- **Python:** Update matrix versions as new releases become available

### Performance Optimization
- **Caching:** UV caching is enabled across all workflows
- **Parallelization:** Jobs run in parallel where possible
- **Resource Usage:** Optimized for GitHub Actions limits

This workflow architecture provides comprehensive quality assurance, security scanning, and deployment automation while maintaining efficiency and reliability.
