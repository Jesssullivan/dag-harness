# Documentation Sprint Plan

**Status**: Nearly Complete
**Goal**: Clean, auto-updating, LLM-friendly documentation with GitLab Pages
**Updated**: 2026-02-02

---

## Objectives

1. **Accurate README** reflecting actual capabilities
2. **Auto-generated API docs** via mkdocstrings
3. **LLM-friendly docs** (llms.txt, llms.md)
4. **GitLab Pages** with curl-to-shell bootstrap
5. **DRY documentation** - single source of truth

---

## Current State (Analysis Results)

| Area | Score | Notes |
|------|-------|-------|
| README accuracy | 60% | Missing MCP tools count, HOTL details, cost tracking |
| API docs | 70% | Good CLI/MCP coverage, no auto-generation |
| Code docstrings | 85% | Excellent coverage, Google-style |
| Vignettes | 30% | Only vignette 1 complete |
| LLM docs | 0% | No llms.txt or llms.md |

## Actual Capabilities (from code analysis)

- **14-node LangGraph DAG** with parallel test execution
- **40+ MCP tools** across 8 categories
- **HOTL mode** with autonomous operation
- **Cost tracking** per session/model
- **Golden metrics** with baselines
- **Hook system** (PreToolUse, PostToolUse)
- **SQLite/PostgreSQL** checkpointing

---

## Sprint Tasks

### Phase 1: Foundation (Tasks 1-5)

- [x] Create sprint plan (this file)
- [x] Add mkdocstrings to mkdocs.yml
- [x] Update .gitlab-ci.yml build-docs
- [x] Fix Python version (3.11+ everywhere)
- [x] Create llms.txt and llms.md

### Phase 2: Content (Tasks 6-10)

- [x] Rewrite README.md
- [x] Add API reference pages (docs/api/python.md)
- [ ] Complete vignette 02 (HOTL) - deferred
- [x] Add troubleshooting guide (docs/troubleshooting.md)
- [ ] Document cost tracking - deferred

### Phase 3: Polish (Tasks 11-15)

- [ ] Add curl-to-shell bootstrap - future
- [x] Create CONTRIBUTING.md
- [ ] Verify all internal links
- [x] Test GitLab Pages deployment (pipeline running)
- [x] Generate initial llms.txt from docs

---

## Technical Approach

### mkdocstrings Configuration

```yaml
plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true
```

### llms.txt Format

```
# DAG Harness

> Self-installing DAG orchestration for Ansible role deployments

## Quick Start
curl -sSL https://... | bash

## CLI
harness bootstrap
harness box-up-role <role>
...

## MCP Tools
40+ tools across 8 categories
...
```

### Auto-Update Strategy

1. mkdocstrings extracts from docstrings on build
2. CI generates llms.txt from built docs
3. GitLab Pages deploys on main branch push

---

## Success Criteria

- [ ] `mkdocs build` succeeds with API docs (CI pipeline verifying)
- [x] llms.txt is < 50KB and comprehensive (~5KB, 178 lines)
- [x] README reflects actual capabilities (14-node DAG, 40+ MCP tools)
- [ ] GitLab Pages shows updated docs (awaiting pipeline)
- [ ] No broken internal links
