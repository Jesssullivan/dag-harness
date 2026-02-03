## Summary

Closes #{{ issue_iid }}

{% if is_new_role %}
**NEW ROLE**: Adds the `{{ role_name }}` Ansible role with full molecule testing and deployment automation.
{% else %}
**VALIDATION**: Re-validates existing `{{ role_name }}` role with molecule testing.

{% if role_diff_stat %}
### Changes from main
```
{{ role_diff_stat }}
```
{% else %}
> No file changes detected - validation run only.
{% endif %}
{% endif %}

**Wave**: {{ wave_number }} ({{ wave_name }})

---

{{ test_evidence }}

---

## Changes

{% if is_new_role %}
### Role Implementation
- `ansible/roles/{{ role_name }}/` - Complete role implementation
  - `tasks/main.yml` - Main task execution
  - `defaults/main.yml` - Default variables
  - `meta/main.yml` - Role metadata and dependencies
  - `molecule/` - Molecule test configuration

### Integration
- `package.json` - npm scripts for deployment
- `ansible/site.yml` - Tag integration for `--tags {{ role_tags }}`
{% else %}
### Validation Only
This MR validates the existing `{{ role_name }}` role passes molecule tests.
{% if role_diff_stat %}
See diff stat above for any changes made.
{% endif %}
{% endif %}

{% if credentials %}
### Credentials (KeePassXC)
{% for cred in credentials %}
- `{{ cred.entry }}` - {{ cred.purpose }}
{% endfor %}
{% endif %}

## Test Plan

### 1. Molecule Tests
```bash
# Full test cycle
npm run molecule:role --role={{ role_name }}

# Or step-by-step
cd ansible/roles/{{ role_name }}
molecule converge
molecule verify
molecule destroy
```

### 2. Integration Deploy
```bash
# Deploy to test target
npm run deploy:{{ deploy_target }} -- --tags {{ role_tags }}

# Verify deployment
npm run platform:health  # For platform roles
```

### 3. Idempotency Check
```bash
# Second run should show no changes
npm run deploy:{{ deploy_target }} -- --tags {{ role_tags }}
```

## Verification Checklist

### Code Quality
- [ ] ansible-lint passes with production profile
- [ ] All variables use `{{ role_name | replace('-', '_') }}_` prefix
- [ ] No hardcoded credentials or secrets
- [ ] Idempotent tasks (can run multiple times safely)

### Testing
- [ ] Molecule tests pass locally
- [ ] Deployed successfully to {{ deploy_target }}
- [ ] No regressions in dependent roles
- [ ] Verified service/application is functional

### Documentation
- [ ] Role README.md is complete
- [ ] meta/main.yml has accurate metadata
- [ ] defaults/main.yml documents all variables

### Security
- [ ] Credentials retrieved from KeePassXC (not hardcoded)
- [ ] No sensitive data in logs or output
- [ ] Proper file permissions set

## Related

{% if explicit_deps %}
### Dependencies (upstream)
{% for dep in explicit_deps %}
- Depends on: `{{ dep }}`
{% endfor %}
{% endif %}

{% if reverse_deps %}
### Dependents (downstream roles that use this one)
{% for dep in reverse_deps %}
- Used by: `{{ dep }}`
{% endfor %}
{% endif %}

---

/label ~role ~ansible ~{{ wave_name | lower | replace(' ', '-') }}
/assign @{{ assignee }}
