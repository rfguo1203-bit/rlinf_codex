# LOCAL_CONTEXT.md

Purpose: persistent local collaboration context for this repository.  
This file stores user-specific constraints and durable working preferences that should be reused in future Codex sessions.

## User Environment Constraints
- Updated: 2026-02-25
- This local machine cannot run RLinf training/inference end-to-end.
- RLinf execution must be done on company servers.
- This local machine cannot connect to company servers.

## Collaboration Policy
- Default working mode is offline development on local code only:
  - architecture reading
  - code/config/doc updates
  - test/codepath preparation
  - runbook generation for remote execution
- Any runtime verification must be marked clearly as "not executed locally".
- Provide commands/checklists that can be copied to server-side execution.
- User preference:
  - AI agent maintains architecture markdown files and Mermaid graphs together.
  - Markdown documents are written in Chinese by default.
  - Standard Mermaid render command to keep graph images in sync:
    `mmdc -i diagrams/single_node_key_call_graph.mmd -o diagrams/single_node_key_call_graph.png -e png -s 3`
  - Git sync target:
    - `origin` uses SSH URL `git@github.com:rfguo1203-bit/rlinf_codex.git`
    - SSH route is configured via `~/.ssh/config` to `ssh.github.com:443`
  - Standard Git connectivity workaround (reuse in new projects):
    - Symptom: `github.com:443` HTTPS git timeout / blocked.
    - Use SSH over 443 instead of HTTPS.
    - Steps:
      - Generate key (if missing): `ssh-keygen -t ed25519 -C 'rkos-codex-github' -f ~/.ssh/id_ed25519_github_codex -N ''`
      - SSH config for GitHub:
        - Host `github.com`
        - HostName `ssh.github.com`
        - Port `443`
        - User `git`
        - IdentityFile `~/.ssh/id_ed25519_github_codex`
        - IdentitiesOnly `yes`
      - Set repo remote to SSH: `git remote set-url origin git@github.com:<owner>/<repo>.git`
      - Verify: `git ls-remote --heads origin`
      - Push: `git push -u origin <branch>`
    - If remote has unrelated bootstrap history, merge once with:
      `git merge origin/<branch> --allow-unrelated-histories`
  - Commit/push default policy:
    - When user says "上库", default behavior is:
      `git add -A` -> `git commit` -> `git push`
    - This means include all current workspace modifications unless the user explicitly asks to exclude files.

## Memory Capture Rule
- During conversations, if new information is likely to be useful across future sessions, append it to this file automatically.
- Candidate items to persist:
  - stable environment limitations
  - tool/network restrictions
  - long-term goals and priorities
  - preferred workflow constraints
- Do not store secrets (tokens, passwords, private keys).

## Installed Codex Skills
- Updated: 2026-02-25
- Installed from `openai/skills` curated list:
  - `doc`
  - `pdf`
  - `gh-fix-ci`
  - `gh-address-comments`
  - `playwright`
