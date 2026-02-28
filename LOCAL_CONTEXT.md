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

## Session Notes (2026-02-25)
- RLinf architecture analysis focus for today:
  - Target run command: `examples/embodiment/run_embodiment.sh libero_10_grpo_openpi_pi05`
  - Scope fixed to single-node path.
  - User focuses on call graph and source mapping (not formulas in doc output).
  - New highest-priority task (status: not completed): user will read code following the call order for the target command, ask about classes/functions and their call relationships; assistant answers, then summarize and draw a call graph from those answers.
  - Code reading priority: focus on core environment classes like `OffScreenRenderEnv`; wrappers for multi-process/distributed are not a priority unless they affect core behavior.
  - New summary doc created for worker init core logic: `summaries/worker_init_core_summary.md` (includes Actor/Rollout/Env init_worker core steps + Mermaid call graph).
  - Env core execution chain identified (Libero path): `EnvWorker` → `LiberoEnv` → `OffScreenRenderEnv` → `ControlEnv` → `Libero_*` task → `BDDLBaseDomain` → `SingleArmEnv` → `ManipulationEnv` → `RobotEnv` → `MujocoEnv`.
  - robosuite dependency pinned locally to v1.4.1 for `SingleArmEnv` (path: `external_deps/robosuite`, version 1.4.1).
- Main architecture artifact created and maintained:
  - `SINGLE_NODE_ARCHITECTURE_LIBERO10_GRPO_OPENPI_PI05.md` (Chinese)
  - Graph source: `diagrams/single_node_key_call_graph.mmd`
  - Rendered image: `diagrams/single_node_key_call_graph.png`
  - Graph render command (HD): `mmdc -i diagrams/single_node_key_call_graph.mmd -o diagrams/single_node_key_call_graph.png -e png -s 3`
- External dependencies prepared for cross-repo tracing:
  - `external_deps/openpi`
  - `external_deps/LIBERO`
- Git connectivity resolution finalized:
  - HTTPS to `github.com:443` may be blocked in execution environment.
  - Use SSH over 443 (`ssh.github.com:443`) via `~/.ssh/config`.
  - Repo remote policy:
    - `origin`: `git@github.com:rfguo1203-bit/rlinf_codex.git`
    - `upstream`: `https://github.com/RLinf/RLinf`
  - Dedicated key used: `~/.ssh/id_ed25519_github_codex`
- Collaboration defaults confirmed:
  - Markdown docs written in Chinese by default.
  - When user says "上库", default flow is `git add -A` -> `git commit` -> `git push` (all current changes unless user excludes files).
