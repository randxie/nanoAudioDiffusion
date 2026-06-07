---
name: git-push
description: Safely review, commit, and push repository changes to git. Use when the user asks Codex to upload code, push code, git push, commit and push, publish changes, or otherwise send local repository changes to the remote branch.
---

# Git Push

## Workflow

1. Inspect repository state with `git status --short` and identify changed, deleted, untracked, and ignored-relevant files.
2. Review the actual diff before staging:
   - Use `git diff -- <paths>` for tracked files.
   - Use targeted file reads for untracked files that may need staging.
   - Do not stage unrelated user changes unless the user explicitly asks.
3. Check repository ignore rules before pushing:
   - Ensure generated artifacts, `.download/`, `.artifacts/`, caches, local data, secrets, virtualenvs, and large downloaded files are ignored.
   - If necessary, update `.gitignore` before committing.
4. Run reasonable verification for the touched code when practical.
   - Use the project’s standard commands.
   - For this autotrader repo, prefer `.venv/bin/python` for Python checks.
5. Stage only intended files with explicit path arguments.
6. Create a concise commit message based on the actual change.
7. Push to `main` unless the user explicitly requests another branch.
8. Report success with commit hash and pushed branch, or report the exact failure if push fails.

## Safety Rules

- Never force push.
- Never run destructive git commands such as `git reset --hard` or `git checkout --` unless the user explicitly requested that operation.
- Do not revert unrelated changes.
- If unrelated dirty files exist, mention them and leave them unstaged.
- If files are deleted and the user did not ask to delete them, confirm they are intended before staging those deletions.
- If secrets, credentials, bulky artifacts, raw data dumps, or local caches appear in the diff, stop and fix ignore rules or ask before proceeding.
- If push is rejected because the remote has new commits, fetch and inspect before deciding whether to rebase, merge, or ask the user.

## Useful Commands

```bash
git status --short
git diff -- <path>
git add <path> ...
git commit -m "message"
git push origin main
git rev-parse --short HEAD
```

Use non-interactive git commands whenever possible.
