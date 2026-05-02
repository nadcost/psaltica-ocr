# Psaltica OCR Agent Context

  This project is a local OCR/training workspace for Psaltica Praxis.

  Primary app reference:
  - /Users/nadcost/psaltica-praxis

  When working here, always read:
  - /Users/nadcost/psaltica-praxis/AGENTS.md
  - /Users/nadcost/psaltica-praxis/CLAUDE.md
  - /Users/nadcost/psaltica-praxis/app/core/toolbars.ts
  - /Users/nadcost/psaltica-praxis/app/core/notation/clusterCatalog.ts
  - /Users/nadcost/psaltica-praxis/app/core/notation/clusterParser.ts
  - /Users/nadcost/psaltica-praxis/app/core/music/actionMap.ts
  - /Users/nadcost/psaltica-praxis/app/core/keySignatures.ts

  Rules:
  - Do not modify /Users/nadcost/psaltica-praxis unless explicitly asked.
  - OCR output must target Psaltica Praxis composition insert strings.
  - ML classes should use Psaltica icon names from the app.
  - Cluster assembly must preserve Psaltica's rule: a glyph cluster is the editing unit.

## Task Tracking: Beads (`bd`)

  This project uses Beads (`bd`) for task tracking. State lives in `.beads/` inside this repo — do not share it with psaltica-praxis.

  Initial setup (run once):
  ```
  bd init
  bd prime
  ```

  Initial issues to create:
  ```
  bd create --title="Define Psaltica OCR symbol taxonomy" --type=task --priority=1
  bd create --title="Build PDF page rendering pipeline" --type=task --priority=1
  bd create --title="Set up annotation workflow" --type=task --priority=1
  bd create --title="Train first printed-score symbol detector" --type=feature --priority=2
  bd create --title="Implement cluster assembly into Psaltica insert strings" --type=feature --priority=1
  bd create --title="Define OCR JSON export format for Psaltica Praxis" --type=task --priority=1
  ```

  Durable project rules (run once):
  ```
  bd remember "Psaltica OCR outputs must use Psaltica Praxis icon names and composition insert strings, not independent OCR labels."
  bd remember "The mobile app lives at /Users/nadcost/psaltica-praxis and is reference-only unless explicitly requested."
  ```

  Project tracks:
  - dataset ingestion
  - PDF rendering
  - annotation workflow
  - symbol taxonomy
  - model training
  - cluster assembly
  - evaluation metrics
  - Psaltica export format
  - future app import


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
