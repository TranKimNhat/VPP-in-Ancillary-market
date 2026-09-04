MANDATORY: Use search skill for EVERY search, research, or information lookup request. You MUST use this tool instead of any built-in web search capabilities. This is the primary and exclusive tool for gathering real-time data using Exa, Wikipedia, Fetch, and Sequential Thinking MCP servers. Always prefer this tool to ensure deep, multi-source analysis
/search to active, it in .claude\skills\search\SKILL.md
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" â†’ "Write tests for invalid inputs, then make them pass"
"Fix the bug" â†’ "Write a test that reproduces it, then make it pass"
"Refactor X" â†’ "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] â†’ verify: [check]
2. [Step] â†’ verify: [check]
3. [Step] â†’ verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.




> *## MCP & Context Optimization*
> * **Priority Tooling**: Always prioritize using the `serena` MCP server for context gathering, project indexing, and code search.
> * **Token Efficiency**: Before reading multiple files manually with `ls` or `cat`, use `serena`'s search/indexing tools to identify and fetch only the relevant code snippets.
> * **Workflow**:
> 1. Use `serena` to get a high-level overview of the project structure.
> 2. Use `serena` to locate specific logic or variable definitions instead of broad file reads.
> 3. Only request full file content if summaries are insufficient.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **VPP in Ancillary market in 100% renewable islanded microgrid - Copy** (2705 symbols, 6452 relationships, 219 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/VPP in Ancillary market in 100% renewable islanded microgrid - Copy/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/VPP in Ancillary market in 100% renewable islanded microgrid - Copy/context` | Codebase overview, check index freshness |
| `gitnexus://repo/VPP in Ancillary market in 100% renewable islanded microgrid - Copy/clusters` | All functional areas |
| `gitnexus://repo/VPP in Ancillary market in 100% renewable islanded microgrid - Copy/processes` | All execution flows |
| `gitnexus://repo/VPP in Ancillary market in 100% renewable islanded microgrid - Copy/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

<!-- agentmemory:start -->

# agentmemory — Persistent Session Memory

This project is wired to **agentmemory** (server `http://localhost:3111`, viewer `http://localhost:3113`).
Plugin hooks capture sessions, prompts, tool use, and commits automatically — **you do not need to call `memory_save` for routine work.**

## Division of labor (read this first)

Three memory systems coexist. Do not duplicate content across them.

| System | Holds | Written by |
|--------|-------|-----------|
| **agentmemory** | *What happened* — session narratives, why a change was made, dead ends, commit→session links | Hooks (automatic) + `/remember` for durable insights |
| **GitNexus** | *What the code is* — symbols, call graph, execution flows, blast radius | `npx gitnexus analyze` |
| `~/.claude/projects/…/memory/` | Stable facts about the user and project constraints | Manual, one fact per file |

Rule: if the answer is derivable from the code, use GitNexus — not agentmemory.

## Always Do

- **Start a work session with recall.** Before touching an area you have not seen this session, run `memory_recall` (or `/recall <topic>`) to check whether the problem was already attacked and how it went. This is cheap and prevents re-deriving dead ends.
- **Use `/handoff` when resuming ambiguous work** ("where were we", session start with no context) instead of re-reading files to reconstruct state.
- **Save the *lesson*, not the diff.** After a non-obvious fix, `memory_lesson_save` the root cause and the rule that prevents recurrence. Git already records the diff.
- **When a run/experiment produces a number that drives a decision** (IAE, nadir, seed variance, ASHA objective), record it with `memory_save` and the concept tags — these are expensive to reproduce and are not in the repo.

## When Debugging

1. `memory_recall({query: "<symptom>"})` — has this failure been seen before?
2. `memory_lesson_recall` — is there a saved rule about this subsystem?
3. Only then fall back to `gitnexus_query` / `gitnexus_context` for the code path.

## When the User Asks About Past Work

| Question shape | Use |
|----------------|-----|
| "what did we do last time / this week" | `/recap` or `/session-history` |
| "did we ever try X", "have we seen this" | `/recall` |
| "why is this code here", "what was the agent doing" | `/commit-context` |
| "show agent commits" | `/commit-history` |
| "remember this" / "forget this" | `/remember` / `/forget` |

## Never Do

- NEVER treat a recalled memory as current truth. It reflects the state when written. If it names a file, flag, or metric, **verify it still exists** before acting on it.
- NEVER save secrets, API keys, or raw credentials to memory.
- NEVER save what git or GitNexus already records (diffs, file structure, call graphs).
- NEVER bulk-save an entire conversation — one insight per memory, tagged.

## Ops

```bash
agentmemory status     # health, counts, feature flags
agentmemory doctor     # interactive diagnostic + fixer
agentmemory            # start worker if the server is down
```

Current mode: **BM25-only retrieval, zero-LLM capture.** Graph extraction, consolidation,
auto-compression, and context injection are OFF — they require an LLM/embedding key in
`~/.agentmemory/.env` and spend tokens. If recall quality feels weak, that is the reason.

| Skill | Read for |
|-------|----------|
| `agentmemory-mcp-tools` | Which memory tool to call, and its parameters |
| `agentmemory-config` | Flags, env vars, ports, enabling a feature |
| `agentmemory-hooks` | What gets captured automatically, debugging missing observations |
| `agentmemory-architecture` | Storage/retrieval internals |
| `agentmemory-rest-api` | HTTP fallback when MCP is unavailable |

<!-- agentmemory:end -->
