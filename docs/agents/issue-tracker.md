# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Which repo

**Code and issues live in different repos. Every `gh` invocation must be pinned explicitly.**

| | Repo |
|---|---|
| Code / branches / PRs | `geeningwang/sglang-jax` — this is `git remote origin` |
| Issues | `primatrix/sglang-jax` — **always pass `--repo primatrix/sglang-jax`** |

Do **not** let `gh` infer the repo from `git remote`. Inside a clone it defaults to
`geeningwang/sglang-jax`, which is the wrong tracker and will silently create or read
issues in the wrong place. The `--repo` flag is mandatory on every `gh issue` command and
every `gh api` path below; the command examples already carry it.

For `gh pr`, the default (`geeningwang/sglang-jax`) is correct — pin it anyway so the
distinction stays visible at the call site.

> **Prerequisite:** the `gh` CLI is not currently installed on this machine.
> Run `sudo apt install gh && gh auth login` before using these operations.

## Conventions

- **Create an issue**: `gh issue create --repo primatrix/sglang-jax --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --repo primatrix/sglang-jax --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --repo primatrix/sglang-jax --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --repo primatrix/sglang-jax --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --repo primatrix/sglang-jax --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --repo primatrix/sglang-jax --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --repo geeningwang/sglang-jax --comments` and `gh pr diff <number> --repo geeningwang/sglang-jax` for the diff.
- **List external PRs for triage**: `gh pr list --repo geeningwang/sglang-jax --state open --json number,title,body,labels,author,authorAssociation,comments` then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close` — all with `--repo geeningwang/sglang-jax`.

Because issues and PRs live in **different repos** here, a bare `#42` is ambiguous across
repos rather than within one: it is `primatrix#42` in issue text and `geeningwang#42` in PR
and commit text. Resolve from context rather than guessing, and write cross-repo references
in full (`primatrix/sglang-jax#42`) so they stay unambiguous.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo primatrix/sglang-jax --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

All of these operate on the **issue** repo, `primatrix/sglang-jax`.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --repo primatrix/sglang-jax --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies** — the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/primatrix/sglang-jax/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/primatrix/sglang-jax/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only — the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --repo primatrix/sglang-jax --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --repo primatrix/sglang-jax --add-assignee @me` — the session's first write.
- **Resolve**: `gh issue comment <n> --repo primatrix/sglang-jax --body "<answer>"`, then `gh issue close <n> --repo primatrix/sglang-jax`, then append a context pointer (gist + link) to the map's Decisions-so-far.
