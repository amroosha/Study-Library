# pokemon-agent2 — Improved Pokémon TCG Agent

An improved, reliable agent for the **Kaggle Pokémon TCG AI Battle Challenge
Playground**, built around the existing 60-card **Dragapult ex** deck (`deck.csv`,
unchanged). The entry point is `agent.py::agent(observation) -> list[int]`.

## Deck & win condition
Dreepy → Drakloak → **Dragapult ex**, attacking with **Phantom Dive** (200 damage to
the Active + 6 damage counters spread across the opponent's Bench). Support:
- **Budew** — free *Itchy Pollen* (10 dmg + opponent Item-lock) for early tempo.
- **Drakloak** (*Recon Directive*) / **Dudunsparce** (*Run Away Draw*) — draw/filter.
- **Munkidori** (*Adrena-Brain*) — move damage counters; **Fezandipiti ex** / **Meowth ex** — utility.
- Trainers: Poffin / Poké Pad / Ultra Ball (search), Boss's Orders (gust), Crispin
  (energy accel), Lillie / Judge / Unfair Stamp (draw/disruption), Night Stretcher
  (recovery), Crushing Hammer (energy denial), Risky Ruins (stadium).
- Energy: Fire + Psychic power Phantom Dive; Darkness powers Munkidori.

## Architecture
1. **Card-aware, context-aware heuristic policy (the backbone).** Handles every
   selection context the engine emits (setup, switch, deck-search, discard,
   damage-counter placement, energy discard, yes/no, counts, skill order, special
   conditions, …). Each legal option is scored by *what it actually does* — which
   card, which target, which attack — not by a fixed action-type priority.
2. **Conservative Monte-Carlo lookahead (optional, `MC_ENABLED`, default OFF).**
   Uses the SDK search API with a *verified-correct* hidden-state estimate, the
   heuristic as the rollout policy, and a richer evaluator. It only searches the
   top cluster of near-equivalent heuristic options and only overrides the
   heuristic on a meaningful rollout advantage. Gated by `remainingOverageTime`.

### Key strategic improvements over the original
- **Real option meaning.** The original scored options only by action type. This
  agent resolves the actual card/target/attack of every option (including PLAY
  options, which carry `index` but no `area`) and scores it situationally.
- **Attack selection** computes damage with weakness/resistance, knockout/prize
  value, game-winning KOs, Phantom Dive's Bench spread, and dynamic-damage attacks
  (Tenacious Tail, Cruel Arrow). Since attacking ends the turn, beneficial setup is
  done first.
- **Energy planning** targets the Dragapult line's Fire+Psychic cost and favours the
  active attacker; avoids wasting Darkness energy off Munkidori.
- **Donk protection:** with an empty Bench, fetching/playing a Basic is prioritised
  to avoid losing to a lone-Active knockout.
- **Draw-supporter timing** (Lillie/Judge) is based on current hand quality.
- **Phantom Dive counters** are allocated to maximise Bench knockouts.
- **Turn order:** prefers going second (Budew's turn-2 Item-lock); the original's
  comment/behaviour mismatch is fixed (YES = go first, NO = go second).

## Reliability
- Every path returns indices in `[0, len(option))`, respecting `minCount`/`maxCount`
  with no duplicates; a global guard falls back to a safe legal choice on any
  unexpected observation.
- No per-game mutable state → safe across Best-of-3 rounds.
- If `cg` metadata is unavailable, a built-in fallback card DB keeps it functional.

## Run / test
The `cg` engine package lives in the sibling `pokemon-agent/` (bundled with a Kaggle
submission). From a harness that puts `pokemon-agent` on `sys.path`:
```python
from agent import agent            # pokemon-agent2/agent.py
action = agent(observation)        # list[int] of option indices (or 60-card deck)
```
Local verification used `cg.game.battle_start / battle_select / battle_finish`.

## Test results (local, engine RNG)
| Opponent | Result |
|---|---|
| Original baseline (`pokemon-agent/agent.py`) | ~95–100% across seeds (e.g. 30–0, 29–1, 19–1, 17–3) |
| Random legal agent | ~95–100% (20–0, 19–1) |
| Self-play | stable ~50/50, mostly prize endings, **0 errors / 0 invalid moves** |
| Robustness/fuzz (malformed + 300 random MAIN states) | **322 tests, 0 failures** |
| Hidden-state reconstruction (6 full games, 283 stadium states) | **466 checks, 0 failures** |
| Fallback metadata (no `cg`) vs random | 8–8 |

Runtime: ~0.1 ms/decision (heuristic), well within the 600 s Best-of-3 overage bank.

## Monte-Carlo decision
The search API, hidden-state construction (deck multiset subtraction incl. stadium),
and evaluator were **verified correct**. However, A/B self-play vs the baseline showed
**no reliable win-rate gain** — short-horizon rollouts undervalue this setup-heavy
deck's long-term energy/evolution planning and can override good heuristic decisions.
Per the "prefer simpler, reliable implementation" guidance, MC is **disabled by
default** but retained (correct and budgeted) for experimentation (`MC_ENABLED=True`).

## File-restriction compliance
- Only `agent.py` is the implementation deliverable. `deck.csv` here is a
  **byte-identical copy** of `pokemon-agent/deck.csv` (SHA-256 verified); the deck
  composition is unchanged. `main.py` is the standard `from agent import agent` shim.
- **No existing project file was modified** (hash-verified). No external dependencies.

## Limitations
- Opponent hidden cards use a mirror-deck prior minus observed cards (only relevant
  if MC is enabled); the true opponent deck is unknowable.
- Heuristic valuations are hand-tuned for this deck; they generalise poorly to other
  decks by design (the deck is fixed by the task).
- Rare deck-out losses in very long self-play games.
