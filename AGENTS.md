# Repository Guidelines

## Project Structure & Module Organization

This repository contains a small Python-based Ottoneu auction/draft simulation. Source files and data currently live together in `data/`:

- `data/main.py` is the executable entry point for the full 480-pick simulation.
- `data/valuation.py` loads and normalizes batter, pitcher, and prospect CSV data.
- `data/agent.py` defines the bidding agent behavior and roster constraints.
- `data/auction.py` runs limit-order-book auction logic.
- `data/*.csv` files are input datasets and generated draft outputs.

Generated caches such as `data/__pycache__/` should not be committed.

## Build, Test, and Development Commands

Run commands from `data/` because `main.py` uses relative CSV paths:

```bash
cd data
python3 main.py
```

This loads the player pool, runs the 12-team simulation, and writes `full_480_draft_results.csv`.

Install runtime dependencies in your preferred virtual environment:

```bash
python3 -m pip install pandas numpy
```

There is no build step or packaging configuration yet.

## Coding Style & Naming Conventions

Use standard Python 3 style with 4-space indentation. Keep functions small and data-oriented. Prefer descriptive `snake_case` names for functions, variables, and CSV-derived columns created by the code. Classes should use `PascalCase`, as in `OttoneuAgent` and `AuctionEnvironment`.

Keep module imports simple and local to the `data/` script layout unless the project is later converted into a package. Add comments only when they clarify domain rules or non-obvious auction behavior.

## Testing Guidelines

No automated test suite exists yet. When adding tests, place them under `tests/` at the repository root and use `pytest`. Recommended test names:

```text
tests/test_valuation.py
tests/test_agent.py
tests/test_auction.py
```

Focus tests on deterministic behavior: CSV cleaning, prospect valuation tiers, roster cap enforcement, budget constraints, and auction clearing-price rules. For stochastic bidding logic, seed or isolate randomness before asserting outcomes.

## Commit & Pull Request Guidelines

This repository has no commit history yet, so use clear imperative commit messages such as `Add auction clearing price tests` or `Refine prospect valuation tiers`.

Pull requests should include a short summary, the commands run, and notes about any changed input or output CSV files. Include before/after output snippets when simulation behavior changes materially. Avoid committing generated caches, local virtual environments, or exploratory notebooks unless they are intentionally part of the change.

## Agent-Specific Instructions

Preserve existing CSV filenames unless updating all call sites. The current scripts depend on relative paths, so verify changes from `data/` before handing off.

# Ottoneu Rules Context

This project models Ottoneu fantasy baseball roster construction, auction strategy, and dynasty valuation.

## Core League Structure

- Ottoneu baseball leagues are dynasty/keeper fantasy baseball leagues.
- Each team has a $400 salary cap.
- Each team can roster up to 40 players.
- A team does not need exactly 40 players at all times, but must preserve at least $1 of cap room for every empty roster spot.

Each team shall during the regular season maintain a roster of 22 major-league players that can fill out a starting lineup as defined below. The remaining 18 roster spots can be used for reserves, consisting of both major and minor leaguers.
A roster of 22 major-league players capable of filling out a starting lineup as defined below shall be maintained regardless of any games started, games played or innings limits.
If at any time during the season any team's cap room is not greater than or equal to 40 minus the number of players on that team's roster up to 40, that team is considered in an invalid state. In this circumstance, the team will be frozen until the issue is resolved. The manager will be forced to cut a player to resolve this issue, and will not be able to perform any other activities in the game, including editing and making auction bids. However, bids already placed will be processed as described in section V. below, and trades that have already been accepted will process as expected.
Functionally, this means that while teams do not need 40 players on their roster at all times, they must have at least $1 available for every free roster spot, not including extra roster spots due to a player being suspended, on the 60-day IL, on the COVID-19 IL, or opted out from the MLB season.
Extra roster spots granted due to a player being suspended, on the 60-day IL, on the COVID-19 IL, or opted out from the MLB season do not grant extra cap space and are not factored when ensuring a team has enough money to fill their roster spots.
At no time shall a team willingly go over roster and salary cap limits. If a team knowingly does this, they will face penalties at the discretion of their league's commissioner.
Team rosters will be filled at an annual auction to be held some time near the start of the major-league season, as determined by the league's participants.
A team's starting lineup depends on the scoring system of the league.
In non-head-to-head leagues, a starting lineup consists of one slot at each infield position (catcher, first base, second base, third base, and shortstop), five outfielder slots, one additional middle infielder (second base or shortstop) slot, one additional hitter from any position (utility slot), five starting pitcher slots, and five relief pitcher slots.
In head-to-head leagues and during the playoffs, a starting lineup consists of one slot at each infield position (catcher, first base, second base, third base, and shortstop), five outfielder slots, one additional middle infielder (second base or shortstop) slot, one additional hitter from any position (utility slot) and five relief pitcher slots. The number of starting pitcher slots depends on the league's Per Week GS Cap setting
If the Per Week GS Cap is set to 0, there are two SP slots
If enabled, there are a maximum 5 SP slots on a given day. The number of slots will decrease as the team gets closer to the GS cap, down to 0 if a team has no more starts available in a matchup
In regular season H2H games longer than one week (for example, the first matchup of the season and around the All-Star Break), the GS cap will be the same as it is for regular one week matchups
In two-week playoff games, the GS cap will reset at the end of the first week and before the second week begins on the second Monday in the matchup
Player positional eligibility
An offensive player is eligible to fill a position if he:
Played in 10 or more regular season major league games at that position in the current or preceding year
Started 5 or more regular season major league games at that position in the current or preceding year
Played in 20 or more regular season minor league games at that position in the current or preceding year
Players will be given SP eligibility if they are the starting pitcher in 5 regular season games in the current or preceding year
Players will be given RP eligibility if they pitch in relief in 5 regular season games in the current or preceding year
Players designated as pitchers by Ottoneu's stats provider will get a default RP eligibility if neither of the previous two rules apply to them
All non-pitcher players are given a default UTIL position elgibility.
The player pool comprises of all players in all Major League and Minor League baseball organizations, as well as living players who in the past have been under contract to play baseball in the Major or Minor Leagues

When implementing Ottoneu logic:

- Put hard rules in pure functions.
- Add tests for every rule.
- Avoid burying rules inside UI components.
- Keep valuation assumptions separate from official roster/cap constraints.
- When unsure whether something is an Ottoneu rule or a modeling assumption, mark it clearly as an assumption.

## Rule Examples

### Empty roster reserve

Team has 37 players and $390 salary.

Required reserve:
40 - 37 = $3

Maximum legal effective salary:
400 - 3 = $397

Team is legal.

### Illegal roster

Team has 37 players and $398 salary.

Required reserve:
40 - 37 = $3

Maximum legal effective salary:
400 - 3 = $397

Team is illegal.

### Cut penalty

Player salary: $7

Penalty:
ceil(7 / 2) = $4

The team now carries a $4 cap penalty unless/until the player is claimed, reauctioned by the cutting team under applicable rules, or the season ends.

# OTTONEU QUANT DESK - DOMAIN KNOWLEDGE

You are an expert Python Quantitative Developer building a Reinforcement Learning (RL) terminal for an Ottoneu Fantasy Baseball league. You must obey these league rules and mathematical strategies when generating or refactoring code.

## 1. THE IMMUTABLE LEAGUE RULES
- **Economy:** 12 teams. Hard salary cap of $400 per team. The minimum bid for any player is $1.
- **Roster Cap:** 40-man hard cap. Teams MUST reserve at least $1 for every empty roster spot they have. 
- **Starting Lineup (24 spots):** C(2), 1B(1), 2B(1), 3B(1), SS(1), MI(1), CI(1), OF(5), UTIL(1), SP(5), RP(5).
- **Bench (16 spots):** Players not filling a starting slot go to the Bench.
- **In-Season Cuts:** If a team cuts a player during the season, they only recover 50% of that player's salary (rounded up to the nearest dollar).
- **Off-Season:** Loans expire. All kept players have their salaries increased by $2. Teams are granted a $25 arbitration budget to allocate to rival players to force their salaries up.

## 2. THE RL STRATEGY ALGORITHMS
- **Limit Order Book (LOB):** We clear auctions instantly. Do not use +$1 while-loops. Collect limits from all 12 agents, highest wins, clearing price is the second-highest bid + $1.
- **Thompson Sampling:** Used for risk assessment. High-variance assets (like Dynasty Prospects) use a wider Beta distribution, allowing agents to occasionally overpay aggressively based on upside.
- **Actor-Critic:** The Actor dictates the bid limit. The Critic evaluates global roster equity. 
- **Scarcity Multiplier:** Agents must apply bid premiums (e.g., 1.2x) if they have open starting slots for premium positions (C, SS, SP), and discounts (e.g., 0.8x) for bench spots.

## 3. DATA SCHEMA EXPECTATIONS
We rely on three CSVs loaded via pandas:
1. `batters_auctioncalc.csv`
2. `pitchers_auctioncalc.csv`
3. `Baseball Composite Prospect List 2026 - List.csv`
- Key extracted columns: `Name` (String), `dollars` (Float - true FanGraphs value), `positions` (List of strings, e.g., ['SS', '2B']), and `is_prospect` (Boolean).