# Game-Theoretic Analysis of Online Advertising Competition

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-20%20passed-1BAF7A)
![Course](https://img.shields.io/badge/Course-Game%20Theory-14213D)

Course project for **Game Theory**, B.Tech Computer Science and Engineering (5th semester), **Manipal University Jaipur**.

Firms that sell online compete for the same customers through **ad budgets** and **ad auctions**. The value of every rupee a firm spends depends on what its rivals spend and bid, which makes this a strategic problem that game theory can solve. This project builds **five game-theoretic models** of that competition in Python, solves each one exactly, checks every solution by simulation, and compares the results with published theory. An **interactive Streamlit dashboard** lets you change the inputs and watch the equilibrium move.

<p align="center"><img src="assets/slide_title.png" width="720" alt="Title slide of the project presentation"></p>

## Team

| S. No. | Name | Registration no. |
|---|---|---|
| 1 | Ananya Shukla | 2427030029 |
| 2 | Ansh | 2427030307 |
| 3 | Aryan Tyagi | 2427030344 |
| 4 | Shaurya Singh | 2427030357 |
| 5 | Rohit Kumar Yadav | 2427030335 |

## Deliverables

| File | What it is |
|---|---|
| [`docs/Project_Report.pdf`](docs/Project_Report.pdf) | Full project report (29 pages): problem statement, objectives, methodology, implementation, results, screenshots, conclusion |
| [`docs/Project_Report.docx`](docs/Project_Report.docx) | Same report in Word format |
| [`docs/Presentation.pptx`](docs/Presentation.pptx) | 20-slide presentation with speaker notes |
| [`docs/Presentation.pdf`](docs/Presentation.pdf) | PDF copy of the slides |

## Problem statement

> How do competing firms choose advertising budgets and bids, what outcome does this competition reach, and how do the platform's auction rules change who gains and who loses?

## The five models and key results

| # | Model | Game-theory concepts | Key result |
|---|---|---|---|
| 1 | **Advertising war**: 2 firms choose a Low or High budget | Normal-form game, dominant strategy, Nash equilibrium, Pareto efficiency | A **Prisoner's Dilemma**: both firms spend High and lose **50% of joint profit** |
| 1b | **Repeated advertising war**: the same game every month | Repeated games, trigger strategies, discount factor | Low budgets survive without any agreement if **δ ≥ 0.20**; reciprocal strategies beat "Always High" |
| 2 | **Budget contest**: any budget, *n* firms | Tullock contest, best-response dynamics | Total spend = **(n−1)/n** of customer value; with 10 firms **90%** goes to the ad platform |
| 3 | **Keyword auctions**: Google / Meta style ad slots | Mechanism design: GSP, VCG, GFP | Truthful bidding is **not stable under GSP**, but GSP in equilibrium earns **exactly the VCG revenue** |
| 4 | **Bidding dynamics**: how bids change over time | Best-response dynamics, balanced bidding | First-price bidding never settles (**saw-tooth war**); GSP converges in **20 / 20** runs |
| 5 | **Display-ad auctions**: real-time bidding | Bayesian Nash equilibrium, revenue equivalence, optimal reserve | Best bid = **(n−1)/n** of value; first- and second-price earn the same; reserve **0.5** adds **+25%** revenue |

Every result above matches the published theory it is tested against (see [Testing](#testing)).

## Results

### Model 1: the advertising war is a Prisoner's Dilemma

<p align="center"><img src="outputs/figures/fig01_payoff_matrix.png" width="460" alt="Payoff matrix: (Low, Low) 40,40; (Low, High) 15,45; (High, Low) 45,15; (High, High) 20,20 is the Nash equilibrium"></p>

High is better whatever the rival does (45 > 40 and 20 > 15), so both firms spend High and earn 20 each instead of 40. The wasted ₹40 lakh goes to the ad platform.

**When does it happen?** We solved 5,670 versions of the game. Overspending happens when ads mainly *steal* customers (growth g < 1). When ads *create* demand (g > 1), spending High is good for both.

<p align="center"><img src="outputs/figures/fig02_game_type_map.png" width="680" alt="Game-type map by High budget and market growth"></p>

### Model 1b: repeated play keeps budgets low

<p align="center"><img src="outputs/figures/fig03_repeated_tournament.png" width="620" alt="200-month round-robin tournament results"></p>

### Model 2: more competitors, more value lost to the platform

<p align="center">
<img src="outputs/figures/fig04_best_response_convergence.png" width="440" alt="Best-response dynamics converge to the Nash equilibrium">
<img src="outputs/figures/fig05_rent_dissipation.png" width="440" alt="Rent dissipation as the number of firms grows">
</p>

### Model 3: GSP in equilibrium earns exactly the VCG revenue

<p align="center"><img src="outputs/figures/fig06_mechanism_revenue.png" width="720" alt="Revenue under VCG, truthful GSP and equilibrium GSP"></p>

| Mechanism (worked example, 5 bidders, 3 slots) | Price per click, slots 1–3 (₹) | Revenue per search | Stable? |
|---|---|---|---|
| VCG, truthful bids | 8.0, 7.0, 6.0 | ₹4.40 | Yes |
| GSP, truthful bids | 10.0, 8.0, 6.0 | ₹5.20 | **No**: Brand A earns more by bidding ≈ ₹8 |
| GSP, equilibrium bids (12, 8, 7, 6, 4) | 8.0, 7.0, 6.0 | ₹4.40 | Yes |

### Model 4: first-price bidding never settles; GSP does

<p align="center">
<img src="outputs/figures/fig07_gfp_sawtooth.png" width="440" alt="Saw-tooth bidding war in a first-price auction">
<img src="outputs/figures/fig08_gsp_convergence.png" width="440" alt="GSP balanced bidding converges to the equilibrium">
</p>

### Model 5: display-ad auctions and platform design

<p align="center">
<img src="outputs/figures/fig09_bid_shading.png" width="290" alt="Optimal bid shading">
<img src="outputs/figures/fig10_revenue_equivalence.png" width="290" alt="Revenue equivalence">
<img src="outputs/figures/fig11_reserve_price.png" width="290" alt="Optimal reserve price">
</p>

## Interactive dashboard

`streamlit run app.py` opens a dashboard with one page per model. Every slider re-solves the game in real time.

| | |
|---|---|
| <img src="assets/dashboard/advertising_war.png" alt="Advertising war page"> | <img src="assets/dashboard/repeated_game.png" alt="Repeated game page"> |
| **Advertising war**: change budgets and market growth; the game is classified and solved live | **Repeated game**: tournament of 7 strategies, with a mistakes slider |
| <img src="assets/dashboard/budget_contest.png" alt="Budget contest page"> | <img src="assets/dashboard/keyword_auctions.png" alt="Keyword auctions page"> |
| **Budget contest**: set firms and costs; equilibrium budgets and convergence | **Keyword auctions**: GSP vs VCG vs GFP, with automatic deviation checks |
| <img src="assets/dashboard/bidding_dynamics.png" alt="Bidding dynamics page"> | <img src="assets/dashboard/display_auctions.png" alt="Display auctions page"> |
| **Bidding dynamics**: the first-price saw-tooth and GSP convergence | **Display auctions**: bid shading, reserve price, revenue equivalence |

## How to run

Requires **Python 3.10 or newer**.

```bash
git clone https://github.com/ananya-csai/game-thoretic-simulations.git
cd game-thoretic-simulations
pip install -r requirements.txt

python main.py            # runs every experiment (~5 s), prints results, saves figures and tables
python main.py --quick    # fewer simulation trials, for a fast demo
python -m pytest -v       # runs the 20 automated tests
streamlit run app.py      # opens the interactive dashboard in the browser
```

On Windows you can also double-click **`run_all.bat`** (experiments + tests) or **`run_dashboard.bat`** (dashboard).

All simulations use fixed random seeds, so every run reproduces the numbers in the report exactly.

## Project structure

```
adgame/
  normal_form.py        Two-player game solver: pure and mixed Nash equilibria, dominance, IESDS, Pareto check
  ad_spend_game.py      Model 1:  advertising-war payoffs, game classification, critical discount factor
  repeated_game.py      Model 1b: seven strategies and the round-robin tournament
  budget_contest.py     Model 2:  closed-form Tullock equilibrium, best-response dynamics
  position_auctions.py  Model 3:  GSP, VCG, GFP auctions, EOS equilibrium bids, deviation checker, Monte Carlo
  dynamics.py           Model 4:  GFP saw-tooth dynamics, GSP balanced bidding
  auction_design.py     Model 5:  bid shading, revenue equivalence, reserve prices
  plotting.py           All figures
main.py                 Runs all experiments; writes outputs/
app.py                  Streamlit dashboard
tests/test_models.py    20 tests checking every model against theory
outputs/                figures/, tables/ (CSV), results.json, screenshots/, terminal output
docs/                   Project report and presentation
assets/                 Images used in this README
```

The game solvers are written from scratch. The `nashpy` library is used only inside the tests, to cross-check our mixed-equilibrium solver.

## Testing

Every model has at least one test comparing the program's answer with a published result. All **20 tests pass**.

| Test | Theory it checks |
|---|---|
| Ad war is a Prisoner's Dilemma with δ* = 0.20 | Standard repeated-game result |
| Mixed equilibria match `nashpy` on 3 games | Support enumeration is correct |
| Tullock NE matches the closed form; no firm gains from any other budget | Equilibrium condition |
| Truthful bidding is a NE under VCG | VCG is truthful |
| Truthful GSP fails on the textbook example (profit 1,200 vs 1,592) | Edelman, Ostrovsky & Schwarz (2007) |
| EOS bids are a NE and earn VCG revenue | Edelman et al. (2007); Varian (2007) |
| Balanced bidding converges to VCG revenue | Cary et al. (2007) |
| Best shading factor is (n−1)/n | First-price Bayesian NE |
| First- and second-price revenue equal (n−1)/(n+1) | Revenue equivalence |
| Optimal reserve is 0.5 | Myerson (1981) |

## Tech stack

Python · NumPy · Matplotlib · Pandas · Streamlit · Pytest

## Limitations

- Money values are illustrative, chosen to show the mechanisms, not fitted to real market data.
- Players are rational and, in Models 1–4, fully informed.
- Clicks depend only on slot position and a quality score.
- No daily budget limits inside the auctions.

## References

1. B. Edelman, M. Ostrovsky, M. Schwarz, "Internet advertising and the generalized second-price auction," *American Economic Review*, 97(1), 242–259, 2007.
2. H. R. Varian, "Position auctions," *International Journal of Industrial Organization*, 25(6), 1163–1178, 2007.
3. M. Cary et al., "Greedy bidding strategies for keyword auctions," *ACM Conference on Electronic Commerce*, 262–271, 2007.
4. B. Edelman, M. Ostrovsky, "Strategic bidder behavior in sponsored search auctions," *Decision Support Systems*, 43(1), 192–198, 2007.
5. G. Tullock, "Efficient rent seeking," in *Toward a Theory of the Rent-Seeking Society*, 97–112, 1980.
6. R. B. Myerson, "Optimal auction design," *Mathematics of Operations Research*, 6(1), 58–73, 1981.
7. W. Vickrey, "Counterspeculation, auctions, and competitive sealed tenders," *Journal of Finance*, 16(1), 8–37, 1961.
8. J. F. Nash, "Equilibrium points in n-person games," *PNAS*, 36(1), 48–49, 1950.
9. R. Axelrod, *The Evolution of Cooperation*, Basic Books, 1984.
10. Google, "Rolling out first price auctions to Google Ad Manager partners," Google Ad Manager blog, 2019.
