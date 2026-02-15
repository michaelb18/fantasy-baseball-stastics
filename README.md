# fantasy-baseball-stastics
An exploratory project to determine the feasability of risk adjusted baseball stats.

## Approaches
1. The [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio) is a way to measure the risk-adjusted return of a security. It is computed via $\frac{\mathbb{E}[R_p - R_b]}{\sigma_p}$, where $R_p$ is a player's stat, $R_b$ is a risk-free stat (usually either replacement level or 0) and $\sigma_p$ is the variance of a player's projection.
2. Represent the entire team as a distribution by representing every player as a gaussian according to the stats we produce in approach 1. Our goal is to maximize total return (probably as measured by the total sharpe ratio of every stat). This can probably be written as a linear program like this:

```math
\begin{aligned}
\min_{x} \quad & -\sum_{j=1}^{n} \sum_{s=1}^{5} \frac{\mathbb{E}[R_{s, j}]}{\sigma_{s, j}}\\
\text{s.t.} \quad & |B| \leq 15 \\
& |P| \leq 9
\end{aligned}
```

A key issue in this LP is that its unbounded, since sharpe ratio can continue to infinity. In practice, it doesn't because there will be a single player with the highest sharpe ratio. This is similar to a backpack problem, which can be solved via linear programming, however it is complicated by having to guess prices of players in an auction draft, and those prices are not determined ahead of time. Furthermore, exact price predictions vary by league and by the real-time draft situation. For example, if there is only one good shortstop left and two teams need a shortstop, then that player will go for more money
