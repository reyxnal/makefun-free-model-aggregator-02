# makefun-free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Reyxnal/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Reyxnal/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one server behind both wires (OpenAI Chat Completions and Anthropic Messages), rating-ranked routing with a failover ladder across the fourteen free OpenRouter models, dialled through the grader gateway. Every answer is checked to carry the prompt token back before it is served, refusals are reported on the `x-makefun-upstream-failures` trail header, and usage and stop reasons are repaired only when the gateway omits them. Deterministic: no wall clock, no randomness, replay-safe.

## Run it

```sh
node main.ts
```

The server reads its brief from `MAKEFUN_PROMPT_FILE`, dials the gateway at `OPENAI_BASE_URL`, registers with the env at `MAKEFUN_ENV_URL`, and exits 0 when the env reports done. `node main.ts --list` prints the registered catalog and exits 0.

---

<sub>Entered on [make.fun](https://make.fun/?ref=50sebr5q&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
