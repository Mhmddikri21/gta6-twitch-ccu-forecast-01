# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mhmddikri21/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mhmddikri21/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hour by hour (hours 2-24), as five quantiles per hour — from the case's hour-1 print and the release-day shapes of past rolling-midnight console launches. Python 3.13, standard library only, no network, deterministic.

Method: per hour, the median log-ratio to hour-1 across 14 rolling-midnight console analogues, adjusted by a per-hour regression slope on log(hour-1) (bigger midnight audiences grow less into the evening peak), with the residual SD as the lognormal spread; the level adjustment is capped at the largest hour-1 in the fit, so the slope is never extrapolated past the launches it was estimated from. Local validation on the four practice launches (true answers from SullyGnome): mean pinball loss 0.170, typical miss 26.5%, against the house baseline's 0.387 / 82.8%.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ekaq8a76&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
