# Repository guide

Kaguya packages a privacy-safe, portable Hermes + OneBot/NapCat QQ bot. The vendored
`onebot_adapter/` is derived from `DNAlec/hermes-onebot-adapter`; keep its attribution.

## Hard privacy boundary

Never commit credentials, QQ/group IDs, member names or preferences, chat/session databases,
logs, media, generated images, test transcripts, profile exports, `.env`, `auth.json`, or
NapCat login state. Public profile memories must describe only generic behavior and domain
knowledge. Run `./kaguya privacy-check` before every commit.

## Verification

```bash
./kaguya bootstrap
./.venv/bin/ruff check .
./.venv/bin/pytest -q
./kaguya privacy-check
```

Configuration changes must remain group-ID agnostic. Group routing is created only by the local
`kaguya bind` command and stored outside the checkout.
