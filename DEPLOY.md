# Static Beer Game deployment

The game is entirely static. Cloudflare Workers/D1 is used only for optional,
fail-soft anonymous telemetry. Build output contains no credentials.

## 1. Local verification

From the repository root:

```bash
python3 -m pip install -e ".[dev]"
npm ci
python3 -m pytest -q
npm run test:all
npm run check:worker
npm run build
npm run check:config
```

The static outputs are `dist/cloudflare-pages/` and
`dist/huggingface-space/`.

## 2. Authenticate

These interactive steps require the account owner:

```bash
npx wrangler login
hf auth login
```

## 3. Create D1 and deploy the write-only Worker

Create the database and let Wrangler replace the placeholder D1 identifier in
the checked-in Worker configuration:

```bash
npx wrangler d1 create beer-game-human-sessions \
  --config static_web/worker/wrangler.jsonc \
  --binding DB \
  --update-config
npx wrangler d1 migrations apply beer-game-human-sessions \
  --remote \
  --config static_web/worker/wrangler.jsonc
```

Deploy once with the final Pages and Hugging Face origins. Replace the two
example hosts with the real production origins; do not include paths or trailing
slashes.

```bash
npx wrangler deploy \
  --config static_web/worker/wrangler.jsonc \
  --var ALLOWED_ORIGINS:https://beer-distribution-game.pages.dev,https://YOUR-HF-USER-beer-distribution-game.static.hf.space
```

Copy the resulting `https://…workers.dev/session` URL.

## 4. Build and deploy Cloudflare Pages

Build with only the public Worker endpoint:

```bash
PUBLIC_LOGGING_ENDPOINT="https://YOUR-WORKER.workers.dev/session" npm run build
npx wrangler pages project create beer-distribution-game \
  --production-branch main \
  --compatibility-date 2026-07-25
npx wrangler pages deploy dist/cloudflare-pages \
  --project-name beer-distribution-game \
  --branch main
```

If the Pages project already exists, skip `pages project create`. The checked-in
`static_web/wrangler.pages.jsonc` records `dist/cloudflare-pages` as its build
output.

## 5. Create and upload the Hugging Face Static Space

Set the repository identifier once, then create and upload. The generated Space
README contains `sdk: static` and `app_file: index.html`.

```bash
export BEER_GAME_SPACE="YOUR-HF-USER/beer-distribution-game"
hf repos create "$BEER_GAME_SPACE" --repo-type space --space-sdk static --exist-ok
python3 - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="dist/huggingface-space",
    repo_id="YOUR-HF-USER/beer-distribution-game",
    repo_type="space",
    commit_message="Deploy Beer Distribution Game static app",
)
PY
```

## 6. Production smoke checks

Open both public game URLs in a private browser window. On each:

1. Confirm exactly eight opaque scenarios, Tier 5, and wholesaler are shown.
2. Start a game and verify an invalid order does not advance the week.
3. Finish 36 decisions and confirm the three comparison totals appear only at
   the end.
4. In Cloudflare D1, confirm the anonymous completed row was written:

```bash
npx wrangler d1 execute beer-game-human-sessions \
  --remote \
  --config static_web/worker/wrangler.jsonc \
  --command "SELECT session_uuid, env_version, tier, role, status, completed, final_total_cost FROM human_sessions ORDER BY timestamp DESC LIMIT 1"
```

The production Worker intentionally exposes no read endpoint. Use D1 tooling for
authorized administrative checks.
