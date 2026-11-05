# Privacy model

This public repository contains behavior templates, not a snapshot of a live bot.

Included:

- generic persona and participation rules;
- generic kigurumi domain guidance;
- OneBot adapter, observer, history-query, and routing code;
- empty public memory seeds and local configuration generators.

Excluded:

- QQ account IDs, group IDs, member names and member-specific preferences;
- relationship claims such as who likes, owns, dates, or is related to whom;
- message/session databases, raw group history, logs and test transcripts;
- received, downloaded, generated, or contextual images and their caches;
- API keys, OAuth sessions, `.env`, `auth.json`, tokens and passwords;
- NapCat/QQ login state and proprietary QQ binaries.

Runtime memories are copied into the user's Hermes data directory during initialization. They are
not symlinked back into the Git checkout, so ordinary bot use cannot accidentally stage them.
