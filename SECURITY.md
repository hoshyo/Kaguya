# Security

- Keep the adapter, Hermes plugin socket, and Web UI on loopback unless you operate a trusted
  reverse proxy with authentication.
- `kaguya init` generates fresh local tokens. Examples never contain working credentials.
- External Codex/Claude logins are not adopted by default. Add a provider explicitly with
  `hermes auth add ...` or run `hermes setup`.
- Automation API and direct messages start disabled.
- Group access starts deny-by-default until `kaguya bind` is run.
- Before publishing changes, run `./kaguya privacy-check` and inspect the staged diff.

Report vulnerabilities privately to the repository owner rather than opening a public issue with
secrets or live identifiers.
