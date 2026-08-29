# Changelog

All notable changes to the Ethicore Engine® Guardian SDK (community edition) are
documented here. This project follows [Semantic Versioning](https://semver.org).

## [2.8.0] — 2026-08-29

### ⚠️ Breaking

- **Community activation is now required.** Constructing a community-edition
  `Guardian()` without credentials now raises `CommunityActivationRequired` until
  the environment is activated with a free, verified-email community token.
  - **Who is affected:** users of the free community edition with no API key.
  - **Who is *not* affected:** anyone passing `api_key=...` or setting
    `ETHICORE_API_KEY` (paid API / self-hosted users) — the gate is skipped
    entirely for them.
  - **How to activate (one-time, free):** register your email at
    <https://portal.oraclestechnologies.com/community>, confirm the verification
    link, and drop the token you receive at `~/.ethicore/community_token` (or set
    `ETHICORE_COMMUNITY_TOKEN`).
  - **Why:** it lets us reach community users with security advisories and
    breaking-change notices, and keeps the free tier sustainable. The token is
    verified **entirely offline** — the SDK makes no network call to check it.

### Added

- `ethicore_guardian/activation.py` — dependency-free, offline Ed25519
  verification of the community token against a bundled public key. No new
  runtime dependencies; nothing leaves your machine.

### Changed

- Registered-mark refresh: `Ethicore Engine™` → `Ethicore Engine®` across the
  package and README.
- Version bumped to 2.8.0.

[2.8.0]: https://github.com/OraclesTech/guardian-sdk/releases/tag/v2.8.0
