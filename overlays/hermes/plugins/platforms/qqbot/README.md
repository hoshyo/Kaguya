# QQ full-group auto replies

QQ must first enable **receive all messages** for the group. This is a QQ-side
setting, not a Hermes permission. Its `GROUP_MESSAGE_CREATE` event shares the
already subscribed `GROUP_AND_C2C_EVENT` intent with mention-only messages:
[official event contract](https://bot.q.qq.com/wiki/develop/api-v2/autogen/event/group_message_create.html).

Opt in individual groups under the **receiving bot's** config, including when
`gateway.profile_routes` sends the conversation to another profile:

```yaml
platforms:
  qqbot:
    extra:
      group_auto_reply:
        "YOUR_GROUP_OPENID":
          enabled: true
          names: [Kaguya, 辉夜]
          probability: 0.12
          cooldown_seconds: 90
          window_seconds: 600
          max_replies: 4
```

Groups absent from this mapping stay mention-only. Existing group access policy
and command authorization still apply; the mapping does not grant access.
Invalid rules are logged and disabled. Limits must be finite and positive;
probability is between zero and one. All fields except `enabled` and `names`
default to the values above; an omitted name list has no name-based triggers.

- Explicit bot mentions and slash commands retain their normal processing and
  are not restricted by the passive-chat budget.
- Names bypass random sampling, **not** cooldown or the rolling budget. Other
  messages need at least six characters and a letter, and are sampled locally.
  Messages mentioning other users are not sampled. Media-only and noise messages
  do not start unsolicited turns.
- At most four unsolicited **turn admissions**, not outbound chunks, are allowed
  in a rolling ten-minute window. Failed/silent admissions still spend budget.
  Admission and successful bot text sends start the cooldown. Counters are local
  to each group/adapter and reset on gateway restart.
- Rejected messages are not downloaded, queued, or sent to a model. Busy sessions
  drop admitted passive inputs instead of interrupting the current turn.
- Passive events have `reply_expected=False` and a `channel_context` label of
  `QQ 普通群消息（未@机器人）`. A persona may return only `NO_REPLY` when there is
  nothing to add; the existing gateway suppresses that marker. Passive text
  cannot invoke gateway control commands. Explicit commands use the existing
  authorized command path instead.

This is adapter code, so deploying it requires restarting the host gateway, not
just hot-reloading a routed profile. No transcript or memory reset is needed.
The first observed full-group event is logged as `Full-group messages observed`;
that receipt distinguishes server readiness from QQ-side enablement.
