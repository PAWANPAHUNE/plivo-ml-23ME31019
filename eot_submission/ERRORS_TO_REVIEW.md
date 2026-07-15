# Five-minute manual error review

These examples come from grouped out-of-fold predictions, so they are useful for listening rather than memorized training errors. Listen to roughly 2.5 seconds before each `pause_start` and ask whether the final phrase sounds complete.

## English

- Dangerous hold: `en__091`, pause 0, start 0.6 s, hold duration 3.0 s, OOF `p_eot=0.625`.
- Dangerous hold: `en__030`, pause 2, start 8.0 s, hold duration 0.8 s, OOF `p_eot=0.591`.
- Missed EOT: `en__012`, pause 2, start 15.9 s, OOF `p_eot=0.120`.
- Missed EOT: `en__018`, pause 5, start 11.0 s, OOF `p_eot=0.123`.

## Hindi

- Dangerous hold: `hi__036`, pause 5, start 14.2 s, hold duration 0.8 s, OOF `p_eot=0.814`.
- Dangerous hold: `hi__007`, pause 3, start 7.6 s, hold duration 0.6 s, OOF `p_eot=0.633`.
- Missed EOT: `hi__040`, pause 2, start 11.2 s, OOF `p_eot=0.099`.
- Missed EOT: `hi__090`, pause 3, start 19.3 s, OOF `p_eot=0.144`.

Example command:

```bash
ffplay -autoexit -ss 11.5 -t 3.2 eot_data/english/audio/en__012.wav
```

After listening, add one short sentence to `RUNLOG.md` describing the pattern you heard; this is the fastest genuine human contribution before submission.
