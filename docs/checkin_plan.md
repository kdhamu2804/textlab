# TextLab: Check-in Plan (Nov 24)

**Project:** Mental Health Detection from Text: A Web App That Finds Stress in Reddit Posts
**Team:** Dhamodhar Konduru · Kushvanth Thati · Uday Kiran Thurumella

## Done

| Item | Result |
|---|---|
| Data | Dreaddit: 2,838 train / 715 test posts, 5 topics |
| Baselines | Majority F1 0.681 · TF-IDF + LR F1 0.753 |
| Main model | RoBERTa F1 **0.834** (+8.1 points over TF-IDF; p < 0.001) |
| Success criterion (+3 F1) | **Met** |
| New-topic test (train on 4 topics, test on the 5th) | RoBERTa 0.71–0.86 on unseen topics, above TF-IDF on seen ones |
| Extra analysis | Confusion matrices, calibration (ECE 0.112 → 0.073), explanations |
| Web app | Working: Analyze, Results, About, labelling page |
| App checks | Same answer as the model on 715/715 posts; about 84 ms per answer (goal < 1 s) |

## Left

| Task | Who | By |
|---|---|---|
| 50-post label check (each person) | All three | Nov 26 |
| Review the 50-error grouping | Kushvanth | Nov 28 |
| Fix the live website (Render deploy) | Uday | Nov 28 |
| Presentation slides and demo rehearsal | All three | Dec 1 |
| Final report (each person writes their own part) | All three | Dec 11 |

## Cut-list (if time runs short, cut in this order)

1. **Live website:** demo from a laptop instead
2. **Full 50-error review:** review 25 errors instead of 50
3. **Label-agreement statistics between teammates:** report only agreement with the dataset labels
4. **Never cut:** test-set F1 comparison, new-topic test, honest limitations

## Risks

- **Noisy labels:** the label check measures this. More than 10 disagreements out of 50 → we report it.
- **Model learns topic words instead of stress:** the new-topic test shows only a small drop, so this risk is low.

**Presentation preference:** Dec ___ (fill in: Dec 3 or Dec 8)
