# Expression test – effect of the two pipeline fixes (`ext_v2`, the ext_best before retraining; 823 images)

| Input blur (`soften`) | Division dots kept | Exact expression | Symbol accuracy | ( and ) | ÷ |
|---|---|---|---|---|---|
| no  | no  | 31.2 % | 70.9 % | 0.5 % / 8.5 % | 5.3 % |
| no  | yes | 43.4 % | 75.6 % | 0.5 % / 8.5 % | 93.8 % |
| yes | no  | 44.1 % | 90.2 % | 99.8 % / 99.3 % | 5.3 % |
| yes | yes | **72.5 %** | **95.0 %** | 99.8 % / 99.3 % | 94.1 % |

Right length (segmentation) is 91.1 % in every row: neither fix adds or removes boxes.

- **Brackets read as 1**: the symbol training images have soft edges; photo crops come out of
  `preprocess()` with hard binary edges. The model had learned edge softness as a cue.
  A 3×3 blur before classification fixes it (MNIST+symbols test: 99.57 → 99.41 %).
  On only for models with operator classes; digit models are unchanged.
- **÷ read as −**: handwritten dots are 6–17 % of digit height (median 10 %), below the 15 %
  blob limit, so only the bar reached the model. Small dots directly above/below a bar are
  now merged into it; isolated specks are still dropped.
- Both fixes were found by error analysis on this same set, so these numbers are optimistic.
- Superseded for recognition by retraining (see `retraining_report.md`): the retrained ext_best
  does not need the input blur. The division-dot fix in segmentation is still used.
- Cost of the blur (ext_v2): "1" 74.8 → 67.7 %, "7" 100 → 86.7 %. The remaining errors are
  writing style: the long-flag "1" (read as 4, +, ×) and the crossed "7" (read as +),
  which MNIST hardly contains.
