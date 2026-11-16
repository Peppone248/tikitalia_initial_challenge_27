| scenario | model | spellF1 | macroF1 | poss pred/true | D F1 | team acc | transF1@25 | n |
|---|---|---|---|---|---|---|---|---|
| degraded | baseline50 | 0.3186 | 0.7272 | 563 / 378 | 0.739 | 0.815 | 0.193 | 3 |
| degraded | hmm_constant | 0.3186 | 0.7272 | 563 / 378 | 0.739 | 0.815 | 0.193 | 3 |
| degraded | hmm_boundary | 0.7482 | 0.8736 | 332 / 378 | 0.861 | 0.932 | 0.772 | 3 |
| harsh | baseline50 | 0.2478 | 0.6596 | 588 / 378 | 0.674 | 0.786 | 0.157 | 3 |
| harsh | hmm_constant | 0.2478 | 0.6596 | 588 / 378 | 0.674 | 0.786 | 0.157 | 3 |
| harsh | hmm_boundary | 0.6760 | 0.8363 | 329 / 378 | 0.814 | 0.917 | 0.681 | 3 |

degraded: boundary - constant = +0.4296 (3/0)

harsh: boundary - constant = +0.4283 (3/0)
