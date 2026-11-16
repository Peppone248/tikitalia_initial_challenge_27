| scenario | model | spellF1 | macroF1 | poss pred/true | D F1 | team acc | transF1@25 | n |
|---|---|---|---|---|---|---|---|---|
| degraded | baseline50 | 0.3252 | 0.7273 | 568 / 378 | 0.742 | 0.812 | 0.202 | 3 |
| degraded | hmm_constant | 0.7148 | 0.8575 | 326 / 378 | 0.837 | 0.928 | 0.736 | 3 |
| degraded | hmm_boundary | 0.7296 | 0.8576 | 325 / 378 | 0.835 | 0.931 | 0.756 | 3 |
| clean | baseline50 | 0.4002 | 0.7669 | 502 / 378 | 0.754 | 0.860 | 0.228 | 3 |
| clean | hmm_constant | 0.7942 | 0.8902 | 329 / 378 | 0.879 | 0.940 | 0.824 | 3 |
| clean | hmm_boundary | 0.8028 | 0.8917 | 331 / 378 | 0.881 | 0.942 | 0.836 | 3 |

clean: boundary - constant = +0.0087 (3/0)

degraded: boundary - constant = +0.0148 (3/0)
