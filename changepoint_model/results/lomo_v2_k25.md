| scenario | model | spellF1 | macroF1 | poss pred/true | D F1 | team acc | transF1@25 | n |
|---|---|---|---|---|---|---|---|---|
| degraded | baseline50 | 0.3252 | 0.7273 | 568 / 378 | 0.742 | 0.812 | 0.202 | 3 |
| degraded | hmm_constant | 0.3252 | 0.7273 | 568 / 378 | 0.742 | 0.812 | 0.202 | 3 |
| degraded | hmm_boundary | 0.7359 | 0.8586 | 333 / 378 | 0.840 | 0.927 | 0.767 | 3 |
| clean | baseline50 | 0.4002 | 0.7669 | 502 / 378 | 0.754 | 0.860 | 0.228 | 3 |
| clean | hmm_constant | 0.4002 | 0.7669 | 502 / 378 | 0.754 | 0.860 | 0.228 | 3 |
| clean | hmm_boundary | 0.8064 | 0.8930 | 332 / 378 | 0.882 | 0.942 | 0.843 | 3 |

clean: boundary - constant = +0.4062 (3/0)

degraded: boundary - constant = +0.4106 (3/0)
