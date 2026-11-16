| scenario | model | spellF1 | macroF1 | poss pred/true | D F1 | team acc | transF1@25 | n |
|---|---|---|---|---|---|---|---|---|
| degraded | baseline50 | 0.3252 | 0.7273 | 568 / 378 | 0.742 | 0.812 | 0.202 | 3 |
| degraded | hmm_constant | 0.3252 | 0.7273 | 568 / 378 | 0.742 | 0.812 | 0.202 | 3 |
| degraded | hmm_boundary | 0.7084 | 0.8571 | 320 / 378 | 0.832 | 0.941 | 0.729 | 3 |
| clean | baseline50 | 0.4002 | 0.7669 | 502 / 378 | 0.754 | 0.860 | 0.228 | 3 |
| clean | hmm_constant | 0.4002 | 0.7669 | 502 / 378 | 0.754 | 0.860 | 0.228 | 3 |
| clean | hmm_boundary | 0.8005 | 0.8839 | 330 / 378 | 0.858 | 0.952 | 0.833 | 3 |

clean: boundary - constant = +0.4003 (3/0)

degraded: boundary - constant = +0.3832 (3/0)
