| scenario | model | spellF1 | macroF1 | poss pred/true | D F1 | team acc | transF1@25 | n |
|---|---|---|---|---|---|---|---|---|
| degraded | baseline50 | 0.3252 | 0.7273 | 568 / 378 | 0.742 | 0.812 | 0.202 | 3 |
| degraded | hmm_constant | 0.3252 | 0.7273 | 568 / 378 | 0.742 | 0.812 | 0.202 | 3 |
| degraded | hmm_boundary | 0.7430 | 0.8574 | 335 / 378 | 0.839 | 0.926 | 0.769 | 3 |
| clean | baseline50 | 0.4002 | 0.7669 | 502 / 378 | 0.754 | 0.860 | 0.228 | 3 |
| clean | hmm_constant | 0.4002 | 0.7669 | 502 / 378 | 0.754 | 0.860 | 0.228 | 3 |
| clean | hmm_boundary | 0.8076 | 0.8921 | 347 / 378 | 0.883 | 0.940 | 0.852 | 3 |

clean: boundary - constant = +0.4074 (3/0)

degraded: boundary - constant = +0.4178 (3/0)
