# Gate9 TRFG vs TRBG Summary

- Average recall delta TRFG-source minus TRBG-source: `0.0074`.
- Average backend_fail delta TRFG-source minus TRBG-source: `5.7500`.
- Fail-only is compared under a pre-registered Gate9 alpha, so this is eligible for method replacement consideration.

## Required Answers

1. Fail-only is more compatible than full TRBG on 14-bus unnecessary burden, but it still does not pass the strict pre-registered 14-bus compatibility rule because B=1 unnecessary is slightly above the +5% limit.
2. Fail-only is not clearly better than full TRBG on case39 fresh: it has slightly higher average recall, but higher backend_fail than TRBG-source.
3. Gate8 suggested the full TRBG cost/time component can create unnecessary or recall damage in some diagnostics, but Gate9 shows removing cost/time is not enough to produce a replacement-quality method.
4. TRFG-source should not replace TRBG-source as the v2 main method under this Gate9 evidence.
5. TRFG-source is best treated as an appendix diagnostic or future-work candidate unless a new pre-registered validation fixes the fresh recall-retention and 14-bus compatibility failures.
