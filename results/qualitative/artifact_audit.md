# Artifact audit — Whisper stock-phrase contamination of the FMR set (TORGO)

Read-only check; no existing results file was modified. Artifact classification is derived from the data (frequency + cross-reference diversity + low semantic similarity), not a hardcoded phrase list — see rule in the script header (MIN_COUNT=3, DIVERSITY_THRESH=0.8, S_THRESH=0.3).

## Step 1 — Whisper FMR-positive impaired-sentence rows

n = 127 (expected 127 from the prior qualitative-table run).

Hypothesis strings occurring more than once within the 127 (exact/near-exact, case+punctuation-insensitive):

| hypothesis (as first seen) | count |
|---|---|
| Thank you. | 2 |

## Step 2 — 40 most frequent hypothesis strings, whole Whisper impaired set

(All prompt types, is_control=False; n=2883 rows.)

| rank | hypothesis (as first seen) | count | n_distinct_refs | mean s |
|---|---|---|---|---|
| 1 | No. | 64 | 14 | 0.391 |
| 2 | (empty) | 39 | 37 | 0.003 |
| 3 | Bye. | 33 | 26 | 0.207 |
| 4 | Right. | 32 | 15 | 0.338 |
| 5 | Yeah. | 22 | 18 | 0.171 |
| 6 | Yes. | 21 | 2 | 0.815 |
| 7 | Thank you. | 18 | 18 | 0.115 |
| 8 | Him. | 16 | 3 | 0.738 |
| 9 | but | 15 | 11 | 0.178 |
| 10 | Go. | 15 | 5 | 0.499 |
| 11 | Rock. | 14 | 5 | 0.758 |
| 12 | Sigh. | 14 | 3 | 0.832 |
| 13 | One. | 14 | 4 | 0.569 |
| 14 | Bad. | 12 | 6 | 0.570 |
| 15 | Wait. | 11 | 7 | 0.216 |
| 16 | here | 11 | 5 | 0.503 |
| 17 | Shoot. | 11 | 2 | 0.680 |
| 18 | Peace. | 10 | 8 | 0.242 |
| 19 | So. | 10 | 6 | 0.187 |
| 20 | Not. | 9 | 1 | 0.097 |
| 21 | Leak. | 9 | 1 | 0.928 |
| 22 | Oh. | 9 | 7 | 0.336 |
| 23 | Slip. | 9 | 2 | 0.868 |
| 24 | Air. | 8 | 2 | 0.840 |
| 25 | Spark. | 8 | 1 | 0.917 |
| 26 | Four. | 8 | 4 | 0.531 |
| 27 | Up. | 8 | 1 | 0.922 |
| 28 | Okay. | 8 | 7 | 0.146 |
| 29 | feet | 8 | 3 | 0.706 |
| 30 | Bit. | 8 | 2 | 0.779 |
| 31 | Tear. | 8 | 2 | 0.851 |
| 32 | Um... | 8 | 5 | 0.328 |
| 33 | Cheer. | 7 | 3 | 0.421 |
| 34 | side | 7 | 1 | 0.987 |
| 35 | Sip. | 7 | 1 | 0.958 |
| 36 | which | 7 | 2 | 0.269 |
| 37 | Mike. | 7 | 3 | 0.756 |
| 38 | park. | 7 | 1 | 0.969 |
| 39 | See you. | 7 | 5 | 0.193 |
| 40 | and | 7 | 2 | 0.854 |

## Step 3 — data-derived artifact classification and FMR recomputation

Artifact list is derived from the WHOLE impaired set (word+sentence, Step 2's population) per system, then applied as an exclusion filter to that system's sentence-only FMR-relevant population.

### whisper_large_v3

Distinct hypothesis strings in the WHOLE impaired set (n=2883, word+sentence): 1474. Flagged as artifact: 27.

| hypothesis | count | n_distinct_refs | diversity | mean s |
|---|---|---|---|---|
| Yeah. | 22 | 18 | 0.818 | 0.171 |
| Thank you. | 18 | 18 | 1.0 | 0.115 |
| Peace. | 10 | 8 | 0.8 | 0.242 |
| Okay. | 8 | 7 | 0.875 | 0.146 |
| Good. | 6 | 5 | 0.833 | 0.235 |
| boom | 6 | 6 | 1.0 | 0.237 |
| Sweet. | 5 | 5 | 1.0 | 0.157 |
| Me. | 5 | 4 | 0.8 | 0.239 |
| Thanks for watching! | 5 | 5 | 1.0 | 0.112 |
| Bleh. | 4 | 4 | 1.0 | 0.292 |
| Whoa. | 4 | 4 | 1.0 | 0.203 |
| CUT | 4 | 4 | 1.0 | 0.274 |
| God. | 4 | 4 | 1.0 | 0.209 |
| Hey. | 4 | 4 | 1.0 | 0.195 |
| Buh-bye. | 3 | 3 | 1.0 | 0.2 |
| Boing. | 3 | 3 | 1.0 | 0.274 |
| Like... | 3 | 3 | 1.0 | 0.103 |
| Fuck. | 3 | 3 | 1.0 | 0.186 |
| there | 3 | 3 | 1.0 | 0.217 |
| Hi. | 3 | 3 | 1.0 | 0.206 |
| Ciao. | 3 | 3 | 1.0 | 0.183 |
| Mew. | 3 | 3 | 1.0 | 0.261 |
| Uh... | 3 | 3 | 1.0 | 0.117 |
| I don't know. | 3 | 3 | 1.0 | 0.15 |
| Dad. | 3 | 3 | 1.0 | 0.29 |
| Weak. | 3 | 3 | 1.0 | 0.276 |
| we | 3 | 3 | 1.0 | 0.279 |

FMR-positive rows for whisper_large_v3: 127. Of these, 2 (1.6%) have an artifact-flagged hypothesis.

| | n (impaired sentences) | FMR | median fluency-given-wrong |
|---|---|---|---|
| before exclusion | 692 | 0.1835 | 0.6324 |
| after excluding artifact-hypothesis rows | 690 | 0.1812 | 0.6321 |

_Published fluency-given-wrong for whisper_large_v3: 0.632._

### wav2vec2_960h

Distinct hypothesis strings in the WHOLE impaired set (n=2883, word+sentence): 1745. Flagged as artifact: 23.

| hypothesis | count | n_distinct_refs | diversity | mean s |
|---|---|---|---|---|
| MA | 8 | 7 | 0.875 | 0.253 |
| S | 8 | 8 | 1.0 | 0.282 |
| SE | 7 | 6 | 0.857 | 0.287 |
| IT | 6 | 6 | 1.0 | 0.297 |
| TE | 5 | 5 | 1.0 | 0.288 |
| THERE | 5 | 4 | 0.8 | 0.264 |
| WAT | 5 | 5 | 1.0 | 0.296 |
| SO | 5 | 4 | 0.8 | 0.255 |
| DON'T | 5 | 4 | 0.8 | 0.252 |
| DE | 5 | 4 | 0.8 | 0.29 |
| RA | 4 | 4 | 1.0 | 0.259 |
| TEN | 4 | 4 | 1.0 | 0.246 |
| LO | 4 | 4 | 1.0 | 0.287 |
| BOT | 3 | 3 | 1.0 | 0.262 |
| MIT | 3 | 3 | 1.0 | 0.149 |
| CO | 3 | 3 | 1.0 | 0.296 |
| MET | 3 | 3 | 1.0 | 0.266 |
| CA | 3 | 3 | 1.0 | 0.269 |
| MER | 3 | 3 | 1.0 | 0.29 |
| P | 3 | 3 | 1.0 | 0.253 |
| POEM | 3 | 3 | 1.0 | 0.26 |
| RAT | 3 | 3 | 1.0 | 0.226 |
| TAT | 3 | 3 | 1.0 | 0.266 |

FMR-positive rows for wav2vec2_960h: 181. Of these, 0 (0.0%) have an artifact-flagged hypothesis.

| | n (impaired sentences) | FMR | median fluency-given-wrong |
|---|---|---|---|
| before exclusion | 692 | 0.2616 | 0.5357 |
| after excluding artifact-hypothesis rows | 692 | 0.2616 | 0.5357 |

_Published fluency-given-wrong for wav2vec2_960h: 0.536._

### parakeet_tdt

Distinct hypothesis strings in the WHOLE impaired set (n=2883, word+sentence): 1412. Flagged as artifact: 17.

| hypothesis | count | n_distinct_refs | diversity | mean s |
|---|---|---|---|---|
| So | 13 | 11 | 0.846 | 0.28 |
| Sure. | 8 | 7 | 0.875 | 0.164 |
| Wait. | 8 | 8 | 1.0 | 0.23 |
| See | 8 | 7 | 0.875 | 0.275 |
| Okay. | 8 | 8 | 1.0 | 0.153 |
| M | 8 | 8 | 1.0 | 0.223 |
| Mm. | 6 | 6 | 1.0 | 0.205 |
| Shit | 6 | 5 | 0.833 | 0.18 |
| Hello. | 5 | 4 | 0.8 | 0.231 |
| Great. | 5 | 5 | 1.0 | 0.184 |
| Mm-hmm. | 4 | 4 | 1.0 | 0.192 |
| Where? | 3 | 3 | 1.0 | 0.239 |
| Really? | 3 | 3 | 1.0 | 0.144 |
| Mark. | 3 | 3 | 1.0 | 0.252 |
| I | 3 | 3 | 1.0 | 0.262 |
| Hi. | 3 | 3 | 1.0 | 0.22 |
| Fine. | 3 | 3 | 1.0 | 0.224 |

FMR-positive rows for parakeet_tdt: 100. Of these, 0 (0.0%) have an artifact-flagged hypothesis.

| | n (impaired sentences) | FMR | median fluency-given-wrong |
|---|---|---|---|
| before exclusion | 692 | 0.1445 | 0.5223 |
| after excluding artifact-hypothesis rows | 691 | 0.1447 | 0.5230 |

_Published fluency-given-wrong for parakeet_tdt: 0.522._

## Step 4 — does excluding artifacts change the published ranking?

| system | fluency-given-wrong BEFORE | fluency-given-wrong AFTER | rank BEFORE | rank AFTER |
|---|---|---|---|---|
| whisper_large_v3 | 0.6324 | 0.6321 | #1 | #1 |
| wav2vec2_960h | 0.5357 | 0.5357 | #2 | #2 |
| parakeet_tdt | 0.5223 | 0.5230 | #3 | #3 |

**Ranking changed: False.**

Whisper artifact share of its own FMR-positive set: 1.6% (2/127 rows).