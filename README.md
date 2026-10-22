# pychromy

[![PyPI](https://img.shields.io/pypi/v/pychromy.svg)](https://pypi.org/project/pychromy/)
[![license: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

Chromy's method of sequential random sampling with probability minimum
replacement, from Python. The procedure selects units one at a time down an
ordered list, each with probability proportional to its size, so that every
unit's realised number of selections differs from its expected number by
less than one — and the sample size is always exact. It is how many large
establishment and household surveys draw their samples, and how
`PROC SURVEYSELECT METHOD=PPS_SEQ` works; until now it lived only inside
proprietary software and one Stata module.

The engine is the dependency-free Rust crate
[`chromy-core`](https://github.com/castlaboratory/chromy-core), shared with
the R package [`chromy`](https://github.com/castlaboratory/chromy). This
module needs nothing besides Python: matrices are lists of columns, so a
pandas user writes `[df[c].tolist() for c in cols]` and a plain-list user
writes the lists directly.

## Install

```sh
pip install pychromy
```

## A design in a few lines

```python
import pychromy

# A frame of 300 schools: a region, a size, an outcome to estimate.
import random
rng = random.Random(2026)
region = [rng.choice(["north", "east", "south", "west"]) for _ in range(300)]
pupils = [round(rng.lognormvariate(4.5, 0.9)) for _ in range(300)]
outcome = [rng.lognormvariate(5.0, 0.7) for _ in range(300)]

# Expected hits from the size; serpentine order within region by size;
# the selection, reproducible from its seed.
s = pychromy.sample(
    size=pupils, n=40,
    keys=[region, [float(p) for p in pupils]],
    seed=20261022,
)
sum(s["hits"])          # 40, always: the sample size is exact

# The estimator that matches the design, with its standard error.
y = [[outcome[u] for u in s["unit"]]]
est = pychromy.estimate_total(
    y, s["position"], s["expected_hits"], s["hits"]
)
est["total"], est["se"]
```

## What it does

- **`sample`** — a whole design in one call: size to expected hits, keys to
  serpentine order, head, selection, weights.
- **`pmr`** — the sequential selection itself, deterministic given the
  uniforms, with an explicit account of every floating-point adjustment.
- **`serpentine_order`** — hierarchical serpentine (or nested) sorting:
  implicit stratification without seams; string keys rank by byte order,
  never by locale.
- **`mpps_size`** and **`eqcv_allocation`** — a maximal size measure over
  several auxiliary variables, and the allocation that equalises their
  anticipated coefficients of variation.
- **`expected_hits`** — sizes into expected hits, certainties resolved by
  iteration, or units allowed several hits under minimum replacement.
- **`select`** — the full procedure with every draw under your control,
  for callers that bring their own generator.
- **`estimate_total`** — the multiplicity estimator with the
  successive-difference variance.

## Randomness, stated plainly

The engine takes its randomness as arguments. `sample(seed=...)` draws from
a fixed, documented generator (SplitMix64), so a published design can name
its seed and be replayed. R and Python reproduce themselves, not each
other: each language consumes its own generator, but both run the same
engine, pinned bit for bit to the same reference implementation by the
golden tests in `chromy-core`.

## References

- Chromy, J. R. (1979). Sequential sample selection methods. *Proceedings
  of the American Statistical Association, Survey Research Methods
  Section*, 401–406.
- Chauvet, G. and Le Gleut, R. (2021). Properties of Chromy's sampling
  procedure. *Survey Methodology*, 47(2), 321–347.
- Folsom, R. E., Potter, F. J. and Williams, S. R. (1987). Notes on a
  composite size measure for self-weighting samples in multiple domains.
  *Proceedings of the ASA, Section on Survey Research Methods*, 792–796.

## License

GPL-3.0-or-later. A project of the [CAST Lab](https://castlab.org), UFPE.
