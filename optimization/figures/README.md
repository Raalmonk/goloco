# Reproduce the scientific figures

These five figures and their tables use only the committed sanitized timing records. They do not import the inference package, download models, load pickles or numeric bundles, compile Rust, or contact Colab/GitHub. Historical model-dependent validation is reported separately from these publication checks.

Use Python 3.12 and a separate environment. From the repository root:

```sh
python3.12 -m venv .figure-venv
.figure-venv/bin/python -m pip install -r optimization/environments/figures.lock.txt
.figure-venv/bin/python optimization/figures/generate_figures.py
```

After that one generation command, SVG, vector PDF, 300-dpi PNG, captions, exact figure provenance, and all result tables are regenerated in this directory. The dependency lock contains the versions used for the publication renders. It is independent of the legacy inference lock. SVG text remains editable; ordinary DejaVu Sans text is used without distributing font files. Fixed SVG hash salt and omitted wall-clock image metadata make renders deterministic in the pinned environment. Rendering on another platform can still change font-engine bytes; plotted data and generated tables remain fully specified.

To check the evidence without plotting dependencies:

```sh
python3 optimization/scripts/check_evidence.py
```

To regenerate somewhere else, use `generate_figures.py --out /tmp/goloco-figures`. The input paths resolve relative to the script, so the current working directory does not change the data source.

| Figure | Purpose | Downloads |
|---|---|---|
| [Matched residency](matched_residency.png) | Five paired B3/C0 measurements; 6.85× is the reproduced comparison | [SVG](matched_residency.svg) · [PDF](matched_residency.pdf) |
| [Reusable resident performance](reusable_resident.png) | Four packaged variants, five resident requests each | [SVG](reusable_resident.svg) · [PDF](reusable_resident.pdf) |
| [First output](startup_latency.png) | Three fresh-process trials per packaged variant | [SVG](startup_latency.svg) · [PDF](startup_latency.pdf) |
| [Standalone memory](standalone_memory.png) | Maximum of five measured resident RSS peaks per variant | [SVG](standalone_memory.svg) · [PDF](standalone_memory.pdf) |
| [Negative chunking result](negative_chunking.png) | Chunks 1/64/256; fewer FFI calls were slower | [SVG](negative_chunking.svg) · [PDF](negative_chunking.pdf) |

Read the [complete captions and alt text](CAPTIONS.md), [per-figure provenance](provenance.json), [data export map](../benchmarks/PROVENANCE.md), and [generated numeric tables](source_tables/summary.md). The observations table retains all 92 records, including original workload oracles, pilots and historical C0 startup, even when they are not part of the five headline figures.

The five numerical charts were individually visually inspected for clipping, overlaps, glyphs, labels and values during publication. This is a rendering check, not a new inference test.
