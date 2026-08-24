# The submission report

`report.pdf` is the formatted report: 54 pages, containing a title page,
abstract, table of contents, list of tables, list of figures,
nomenclature, six chapters and a bibliography of 43 verified sources.

## Building it

```sh
python report/make_assets.py    # tables and figures, from the data
sh report/build.sh              # three LaTeX passes plus BibTeX
```

A TeX distribution is required; the report was built with TeX Live 2026
Basic. `make_assets.py` reads only the committed data tables, so the
report can be rebuilt without the raw archives.

## Structure

| File | Contents |
|---|---|
| `report.tex` | The document; includes the files below |
| `preamble.tex` | Class, packages, page style and spacing |
| `front.tex` | Title page, abstract, contents, lists, nomenclature |
| `ch1_intro.tex` | Introduction: problem, questions, scope, contribution |
| `ch2_literature.tex` | Literature review and the gap |
| `ch3_method.tex` | Corpus, labelling procedure, parser, generator, score |
| `ch4_models.tex` | The four learned components and the fairness controls |
| `ch5_results.tex` | Coverage, the judges, the variants, sensitivity analyses |
| `ch6_conclusion.tex` | Discussion, limitations, conclusions, further work |
| `references.bib` | Bibliography, copied from `01_literature_review/` |
| `tables/` | Generated LaTeX tables |
| `figures/` | Generated figures, without in-figure titles |

## Why the assets are generated

Every table and figure in the report is produced from the committed CSV
and JSON results by `make_assets.py`. Nothing is transcribed by hand, so
the report cannot drift out of step with the data, and rerunning any
experiment then rebuilding the report propagates the new numbers
automatically.

The report's figures differ from those in `figures/` in one respect: the
in-figure titles are removed, because in a formal report the caption
carries the explanation.
