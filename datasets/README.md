# Datasets

All files are JSONL: one JSON object per line. Upload them in the app with **New Project**; the column types are detected automatically.

| File | Rows | Primary key | Text column | Other columns |
|---|---|---|---|---|
| `hotels_100.jsonl` | 100 | `review_id` | `text` | `user_id`, `business_id`, `stars`, `useful`, `funny`, `cool`, `date` |
| `hotels_50k.jsonl` | 50,000 | `review_id` | `text` | same as above |
| `europarl_en_26k.jsonl` | 26,391 | `speech_id` | `text` | `speaker`, `date`, `chapter`, `source_file` |

`stars` is used as the rating column: it drives the star display and the sentiment check of each cluster. The EuroParl file has no rating column. It contains 147 repeated `speech_id` values, which the app drops during import (26,244 speeches remain).

## Hotel reviews

Source: [Yelp Open Dataset](https://www.yelp.com/dataset). The files contain the first reviews, in dataset order, of businesses tagged `Hotels` with at least 50 characters of text. `hotels_100.jsonl` holds the first 100 lines of `hotels_50k.jsonl`.

The Yelp data is subject to the Yelp Dataset Terms of Use, which permit academic use only and restrict redistribution. Obtain your own copy from Yelp if you need more data or a different category.

To rebuild the files from the original download, place `Yelp-JSON.zip` in `data/raw/` and run from the repository root:

```
python -m reviewscope_ml.data.sources.yelp --sample-size 50000
```

The output is written to `data/cache/sample_hotels_50k.jsonl`. `--category` selects another Yelp category.

## European Parliament speeches

Source: [Europarl parallel corpus v7](https://www.statmt.org/europarl/) (Koehn, 2005), proceedings of the European Parliament 1996 to 2011. The file contains speeches originally given in English, identified through the translation metadata of the other language versions; procedural speeches of the chair are removed. Reuse of European Parliament material is authorised provided the source is acknowledged.

To rebuild the file, place `europarl.tgz` from the source page in `data/raw/` and run:

```
python -m reviewscope_ml.data.sources.europarl --sample-size 26391
```

The output is written to `data/cache/sample_europarl-en_26391.jsonl`.
