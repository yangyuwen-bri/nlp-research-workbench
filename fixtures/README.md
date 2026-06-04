# Public Demo Datasets

这些文件是“公开来源的采样 demo”，不是完整 benchmark，也不是权威评测训练集缓存。
完整公开数据请看 [bench_data/README.md](/Users/gsdata/work/nlp_tool/bench_data/README.md)。

- [public_online_shopping_demo.csv](/Users/gsdata/work/nlp_tool/fixtures/public_online_shopping_demo.csv)
  Source: ChineseNlpCorpus `online_shopping_10_cats`
- [public_waimai_demo.csv](/Users/gsdata/work/nlp_tool/fixtures/public_waimai_demo.csv)
  Source: ChineseNlpCorpus `waimai_10k`
- [public_weibo_senti_demo.csv](/Users/gsdata/work/nlp_tool/fixtures/public_weibo_senti_demo.csv)
  Source: dirtycomputer `weibo_senti_100k`
- [public_chnsenticorp_htl_demo.csv](/Users/gsdata/work/nlp_tool/fixtures/public_chnsenticorp_htl_demo.csv)
  Source: dirtycomputer `ChnSentiCorp_htl_all`
- [public_rumor_opinion_demo.csv](/Users/gsdata/work/nlp_tool/fixtures/public_rumor_opinion_demo.csv)
  Source: THUNLP `Chinese_Rumor_Dataset`
- Source metadata: [public_datasets.json](/Users/gsdata/work/nlp_tool/fixtures/public_datasets.json)

These files are sampled public subsets intended for realistic demo flows:

- `public_online_shopping_demo.csv`: 400 rows across 5 product categories, balanced positive and negative labels.
- `public_waimai_demo.csv`: 240 rows of delivery reviews, balanced positive and negative labels.
- `public_weibo_senti_demo.csv`: balanced Weibo sentiment subset for social public-opinion analysis.
- `public_chnsenticorp_htl_demo.csv`: balanced hotel-review sentiment subset for industry-style sentiment evaluation.
- `public_rumor_opinion_demo.csv`: rumor and public-opinion narrative corpus for topic, event, and reporting demos.

To regenerate them from public upstream sources:

```bash
cd /Users/gsdata/work/nlp_tool
python3 scripts/build_public_demos.py
```

To run the unified benchmark entry:

```bash
cd /Users/gsdata/work/nlp_tool
python3 scripts/run_public_benchmarks.py --rebuild
```
