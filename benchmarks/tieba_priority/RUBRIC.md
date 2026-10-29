# Tieba hotspot priority benchmark rubric

Target decision: how much collection budget a Hot News Radar Tieba hotspot deserves **before** its full thread corpus is crawled.

Labels:
- 0 — skip: entertainment/sports/game/fandom/noise with little social research value
- 1 — light: some public-interest signal, but low expected first-person evidence or narrow relevance
- 2 — medium: meaningful ordinary-life/social issue with useful discussion potential
- 3 — deep: strong social relevance plus likely first-person experience, structural conflict, or unusually rich research value

Primary evaluation: Precision@10 for label >=2 and NDCG@10 using labels 0–3.
