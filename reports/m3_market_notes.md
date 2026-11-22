# M3 — Swissgrid mFRR auction mechanics, bidder-data limitations and hypotheses

Owner: Iustin (M3, learning bidders) · Reviewer: Carolina · Sprint 1 · 2026-11-22

Builds on the Sprint 0 research note (research questions, working hypotheses, assumptions) and on the
evidence in `notebooks/m3_eda_bidbook.ipynb`. Purpose: know the rules the bidders play by, explain what
the data shows, and turn the research questions into hypotheses we can test in Week 2.

**Sources**
- [TRL] Swissgrid, *Anhang: Ausschreibungsbedingungen Tertiärregelung*, version 06.05.2026, §2.2.1, §3.1, §3.2.
- [P20] Swissgrid, *Principles of ancillary services products*, V20, valid from 01.11.2024, §3.3–3.4.
- [P21] Same document, V21 (German), valid from 06.05.2026.
- Project brief (Carina Power, Nov 2026).

---

## 1. Auction mechanics (mFRR capacity = TRL)

| Topic | Rule | Source |
|---|---|---|
| Products | Separate up (TRL+) and down (TRL−) capacity bands. Weekly (Mon 00:00 – Sun 24:00) and daily (six 4-h blocks). | [TRL] §2.2.1 |
| Gate closure | Per the tender calendar. In our data: daily Mon→Fri 14:30, Tue→Fri 15:30, Wed–Sat→2 days before 14:30, Sun→Thu 15:30; weekly→Tue 13:00 the week before. Never shifted for holidays (2019–2026). | brief; `tests/test_data.py` |
| Bid format | An **offer** = one or more (volume, price) combinations ("Stufenangebot"), first slice ≥ 5 MW, then ±1 MW steps, max 100 MW. One offer may mix up and down combinations. Unlimited offers per provider. | [TRL] §3.1(a)(b)(d) |
| Divisibility | **Indivisible**: Swissgrid awards only a combination explicitly in the offer, and **at most one combination per offer** ("höchstens eine der darin enthaltenen Kombinationen"). | [TRL] §3.1(d) |
| Clearing | **Not a merit order.** Minimise the *total* holding cost of tertiary **and secondary** (aFRR) control power, subject to minimum required volumes. Ties go to the earliest submitted offer. Swissgrid may procure less than tendered if the last offer would over-procure. | [TRL] §3.1(f) |
| Demand | No fixed quantity. Joint **stochastic optimisation of SRL + TRL** against a deficit probability, recalibrated **once a year** on the previous year's imbalances; "quantities range between SRL± and TRL± depending on the prices". Reference volumes ≈ 480 MW TRL+ / 508 MW TRL−. | [P20] §3.3.1, §3.4.1 |
| Weekly vs daily split | Based on the **bid prices in the weekly tender and the expected prices of the daily tenders**. | [P20] §3.4.1 |
| Second daily tender | If the first daily tender leaves a deficit, a second one runs; first-round offers are frozen and new offers can be added; the award is over both rounds. If still short, Swissgrid may reduce demand or shift volume between products. | [TRL] §3.1(i), §3.2 |
| Settlement | **Pay-as-bid** for capacity (CHF/MW for the whole period). Network fees are inside the bid. | [TRL] §3.1(h), §2.2.1 |
| Obligation | Awarded capacity must be offered as energy (TRE_mFRR, activated via MARI or nationally). | [TRL] §3.1(j)(k) |
| Information to bidders | The result is available to providers **anonymised at gate closure**; a losing bidder's capacity is bound until informed, at most 12 h after closure. Swissgrid publishes tendered and awarded quantities. | [TRL] §3.1(e)(g), §2.2.1 |
| Change in V21 (06.05.2026) | **Secondary control (SRL/aFRR) is now tendered weekly *and* daily** (V20: weekly only). Tertiary stays weekly + daily. | [P21] §3.1 vs [P20] §3.1 |

## 2. What the rules explain in our data

| EDA observation | Explanation from the rules | Confidence |
|---|---|---|
| ~46 % of two-sided auctions reject a bid **cheaper** than the marginal price; only 25 % are explained by "bid larger than the remaining need". | Indivisible offers + at most one combination per offer + joint cost minimisation with aFRR. A cheaper combination is "rejected" when another combination of the same offer was chosen, or when a cheaper total cost is reached with a different set of offers. | High (rules are explicit) |
| ~30 k bids **partially awarded** (offered > awarded), 54 % of them not at the marginal price. | Most likely the row is an offer and the award is one of its smaller step combinations. It cannot be an ordinary partial cut, since offers are "nicht einkürzbar". | Medium: rows may be offers or steps, and the data does not say |
| **12,297 identical bid rows.** | Unlimited offers per provider, step structure, and pooled resources make identical (volume, price) rows plausible. Keep them. | Medium |
| Bid sizes always 5–100 MW, standardised (5, 10, 20, 100). | Min first slice 5 MW, max 100 MW per offer. | High (data matches 100 %) |
| **No-buy daily auctions come in whole delivery weeks** and coincide with a larger-than-usual weekly award (AUC 0.73 / 0.84). | The weekly/daily split is price-based: when weekly offers are cheap relative to expected daily prices, Swissgrid covers the week in the weekly tender and the daily auctions of that week buy nothing. | High (mechanism stated, pattern matches) |
| **2024: daily procurement almost suspended** (down Jan–Dec 1, up Feb 5–Dec 1); weekly awards high. | Same split rule, applied for most of a year. The documents do not mention a suspension. **Ask Jean / Swissgrid.** | Hypothesis |
| Demand (awarded volume) does not move before spikes; level shifts appear across years. | Demand is set by a yearly recalibrated optimisation and depends on prices, including aFRR prices, which we don't model. | Medium |
| Weekly spikes are preceded by a thinner book (fewer bids, less offered volume). | With pay-as-bid and a price-dependent demand, less competing volume lets the last accepted offers ask more. | Descriptive, consistent |

## 3. Bidder-data limitations

1. **No bidder identity.** Results are anonymised by design. We can't prove that the same company changed its bid.
2. **Offer vs combination is ambiguous.** The data has one row per published bid, with no offer ID. We can't tell which rows belong to the same step offer, or how up and down combinations of one offer are linked.
3. **Volume is a weak fingerprint.** Only 8.4 % of weekly bids have a volume that is unique in two consecutive auctions.
4. **Clearing can't be reproduced exactly.** It is a joint TRL + aFRR cost minimisation with an unpublished demand curve. A merit-order simulator is an approximation by construction.
5. **aFRR side is invisible here** (accepted aFRR bids only, in the other data set). Its prices affect the TRL volume.
6. **Information timing.** Bidders see the anonymised result at gate closure. Our 2 h rule is stricter than what bidders know, which is fine for forecasting. For behaviour, it means bidders can react to auctions we treat as unknown (e.g. Tuesday's bidders already know Monday's result).
7. **Data gaps.** 31 missing daily dates (2024) and one correction auction (`TRL_19_KW42_KORR_up`).

## 4. Research questions revisited (Sprint 0 → after Sprint 1)

| Sprint 0 question | What we know now | Week 2 approach |
|---|---|---|
| Q1. After a rejection, do offered prices fall in later comparable auctions? How fast and by how much? | No bidder IDs, and volume alone tracks only 8 %. Individual tracking needs a better match. | Match each bid at *t* to the nearest-price bid of the **same volume** at *t−1* (same series), and compare outcomes; also analyse the **supply curve** (price quantiles of the book) without identities. |
| Q2. Do bidders react to their own outcome or to the published marginal price? | Bidders know their own outcome and the anonymised book at gate closure. | Regress the matched price change on (own rejected?, own distance to the marginal, change in the marginal). |
| Q3. Does an adaptive-agent simulator beat the baseline and supervised models during shifts? | Clearing is not pure merit order, and demand depends on prices. | Simulator v0 with merit-order clearing + demand = recent awarded volume; measure how much the "out-of-merit" approximation costs. |

The Sprint 0 note assumed Swissgrid accepts the cheapest offers upward. **That assumption is wrong for ~46 % of auctions**, because of indivisible step offers and joint optimisation with aFRR. The simulator must treat this as a known approximation.

## 5. Testable hypotheses for Week 2

Each uses only information available before the target auction's gate closure, unless stated.

| # | Hypothesis | Test | Data |
|---|---|---|---|
| H1 | A bid rejected at *t−1* reappears **cheaper** at *t* (same series, same volume, nearest price); an accepted bid reappears at the same or a higher price. | Distribution of matched log price changes, rejected vs accepted; Mann-Whitney. Match quality reported. | mFRR weekly first, then daily per block |
| H2 | The price change of a rejected bid scales with its distance above the marginal price (own outcome), not with the change in the published marginal price. | Regression of Δlog price on the distance above the marginal and on Δ marginal. | Matched bids |
| H3 | Reaction is fast: most of the adjustment happens within 1 auction (weekly) or 1–2 days (daily). | Matched changes at lags 1, 2, 3. | Matched bids |
| H4 | A thinner book at *t−1* (lower offered volume and bid count vs recent level) raises the probability of a weekly jump at *t*. | Already descriptive (AUC 0.65–0.68). Turn into a feature and test against the baseline in the walk-forward harness. | Weekly |
| H5 | Daily no-buy weeks are predictable from the weekly auction of the same week (awarded volume above its recent level). | Classifier on weekly awarded vs recent level; AUC/precision by year. | Weekly → daily |
| H6 | Daily prices of a week depend on how much the weekly auction covered: a large weekly award means less daily demand and lower daily marginals. | Daily marginal vs weekly awarded-relative, same week. | Weekly → daily |
| H7 | The V21 change (daily aFRR tenders from 06.05.2026) shifts daily mFRR prices or volumes. | Before/after comparison around 2026-05-06; regime-shift test case for Sprint 3. | Daily 2026 |
| H8 | Yearly demand recalibration causes level shifts around the turn of the year. | Awarded volume and price levels before/after each 1 January. | Weekly + daily |

## 6. Consequences for the simulator (v0, Week 2)

- **Clearing:** merit order on the price, as a first approximation. Measure the error against the real marginal and keep "out-of-merit" as a known gap. A v1 could add indivisibility (an offer is either fully taken or not).
- **Demand:** the awarded volume of recent known auctions (same series). For daily auctions, condition on the weekly award of the same week (H5/H6); weeks fully covered by the weekly tender produce no-buy.
- **Agents:** groups defined by volume class (5, 10, 20, 50–100 MW) rather than individual bidders; adaptive rule rejected → price × (1−α), accepted → price × (1+β), plus noise.
- **Information set:** agents may use the result at gate closure (true bidder information); the forecast may only use results ≥ 2 h old (project rule).

## 7. Open questions (for Jean / TAs)

1. Is a row in the offers data one offer or one step combination of an offer? Is there an offer ID in the original Swissgrid files?
2. Why was daily mFRR procurement almost suspended between January and December 2024?
3. Does Carina Power treat the 2026 daily aFRR tenders (V21) as a known regime change?
4. Are second daily tenders visible in the data (e.g. as separate auction IDs), or merged with the first?
