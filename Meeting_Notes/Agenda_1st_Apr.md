# Meeting Agenda: EUR/USD Macro Report
**Date/Time:** Wednesday, 1:00 PM | **Duration:** 30–40 Minutes

## 1. Agenda (Time-Boxed)
* **a.) | Alignment:** Open the shared 10-part report and compare engineered data (Will) with market sentiment analysis (Rahul).
* **b.) | Qualitative Review:** Rahul presents institutional consensus outlooks. How safe of a bet is assuming that the EUR will appreciate? With what horizon?
* **c.)_ | Quantitative Review:** Will to present the Python/C++ VAR models. Do we have a case that breaks from consensus? How reliable? Will and Rohan to discuss.
* **d.) | Scenario Locking:** Combine narrative and data to decide on teh Base, Bear and Bull probabilities and triggers.
* **e.) | Delegation:** Assign specific blank sections of the template and set a hard deadline for the first draft.

---

## 2. Pre-Meeting Tasks

### Will (Quant & Structure)
- [ ] **Template:** Create the Google Doc using the 10-part format and share the link. (Done)
- [ ] **Data Ingestion:** Pull 2-year Bund vs. US Treasury yields, Brent crude, and CFTC COT positioning data. (Sourcing from the US Fed via API - waiting for API registration approval)
- [ ] **Model Execution:** Run the baseline Vector Autoregression (VAR) model to establish correlations/p-values. (This may not be done in time)
- [ ] **Visuals:** Prepare visual charts to help us decide on wether the euro appreciation is a safe bet.

### Rahul (Fundamentals & Narrative)
- [ ] **Consensus Data:** Find and summarise FX market sentiment using reputable sources (USe AI to summarise if you need, but check over). Focus on forecasts
- [ ] **Market Info:** Fill sections 3 and 4 of the template. For section 3, a simple table with market data is fine - yahoo finance is a freat source.
- [ ] **Risk Triggers:** Define (possible or certain) upcoming event(s) that could cause a EUR dip/spike. (It's fine if theres more of one that=n the other)
