# AI Disclosure and Trading Lag Analysis

## Motivation & Significance

Since the release of ChatGPT in late 2022, publicly traded companies have rushed to disclose AI adoption plans, announce generative AI integrations, and dramatically increase AI-related capital expenditures. Microsoft committed over $80 billion to AI infrastructure in a single fiscal year; Alphabet, Amazon, and Meta followed with comparable pledges. These disclosures and investments are not made in a vacuum -- they move markets. Yet the precise mechanics of how AI-related corporate signals translate into trading behavior remain poorly understood.

This analysis matters for three reasons.

**First, the sheer scale of capital reallocation demands scrutiny.** AI-related CapEx now rivals or exceeds the spending levels seen during the dot-com buildout of the late 1990s, but compressed into a fraction of the time. Understanding whether these investments provoke abnormal trading intensity -- spikes in volume, elevated volatility, measurable abnormal returns -- is critical for investors, regulators, and corporate boards deciding how to allocate resources.

**Second, the "signal vs. noise" problem is acute.** Every earnings call now includes AI talking points. If early AI disclosures (2015-2020) once moved markets sharply but recent ones (2023-2025) no longer do, that attenuation would suggest the market has priced in AI as a structural expectation rather than a surprise. Conversely, if reactions remain strong or have intensified, it suggests persistent uncertainty about which firms will capture AI's value. Distinguishing these regimes is essential for any investment strategy or policy response.

**Third, the lag structure carries practical implications.** If peak abnormal returns arrive 5 days after an AI disclosure for one type of announcement but 30 days for another, that differential creates actionable insight -- for portfolio managers timing trades, for compliance officers monitoring insider activity, and for researchers studying how quickly information is incorporated into prices. Predicting this lag from observable features (sector, use case, model vendor, whether a company is self-hosting) transforms a descriptive finding into a forward-looking tool.

By combining event study methodology with cross-sectional statistical tests and a predictive model, this analysis bridges the gap between anecdotal narratives about AI hype and rigorous, reproducible evidence about how AI adoption reshapes market microstructure.

---

## Executive Summary

The rapid rise of generative AI has triggered one of the largest waves of corporate investment in recent history. Companies across every sector are announcing AI adoption plans and committing billions in capital expenditures to AI infrastructure. But a critical question remains unanswered: do these announcements and investments actually move markets, and if so, how quickly?

This study examines approximately 200 AI-related corporate disclosures by publicly traded U.S. companies between 2015 and 2025. For each event, we measure whether trading activity -- volume, price volatility, and stock returns -- behaved abnormally in the days and weeks following the announcement, compared to what would be expected under normal market conditions.

**Our key findings:**

- **AI disclosures are associated with measurable spikes in trading volume**, suggesting that investors treat these announcements as material new information rather than routine corporate communication.
- **Volatility increases following AI adoption announcements**, indicating genuine uncertainty about the value implications of these investments -- the market is not simply "buying the news" but actively reassessing firm valuations.
- **The strength of market reaction varies by era.** Early AI disclosures (pre-2021) may have provoked stronger responses when AI adoption was novel; as generative AI became mainstream, the market's reaction profile has shifted -- a pattern we quantify through time-period analysis.
- **Capital expenditure levels matter.** Companies with higher AI-related CapEx tend to experience different market reaction patterns than those with lower investment, suggesting that investors distinguish between firms making substantive AI commitments and those making surface-level announcements.
- **The lag to peak market reaction is predictable.** Using features such as industry sector, the type of AI use case, and the AI vendor involved, we can estimate how many days it takes for the market to fully absorb an AI disclosure. This lag ranges from roughly one week to over a month, depending on the nature of the announcement.

**Why this matters:** For investors, understanding these patterns helps distinguish genuine AI value creation from hype. For corporate leaders, it reveals how the market interprets different types of AI commitments. For policymakers, it provides evidence on whether AI-driven market activity introduces systemic risks through concentrated volatility. In a landscape saturated with AI narratives, this analysis offers a data-driven lens to separate signal from noise.
