-- CREDIT RISK PIPELINE - ANALYTICAL QUERIES ON THE AUDIT TRAIL
-- ======================================================================
-- Every query here runs directly against credit_risk.db (the table
-- database.py creates and api.py's /decide + /narrate write to). Run
-- them with:
--     sqlite3 -header -column credit_risk.db < queries.sql
-- or paste one at a time into `sqlite3 credit_risk.db`.
--
-- These exist because the whole point of storing X1..X24 as their own
-- typed columns (not one JSON blob) is that real questions like these
-- become a normal WHERE/GROUP BY, not application code.


-- 1. DECISION VOLUME AND MIX
-- "How many applicants got each outcome, and what share of the total?"
-- The first thing anyone reviewing this system would ask.
SELECT
    decision_tier,
    COUNT(*) AS num_decisions,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM scoring_log), 1) AS pct_of_total
FROM scoring_log
GROUP BY decision_tier
ORDER BY num_decisions DESC;


-- 2. AVERAGE RISK BY TIER (a sanity check on the policy itself)
-- If the policy thresholds are working as intended, REJECT's average
-- PD should clearly exceed REVIEW's, which should clearly exceed
-- APPROVE's -- this query is how you'd actually verify that, not just
-- assume it.
SELECT
    decision_tier,
    ROUND(AVG(xgb_pd), 4) AS avg_pd,
    ROUND(MIN(xgb_pd), 4) AS min_pd,
    ROUND(MAX(xgb_pd), 4) AS max_pd
FROM scoring_log
GROUP BY decision_tier
ORDER BY avg_pd;


-- 3. DECISION VOLUME OVER TIME
-- "How many applicants did we score per day?" -- the basis for any
-- volume or trend chart.
SELECT
    DATE(created_at) AS decision_date,
    COUNT(*) AS num_decisions
FROM scoring_log
GROUP BY decision_date
ORDER BY decision_date;


-- 4. OPERATIONAL BACKLOG: decisions with no plain-language explanation yet
-- Phase 7's narrative is generated on demand, not automatically -- this
-- is the query an ops team would run to find what's still outstanding.
SELECT id, decision_tier, xgb_pd, created_at
FROM scoring_log
WHERE narrative IS NULL
ORDER BY id;


-- 5. LLM FACT-CHECK PASS RATE
-- Of the narratives that HAVE been generated, what fraction passed
-- their independent fact-check on the first attempt? A dropping trend
-- here would mean the prompts or the policy document need attention.
SELECT
    COUNT(*) AS total_narratives,
    SUM(narrative_verified) AS passed_fact_check,
    ROUND(100.0 * SUM(narrative_verified) / COUNT(*), 1) AS pct_passed
FROM scoring_log
WHERE narrative IS NOT NULL;


-- 6. WHICH FEATURES MOST OFTEN DRIVE A DECISION
-- shap_top_features is stored as a JSON array of strings (a natural
-- nested list, unlike X1..X24 -- see database.py's docstring for why
-- that split makes sense). json_each() unpacks it one element per row,
-- and substr()/instr() pulls the feature name off the front of each
-- string (e.g. "X4 (value=2) increased..." -> "X4").
SELECT
    substr(feature.value, 1, instr(feature.value, ' ') - 1) AS feature_name,
    COUNT(*) AS times_in_top_5
FROM scoring_log, json_each(scoring_log.shap_top_features) AS feature
GROUP BY feature_name
ORDER BY times_in_top_5 DESC;
