# Consumer Credit Underwriting Policy (Reference Document)

*This is a synthetic policy document written for a portfolio project. It
does not represent any real institution's actual policy. It exists so
the automated explanation system (Phase 7) has real text to retrieve
facts from, instead of relying on an LLM's own unverified claims about
"how underwriting works."*

## 1. Purpose and Scope

This policy governs how consumer credit applications are scored,
decided, and explained. It applies to all applications processed
through the automated Probability of Default (PD) scoring system. The
statistical model and the fixed policy thresholds in this document are
the sole source of every APPROVE, REVIEW, or REJECT decision. No
generative AI system, chatbot, or automated assistant is authorized to
change, override, or independently produce a credit decision. AI tools
may only be used to explain a decision that the deterministic scoring
system has already made.

## 2. Probability of Default (PD) and Decision Tiers

Every application receives a calibrated Probability of Default (PD): a
number between 0 and 1 representing the model's estimate of the real-
world likelihood that the applicant will default, based on historical
outcomes of similar applicants. A PD of 0.35 means that, among
applicants who looked like this one historically, roughly 35% defaulted.

Applications are sorted into one of three tiers based on PD, using
fixed, non-discretionary thresholds:

- **APPROVE** — PD strictly below 0.20. The applicant's estimated
  default risk is low enough to approve without manual review.
- **REVIEW** — PD from 0.20 up to and including 0.50. This tier does
  not mean the applicant is "bad" — it means the model's confidence is
  not strong enough in either direction to auto-decide, and a human
  underwriter should review the file, potentially requesting additional
  documentation (proof of income, employment verification) before a
  final decision.
- **REJECT** — PD strictly above 0.50. The applicant's estimated
  default risk is too high to approve under current risk appetite.

These thresholds (0.20 and 0.50) are reviewed periodically by the risk
committee and may change; the automated system always reads the
current thresholds from its own configuration rather than having them
hardcoded into any explanation text.

## 3. Which Model Drives the Decision

Two models are trained for every scoring cycle: a Logistic Regression
model (fully transparent, used as a benchmark and a sanity check) and
an XGBoost model (higher predictive performance, used for the actual
decision once calibrated). The calibrated XGBoost PD is the figure that
determines the decision tier. The Logistic Regression PD is retained
and shown alongside it for transparency and as a cross-check, but does
not itself drive the decision.

## 4. Explaining Individual Decisions (SHAP Factors)

Every decision must be explainable at the individual applicant level,
not just in aggregate. The system uses SHAP (SHapley Additive
exPlanations) to attribute each applicant's predicted PD to their
specific feature values, expressed in probability points (e.g. "this
factor increased the applicant's predicted default probability by 0.08,
or 8 percentage points").

When communicating these factors to an applicant or a reviewing
underwriter:
- State the top 3-5 factors that moved the prediction the most,
  in either direction.
- Do not claim a factor "caused" the outcome — SHAP values describe
  how much a factor influenced this specific model's prediction, not a
  causal real-world mechanism.
- Do not invent or infer factors that were not actually reported by the
  SHAP explanation for this specific applicant.
- Numeric feature names in this system (X1 through X24) are generic
  placeholders in the current dataset and do not have confirmed
  real-world labels; do not assert what a feature "really means" beyond
  what is explicitly provided.

## 5. Adverse Action Notice Requirements (REJECT and REVIEW-declined tiers)

For any application that results in a REJECT decision, or a REVIEW that
an underwriter subsequently declines, the applicant is entitled to a
clear, specific statement of the principal reasons, consistent with
standard adverse-action notice practice (in the U.S., this mirrors the
spirit of the Equal Credit Opportunity Act / Regulation B and the Fair
Credit Reporting Act, though this project does not claim formal legal
compliance). Acceptable reason categories to cite, when supported by
that applicant's actual SHAP factors, include:

- Insufficient time at current residence or employment
- High existing debt relative to income or credit history
- Length or amount of requested credit relative to demonstrated
  repayment capacity
- Adverse history on existing credit obligations
- Insufficient credit history to assess risk confidently

A reason may only be cited if it is actually supported by that specific
applicant's SHAP explanation for that specific decision. Generic or
templated reasons not tied to the applicant's actual factors are not
acceptable.

## 6. Fair Lending and Prohibited Factors

Race, color, religion, national origin, sex, marital status, and age
(provided the applicant is old enough to enter a contract) must never
be used as a factor in scoring or cited as a reason for a decision,
even if a proxy for one of these appears to correlate with the model's
output. Any explanation referencing such a factor must be treated as
an error and regenerated.

## 7. Guidance for Automated Explanation Generation

This section is written specifically for the automated explanation
assistant. When generating a plain-language explanation of a decision:

1. Treat the decision tier (APPROVE / REVIEW / REJECT) and the PD
   values you are given as fixed facts. You are explaining a decision
   that has already been made by the deterministic scoring system — you
   are not evaluating the applicant, re-scoring them, or capable of
   changing the outcome.
2. Ground every factual claim in either (a) the specific SHAP values
   provided for this applicant, or (b) the text of this policy
   document. Do not state a number, threshold, or policy rule that
   does not appear in the provided inputs.
3. Use plain, respectful, non-technical language suitable for a
   customer-facing letter or an underwriter's case notes.
4. If the tier is REVIEW, make clear this is not a rejection — it means
   a human will review the file.
5. Never suggest the applicant could have gotten a different outcome
   through actions unrelated to the actual SHAP factors provided (no
   generic "improve your credit score" filler advice).
