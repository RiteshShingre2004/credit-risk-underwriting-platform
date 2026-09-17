"""
CREDIT RISK PIPELINE - PHASE 7b: LLM EXPLANATION + FACT-CHECK
========================================================
Goal: turn an ALREADY-DECIDED credit decision into a plain-language
explanation a human can read, using an LLM -- WITHOUT letting the LLM
touch the decision itself.

THE HARD CONSTRAINT THIS FILE IS BUILT AROUND:
The statistical model (XGBoost/Logistic Regression) plus the fixed
policy thresholds in api.py's apply_policy() are the ONLY things that
ever produce APPROVE / REVIEW / REJECT. This file never calls that
function, never asks an LLM to classify or score an applicant, and
never accepts an LLM's opinion as a reason to change a tier. Every
function below receives a decision that has already been made and
treats it as a fixed fact to explain -- not a question to answer.

TWO SEPARATE LLM CALLS, ON PURPOSE:
1. generate_explanation() -- writes a plain-language explanation,
   grounded in (a) the specific SHAP factors for this applicant and
   (b) policy text retrieved from policy_docs/underwriting_policy.md.
2. verify_explanation() -- a SECOND, independent call that re-reads the
   draft against only the same source facts and policy text, and
   reports whether every claim in it is actually supported. It does
   not see the first call's reasoning, only its output -- so it can't
   just agree with itself.
If verification fails, we regenerate once with the specific issues fed
back, then return whatever we have with an honest "verified" flag --
we never silently hide a failed check.

RUNS VIA GROQ'S FREE API (open-source Llama models, hosted, no local
downloads -- Groq's business is serving open models fast, not selling
their own model, and their free tier is rate-limited rather than
metered-per-token for this kind of light usage):
    1. Get a free key at https://console.groq.com/keys
    2. export GROQ_API_KEY=gsk_...
Groq's API is OpenAI-compatible, so we just POST to
https://api.groq.com/openai/v1/chat/completions with the standard
{"model", "messages", ...} shape and read the reply back out of
choices[0].message.content -- same request/response shape as OpenAI's
Chat Completions API, just pointed at Groq's servers and Groq's models.
"""

import json
import os

import requests

from policy_retrieval import PolicyRetriever

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Both are OpenAI's open-weight "gpt-oss" models (Apache 2.0 license),
# served on Groq's hardware -- genuinely open source, not just "hosted
# by a company called Groq."
GENERATOR_MODEL = "openai/gpt-oss-120b"
# The verifier is a grading/judge task, not the main generation task --
# a smaller, faster model is the right tool here, the same way you
# wouldn't need your most capable writer to run a checklist.
VERIFIER_MODEL = "openai/gpt-oss-20b"

_retriever = PolicyRetriever()


class LLMServiceError(Exception):
    """
    Raised for any problem reaching or getting a valid response from
    Groq -- missing API key, rate limit, network error, or a bad
    upstream response. api.py catches this ONE type and turns it into
    a clean 503, instead of every different underlying failure mode
    surfacing as a generic, unexplained 500.
    """


def _groq_chat(model, system_prompt, user_prompt, json_schema=None):
    """
    POSTs one chat request to Groq's hosted API and returns the model's
    reply text. `json_schema`, when given, asks Groq to constrain output
    to match that exact JSON Schema (`response_format: {"type":
    "json_schema", ...}`) -- more reliable than plain JSON mode since
    the shape itself is enforced, not just "some valid JSON."
    verify_explanation() below still validates the parsed result rather
    than trusting it blindly, since a model can still return a
    schema-valid-but-wrong answer.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMServiceError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and run: export GROQ_API_KEY=gsk_..."
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "verification_result", "schema": json_schema},
        }

    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after", "a bit")
            raise LLMServiceError(
                f"Groq's free tier rate limit was hit. Wait {retry_after} "
                "seconds and try again -- this is a usage limit, not a bug."
            ) from e
        raise LLMServiceError(f"Groq API returned an error: {e}") from e
    except requests.exceptions.RequestException as e:
        raise LLMServiceError(f"Could not reach Groq's API: {e}") from e

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------------------------------------------
# STEP 1: BUILD A RETRIEVAL QUERY FROM THE DECISION
# -----------------------------------------------------------------
_TIER_HINTS = {
    "APPROVE": "approved application, low estimated risk",
    "REVIEW": "borderline application requiring manual underwriter review",
    "REJECT": "rejected application, adverse action reasons",
}


def _build_query(decision):
    tier = decision["decision_tier"]
    factors = "; ".join(decision["explanation"]["top_features"])
    return f"{_TIER_HINTS.get(tier, '')}. Decision tier: {tier}. Factors: {factors}"


def _decision_facts_for_prompt(decision):
    """
    Builds the exact fact set the LLM is allowed to see and cite --
    deliberately NOT the raw decision dict.

    decision["explanation"]["predicted_probability"] comes from SHAP,
    which explains the RAW (uncalibrated) XGBoost model, while
    decision["xgb_probability_of_default"] is the CALIBRATED PD that
    actually drove the decision (see explainability.py / api.py's
    /decide). The two numbers are usually close but can differ
    noticeably. Handing the LLM a JSON blob with two similarly-named
    "probability" fields let it pick the wrong one and state it as THE
    probability of default -- caught during testing (applicant #54:
    narrative said 0.908, the real decision-driving PD was 0.760). The
    fix is here, not in the prompt wording: only expose one
    unambiguous number in the first place.
    """
    return {
        "decision_tier": decision["decision_tier"],
        "probability_of_default": decision["xgb_probability_of_default"],
        "policy_approve_below": decision["policy_approve_below"],
        "policy_reject_above": decision["policy_reject_above"],
        "top_shap_factors": decision["explanation"]["top_features"],
    }


# -----------------------------------------------------------------
# STEP 2: GENERATE A DRAFT EXPLANATION
# -----------------------------------------------------------------
def generate_explanation(decision, policy_chunks, feedback=None):
    policy_text = "\n\n".join(f"### {c['title']}\n{c['text']}" for c in policy_chunks)
    facts = _decision_facts_for_prompt(decision)

    system_prompt = (
        "You explain credit underwriting decisions in plain language. "
        "The decision below has ALREADY been made by a deterministic "
        "statistical model and a fixed policy engine -- you are not "
        "deciding anything and have no ability to change the outcome. "
        "`probability_of_default` is THE probability that drove this "
        "decision -- the only default-probability number to cite. "
        "Ground every factual claim (numbers, reasons, what a tier means) "
        "in either the DECISION FACTS or the POLICY EXCERPTS provided. "
        "Do not invent numbers, thresholds, or policy rules. Do not "
        "mention race, sex, age, religion, national origin, or marital "
        "status. Write 3-6 short sentences suitable for an applicant or "
        "an underwriter's case notes. Output ONLY the explanation text, "
        "no preamble like 'Here is the explanation:'."
    )

    user_prompt = (
        f"DECISION FACTS (already final):\n{json.dumps(facts, indent=2)}\n\n"
        f"POLICY EXCERPTS:\n{policy_text}\n\n"
    )
    if feedback:
        user_prompt += (
            f"Your previous attempt had these problems -- fix them: "
            f"{feedback}\n\n"
        )
    user_prompt += "Write the plain-language explanation now."

    return _groq_chat(GENERATOR_MODEL, system_prompt, user_prompt).strip()


# -----------------------------------------------------------------
# STEP 3: VERIFY THE DRAFT AGAINST THE SAME SOURCES (independent call)
# -----------------------------------------------------------------
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded", "issues"],
    "additionalProperties": False,
}


def verify_explanation(draft_text, decision, policy_chunks):
    policy_text = "\n\n".join(f"### {c['title']}\n{c['text']}" for c in policy_chunks)
    facts = _decision_facts_for_prompt(decision)

    system_prompt = (
        "You are a strict fact-checker. You will be given a draft "
        "explanation of a credit decision, the actual decision facts, "
        "and policy excerpts. Check EVERY factual claim in the draft: "
        "every number, every stated reason, every claim about what a "
        "policy tier means. A claim is grounded only if it is directly "
        "supported by the DECISION FACTS or the POLICY EXCERPTS -- not "
        "by general knowledge about how banks usually work. "
        "`probability_of_default` in the facts is THE only correct "
        "default-probability number -- flag it as an issue if the draft "
        "cites any other number as 'the' probability of default. Also "
        "flag any mention of race, sex, age, religion, national origin, "
        "or marital status as an issue, and flag if the draft implies "
        "the decision could be different than the tier actually given. "
        "List every unsupported or incorrect claim in `issues` (empty "
        "list if none). Set `grounded` to true only if `issues` is empty."
    )

    user_prompt = (
        f"DRAFT EXPLANATION:\n{draft_text}\n\n"
        f"DECISION FACTS:\n{json.dumps(facts, indent=2)}\n\n"
        f"POLICY EXCERPTS:\n{policy_text}"
    )

    raw = _groq_chat(VERIFIER_MODEL, system_prompt, user_prompt, json_schema=VERIFY_SCHEMA)

    # Even with response_format requested, models can still wrap JSON
    # in stray text or omit a key. Parse defensively and
    # fail SAFE (treat as ungrounded) rather than crash or silently
    # trust a malformed response -- an explanation we can't verify is
    # exactly the case this whole file exists to catch.
    try:
        parsed = json.loads(raw)
        grounded = bool(parsed.get("grounded", False))
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        if grounded and issues:
            grounded = False  # inconsistent response -- don't trust it
        return {"grounded": grounded, "issues": issues}
    except (json.JSONDecodeError, AttributeError):
        return {
            "grounded": False,
            "issues": [f"Verifier returned unparseable output: {raw[:200]!r}"],
        }


# -----------------------------------------------------------------
# STEP 4: ORCHESTRATE -- generate, verify, regenerate once if needed
# -----------------------------------------------------------------
def explain_decision(decision, max_attempts=2):
    """
    Given a decision dict shaped like api.py's /decide response (must
    contain decision_tier and explanation.top_features at minimum),
    returns a narrative explanation plus an honest verification result.
    Never returns a narrative that failed verification without saying so.
    """
    query = _build_query(decision)
    policy_chunks = _retriever.retrieve(query, k=3)

    feedback = None
    draft = None
    verification = None

    for attempt in range(1, max_attempts + 1):
        # Feed the previous attempt's problems back in as extra prompt
        # context, instead of starting from a blank slate each time.
        draft = generate_explanation(decision, policy_chunks, feedback=feedback)

        verification = verify_explanation(draft, decision, policy_chunks)
        if verification["grounded"]:
            break
        feedback = "; ".join(verification["issues"])

    return {
        "narrative": draft,
        "verified": verification["grounded"],
        "verification_issues": verification["issues"],
        "attempts": attempt,
        "policy_sections_used": [c["title"] for c in policy_chunks],
    }


# -----------------------------------------------------------------
# STEP 5: DEMO WHEN RUN DIRECTLY
# -----------------------------------------------------------------
if __name__ == "__main__":
    import joblib

    from explainability import X_test, explain_applicant

    POLICY_APPROVE_BELOW = 0.20
    POLICY_REJECT_ABOVE = 0.50

    def apply_policy(pd_value):
        if pd_value < POLICY_APPROVE_BELOW:
            return "APPROVE"
        if pd_value > POLICY_REJECT_ABOVE:
            return "REJECT"
        return "REVIEW"

    xgb_calibrated = joblib.load("xgb_calibrated.joblib")
    all_pds = xgb_calibrated.predict_proba(X_test)[:, 1]

    # Pick one clearly-REJECT and one clearly-APPROVE applicant so the
    # demo shows the explanation style actually changes with the tier.
    reject_idx = int(all_pds.argmax())
    approve_idx = int(all_pds.argmin())

    for label, idx in [("HIGH-RISK (expect REJECT)", reject_idx),
                        ("LOW-RISK (expect APPROVE)", approve_idx)]:
        pd_value = float(all_pds[idx])
        tier = apply_policy(pd_value)
        explanation = explain_applicant(idx)

        decision = {
            "decision_tier": tier,
            "xgb_probability_of_default": round(pd_value, 4),
            "policy_approve_below": POLICY_APPROVE_BELOW,
            "policy_reject_above": POLICY_REJECT_ABOVE,
            "explanation": explanation,
        }

        print("=" * 60)
        print(f"{label} -- applicant #{idx}, tier={tier}, PD={pd_value:.3f}")
        print("=" * 60)

        result = explain_decision(decision)
        print("\nNarrative:\n" + result["narrative"])
        print(f"\nVerified: {result['verified']} (attempts: {result['attempts']})")
        if result["verification_issues"]:
            print("Issues:", result["verification_issues"])
        print("Policy sections used:", result["policy_sections_used"])
        print()
