import streamlit as st
import re
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
def sanitize_input(prompt):
    return prompt.strip()

# =========================================================
# AI GUARDIAN - LLM SAFETY & HALLUCINATION DEFENSE ENGINE
# =========================================================

st.set_page_config(
    page_title="AI Guardian",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------
# PAGE DESIGN
# -------------------------

st.title("🛡️ AI Guardian")
st.subheader("LLM Safety & Hallucination Defense Engine")

st.markdown(
    """
    **AI Guardian** analyzes user prompts for jailbreak attacks,
    generates an AI response, and checks the response against
    trusted information for possible hallucinations.
    """
)

# =========================================================
# MODELS
# =========================================================

@st.cache_resource
def load_generation_model():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


@st.cache_resource
def load_nli_model():
    return pipeline(
        "text-classification",
        model="cross-encoder/nli-MiniLM2-L6-H768",
        top_k=None
    )


# =========================================================
# JAILBREAK DETECTOR
# =========================================================

JAILBREAK_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all previous instructions",
    r"forget your instructions",
    r"forget all instructions",
    r"ignore your system prompt",
    r"reveal your system prompt",
    r"show me your system prompt",
    r"reveal hidden instructions",
    r"bypass safety",
    r"bypass your safety",
    r"disable safety",
    r"act as an unrestricted",
    r"act as an uncensored",
    r"jailbreak",
    r"developer mode",
    r"do anything now",
    r"dan mode"
]

def detect_jailbreak(prompt):
    prompt_lower = prompt.lower()

    jailbreak_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?instructions",
        r"forget\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?previous\s+instructions",

        r"reveal\s+(your\s+)?system\s+prompt",
        r"show\s+(me\s+)?(your\s+)?system\s+prompt",
        r"reveal\s+(your\s+)?hidden\s+instructions",
        r"show\s+(me\s+)?hidden\s+instructions",
        r"reveal\s+(your\s+)?instructions",

        r"bypass\s+(the\s+)?safety",
        r"bypass\s+(your\s+)?safety",
        r"disable\s+(the\s+)?safety",
        r"disable\s+(your\s+)?safety",
        r"override\s+(the\s+)?safety",

        r"act\s+as\s+an?\s+unrestricted",
        r"act\s+as\s+an?\s+uncensored",
        r"pretend\s+you\s+have\s+no\s+restrictions",

        r"developer\s+mode",
        r"jailbreak",
        r"dan\s+mode",
        r"do\s+anything\s+now",

        r"system\s+instructions",
        r"hidden\s+instructions",
        r"ignore\s+your\s+rules",
        r"ignore\s+your\s+guidelines"
    ]

    detected = []

    for pattern in jailbreak_patterns:
        if re.search(pattern, prompt_lower):
            detected.append(pattern)

    if detected:
        # Higher score when multiple jailbreak patterns are detected
        risk_score = min(100, 75 + len(detected) * 5)
        return True, risk_score, detected

    return False, 5, []


# =========================================================
# GENERATE RESPONSE
# =========================================================

def generate_response(prompt, context):

    tokenizer, model = load_generation_model()

    if context.strip():

        final_prompt = f"""
Answer the user using ONLY the information provided in the context.

Context:
{context}

User Question:
{prompt}

If the context does not contain enough information, say:
I don't have enough information in the provided context.

Answer:
"""

    else:

        final_prompt = f"""
Answer the following question clearly and briefly:

{prompt}

Answer:
"""

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


# =========================================================
# HALLUCINATION / NLI CHECK
# =========================================================

def check_factuality(context, response):

    if not context.strip() or not response.strip():
        return {
            "score": 50.0,
            "status": "UNKNOWN",
            "claims": []
        }

    nli = load_nli_model()

    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", response)
        if s.strip()
    ]

    claims = []
    scores = []

    for sentence in sentences[:5]:

        try:

            result = nli(
                {
                    "text": context,
                    "text_pair": sentence
                }
            )

            # Normalize result
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    result = result[0]

            if not isinstance(result, list):
                result = [result]

            best = max(
                result,
                key=lambda x: x["score"]
            )

            label = best["label"].upper()
            score = float(best["score"])

            # -------------------------
            # LABEL HANDLING
            # -------------------------

            if "ENTAIL" in label:

                claim_score = score * 100
                claim_status = "SUPPORTED"

            elif "CONTRADICT" in label:

                claim_score = 0
                claim_status = "CONTRADICTION"

            elif "NEUTRAL" in label:

                claim_score = 50
                claim_status = "UNCERTAIN"

            else:

                # Unknown label
                claim_score = 50
                claim_status = "UNCERTAIN"

            scores.append(claim_score)

            claims.append(
                {
                    "claim": sentence,
                    "score": claim_score,
                    "status": claim_status
                }
            )

        except Exception as e:

            claims.append(
                {
                    "claim": sentence,
                    "score": 50,
                    "status": "UNCERTAIN"
                }
            )

            scores.append(50)

    # -------------------------
    # OVERALL SCORE
    # -------------------------

    if not scores:

        return {
            "score": 50.0,
            "status": "UNKNOWN",
            "claims": []
        }

    factuality = sum(scores) / len(scores)

    # -------------------------
    # HALLUCINATION LEVEL
    # -------------------------

    contradiction_count = sum(
        1
        for claim in claims
        if claim["status"] == "CONTRADICTION"
    )

    if contradiction_count > 0 or factuality < 30:

        status = "HIGH HALLUCINATION RISK"

    elif factuality < 60:

        status = "MEDIUM HALLUCINATION RISK"

    else:

        status = "LOW HALLUCINATION RISK"

    return {
        "score": factuality,
        "status": status,
        "claims": claims
    }


# =========================================================
# USER INTERFACE
# =========================================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("🔐 Input")

    user_prompt = st.text_area(
        "Enter your prompt:",
        height=150,
        placeholder="Example: Explain machine learning..."
    )

with col2:

    st.subheader("📚 Trusted Context")

    trusted_context = st.text_area(
        "Enter trusted information:",
        height=150,
        placeholder="Example: Machine learning is a branch of AI..."
    )


st.divider()

check_button = st.button(
    "🛡️ CHECK & GENERATE",
    type="primary",
    use_container_width=True
)


# =========================================================
# MAIN PIPELINE
# =========================================================

if check_button:

    if not user_prompt.strip():

        st.warning("Please enter a prompt.")

    else:

        start_time = time.time()

        # -------------------------
        # 1. SANITIZE
        # -------------------------

        clean_prompt = sanitize_input(user_prompt)

        # -------------------------
        # 2. JAILBREAK CHECK
        # -------------------------

        is_jailbreak, risk_score, detected_patterns = \
            detect_jailbreak(clean_prompt)

        if is_jailbreak:

            st.error("🚨 JAILBREAK / PROMPT INJECTION DETECTED")

            st.write(
                f"**Risk Score:** {risk_score:.0f}%"
            )

            st.write(
                "**Action:** BLOCKED"
            )

            if detected_patterns:

                st.write("Detected patterns:")

                for pattern in detected_patterns:
                    st.code(pattern)

            st.info(
                "The prompt was blocked before being sent to the AI model."
            )

        else:

            # -------------------------
            # 3. SAFE PROMPT
            # -------------------------

            st.success(
                f"✅ Prompt appears safe — Risk Score: {risk_score:.0f}%"
            )

            # -------------------------
            # 4. GENERATE
            # -------------------------

            with st.spinner("Generating AI response..."):

                try:

                    response = generate_response(
                        clean_prompt,
                        trusted_context
                    )

                except Exception as e:

                    st.error(
                        "Model error. Please try again."
                    )

                    st.exception(e)

                    response = None

            if response:

                # -------------------------
                # 5. FACTUALITY CHECK
                # -------------------------

                with st.spinner(
                    "Checking response for hallucinations..."
                ):

                    factuality = check_factuality(
                        trusted_context,
                        response
                    )

                # -------------------------
                # 6. DISPLAY RESPONSE
                # -------------------------

                st.divider()

                st.subheader("🤖 AI Response")

                st.info(response)

                # -------------------------
                # 7. DASHBOARD
                # -------------------------

                st.subheader("📊 Safety Analysis")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric(
                        "Jailbreak Risk",
                        f"{risk_score:.0f}%"
                    )

                with c2:

                    if trusted_context.strip():

                        st.metric(
                            "Factuality",
                            f"{factuality['score']:.1f}%"
                        )

                    else:

                        st.metric(
                            "Factuality",
                            "N/A"
                        )

                with c3:

                    st.metric(
                        "Hallucination",
                        "YES"
                        if "HALLUCINATION"
                        in factuality["status"]
                        else "LOW"
                    )

                with c4:

                    latency = time.time() - start_time

                    st.metric(
                        "Latency",
                        f"{latency:.2f}s"
                    )

                # -------------------------
                # 8. CLAIM ANALYSIS
                # -------------------------

                if "claims" in factuality:

                     
                    st.divider()

                    st.subheader(
                        "🔍 Claim-Level Analysis"
                    )

                    for claim in factuality["claims"]:

                        if claim["status"] == "SUPPORTED":

                            st.success(
                                f"✅ {claim['claim']}"
                            )

                        elif claim["status"] == "CONTRADICTION":

                            st.error(
                                f"❌ {claim['claim']}"
                            )

                        else:

                            st.warning(
                                f"🟡 {claim['claim']}"
                            )

                    st.write(
                        f"**Overall Result:** "
                        f"{factuality['status']}"
                    )