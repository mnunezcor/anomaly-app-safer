import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Transaction Monitor", layout="wide")

EXPECTED_FEATURES = [
    "log_amount",
    "amount_ratio",
    "transactions_last_hour",
    "log_gap",
    "log_age",
]

REQUIRED_COLUMNS = [
    "transaction_id",
    "amount",
    "customer_typical_amount",
    "transactions_last_hour",
    "minutes_since_previous",
    "account_age_days",
]


def load_model_bundle():
    """Load the existing trained model without retraining or replacing it."""
    try:
        bundle = joblib.load("model.joblib")
    except Exception as exc:
        st.error(
            "The trained model could not be loaded. The app cannot score transactions. "
            "Check that model.joblib exists in the repository and is a valid model file."
        )
        st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
        st.stop()

    try:
        model = bundle["model"]
        history_scores = np.asarray(bundle["history_scores"], dtype=float)
        features = list(bundle["features"])
    except Exception as exc:
        st.error(
            "The model file is incomplete or has an unexpected structure. "
            "The app cannot safely score transactions."
        )
        st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
        st.stop()

    if history_scores.size == 0 or not np.isfinite(history_scores).all():
        st.error("The model's historical scores are empty or contain invalid values.")
        st.stop()

    if features != EXPECTED_FEATURES:
        st.error(
            "The model's feature list does not match the features expected by this app. "
            "Scoring has been stopped to avoid incorrect results."
        )
        st.stop()

    return model, history_scores, features


model, history_scores, features = load_model_bundle()


def score(amount, typical, last_hour, gap, age):
    """Apply the original feature transformations and score with the existing model."""
    values = [amount, typical, last_hour, gap, age]
    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError("All transaction inputs must be numeric.")

    if not np.isfinite(values).all():
        raise ValueError("Transaction inputs must be finite numbers.")
    if values[0] <= 0:
        raise ValueError("Transaction amount must be greater than zero.")
    if values[1] <= 0:
        raise ValueError("Customer's typical amount must be greater than zero.")
    if values[2] < 0 or values[3] < 0 or values[4] < 0:
        raise ValueError(
            "Transactions in the last hour, minutes since previous transaction, "
            "and account age cannot be negative."
        )

    row = pd.DataFrame([{
        "log_amount": np.log1p(values[0]),
        "amount_ratio": values[0] / values[1],
        "transactions_last_hour": values[2],
        "log_gap": np.log1p(values[3]),
        "log_age": np.log1p(values[4]),
    }])[features]
    return -model.score_samples(row)[0]


def validate_stream_file(stream):
    """Return a list of problems; do not score any row if the file is invalid."""
    problems = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in stream.columns]
    if missing_columns:
        problems.append("Missing required column(s): " + ", ".join(missing_columns))
        return problems

    if stream.empty:
        problems.append("The CSV contains no transaction rows.")
        return problems

    if stream["transaction_id"].isna().any():
        rows = (stream.index[stream["transaction_id"].isna()] + 2).tolist()
        problems.append(f"Missing transaction_id in CSV row(s): {rows}")

    numeric_columns = [
        "amount",
        "customer_typical_amount",
        "transactions_last_hour",
        "minutes_since_previous",
        "account_age_days",
    ]

    for column in numeric_columns:
        converted = pd.to_numeric(stream[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted)
        if invalid.any():
            rows = (stream.index[invalid] + 2).tolist()
            problems.append(
                f"Non-numeric, missing, or non-finite value in '{column}' "
                f"at CSV row(s): {rows}"
            )
        stream[column] = converted

    # Check ranges independently for every numeric column, even if another
    # column contains a conversion error.
    range_rules = {
        "amount": lambda values: values <= 0,
        "customer_typical_amount": lambda values: values <= 0,
        "transactions_last_hour": lambda values: values < 0,
        "minutes_since_previous": lambda values: values < 0,
        "account_age_days": lambda values: values < 0,
    }
    for column, invalid_rule in range_rules.items():
        invalid = stream[column].notna() & invalid_rule(stream[column])
        if invalid.any():
            rows = (stream.index[invalid] + 2).tolist()
            problems.append(
                f"Invalid value in '{column}' at CSV row(s): {rows}. "
                "Amounts must be greater than zero; counts, gaps, and ages "
                "cannot be negative."
            )

    return problems


# ---- sidebar: preserve the original review-budget logic ----
st.title("Transaction Monitor")
st.write(
    "Each new transaction is scored by Isolation Forest. "
    "If the score is above the cutoff, it goes to review."
)
st.caption(
    "An anomaly flag indicates unusual behavior; it is not proof of fraud."
)

pct = st.sidebar.slider("Send this share of transactions to review (%)", 1, 10, 4)
cutoff = np.percentile(history_scores, 100 - pct)
st.sidebar.write(f"Score cutoff: **{cutoff:.3f}**")
st.sidebar.caption(
    "The cutoff is set from past transactions: a transaction scoring above it "
    f"is more unusual than the top {pct}% of history."
)

tab1, tab2 = st.tabs(["Check one transaction", "Live stream"])

# ---- tab 1: type in one transaction ----
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount ($)", min_value=1.0, value=60.0)
    typical = c1.number_input("Customer's typical amount ($)", min_value=1.0, value=50.0)
    last_hour = c1.number_input("Transactions in the last hour", min_value=0, value=1)
    gap = c2.number_input("Minutes since previous transaction", min_value=0.0, value=120.0)
    age = c2.number_input("Account age (days)", min_value=0, value=700)

    if st.button("Score this transaction"):
        try:
            s = score(amount, typical, last_hour, gap, age)
            if s > cutoff:
                st.error(f"REVIEW (score {s:.3f}, cutoff {cutoff:.3f})")
            else:
                st.success(f"OK (score {s:.3f}, cutoff {cutoff:.3f})")
            share = (history_scores < s).mean() * 100
            st.write(f"This is more unusual than {share:.0f}% of past transactions.")
            st.caption(
                "An anomaly flag indicates unusual behavior; it is not proof of fraud."
            )
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            st.error(f"This transaction was not scored: {exc}")
        except Exception as exc:
            st.error(
                "The model could not score this transaction. No OK or REVIEW decision "
                "was produced. Contact the app administrator before relying on this check."
            )
            st.caption(f"Technical detail: {type(exc).__name__}: {exc}")

# ---- tab 2: transactions arriving one by one ----
with tab2:
    st.write(
        "Press the button. The app receives the new transactions one at a time "
        "and scores each one as it arrives."
    )
    speed = st.select_slider("Speed", options=["Slow", "Medium", "Fast"], value="Medium")
    pause = {"Slow": 0.5, "Medium": 0.15, "Fast": 0.02}[speed]

    if st.button("Start the stream"):
        try:
            stream = pd.read_csv("new_transactions.csv")
        except Exception as exc:
            st.error(
                "The live stream was not started because new_transactions.csv "
                "could not be read. Check that the file exists and is a valid CSV."
            )
            st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
            st.stop()

        problems = validate_stream_file(stream)
        if problems:
            st.error(
                "The live stream was not started. The file contains missing or invalid "
                "data, so scoring it could produce misleading results."
            )
            for problem in problems:
                st.write(f"- {problem}")
            st.info(
                "Correct new_transactions.csv, commit the corrected file to GitHub, "
                "and try the stream again. No transactions were scored."
            )
            st.stop()

        box = st.empty()
        counts = st.empty()
        st.subheader("Sent to review")
        table = st.empty()
        flagged = []

        for i, r in stream.iterrows():
            try:
                s = score(
                    r["amount"],
                    r["customer_typical_amount"],
                    r["transactions_last_hour"],
                    r["minutes_since_previous"],
                    r["account_age_days"],
                )
            except Exception as exc:
                st.error(
                    f"Stream stopped at transaction {r['transaction_id']}. "
                    "This row could not be scored, so it was not labeled OK or REVIEW."
                )
                st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
                st.warning(
                    f"{i} transaction(s) were processed before the error. "
                    "The remaining transactions were not processed."
                )
                st.stop()

            msg = f"{r['transaction_id']}: ${r['amount']:.2f}  score {s:.3f}"
            if s > cutoff:
                flagged.append({
                    "id": r["transaction_id"],
                    "amount": r["amount"],
                    "typical": r["customer_typical_amount"],
                    "last_hour": r["transactions_last_hour"],
                    "score": round(s, 3),
                })
                box.error(msg + "  ->  REVIEW")
            else:
                box.info(msg + "  ->  OK")

            counts.write(
                f"Received **{i + 1}** of **{len(stream)}**. "
                f"Sent to review: **{len(flagged)}**"
            )
            if flagged:
                table.dataframe(pd.DataFrame(flagged), hide_index=True)
            time.sleep(pause)

        st.success(f"Done. {len(flagged)} transactions were sent to review.")

        # If the optional answer key exists, report the results.
        if "needs_review" in stream.columns:
            answer = stream.set_index("transaction_id")["needs_review"].astype(bool)
            hits = sum(bool(answer.loc[f["id"]]) for f in flagged)
            st.write(
                f"{hits} of the {len(flagged)} flagged transactions really needed "
                f"review. In total, {int(answer.sum())} transactions needed review."
            )
