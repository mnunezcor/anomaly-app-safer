                st.error(f"REVIEW (score {s:.3f}, cutoff {cutoff:.3f})")
            else:
                st.success(f"OK (score {s:.3f}, cutoff {cutoff:.3f})")
            share = (history_scores < s).mean() * 100
            st.write(f"This is more unusual than {share:.0f}% of past transactions.")
            st.caption("An anomaly flag indicates unusual behavior; it is not proof of fraud.")
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            st.error(
                f"This transaction was not scored: {exc} "
                "Correct the input values and try again."
            )
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
                "The live stream was not started because `new_transactions.csv` "
                "could not be read. Check that the file exists in the GitHub repository "
                "and is a valid CSV, then redeploy."
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
                "Correct the listed rows in `new_transactions.csv`, commit the corrected "
                "file to GitHub, and try the stream again. No transactions were scored."
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
                    "This row could not be scored, so it was not labeled OK or REVIEW. "
                    "Check the row and contact the app administrator if the problem continues."
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
            answer = stream.set_index("transaction_id")["needs_review"]
            answer = answer.astype(bool)
            hits = sum(bool(answer.loc[f["id"]]) for f in flagged)
            st.write(
                f"{hits} of the {len(flagged)} flagged transactions really needed "
                f"review. In total, {int(answer.sum())} transactions needed review."
            )
