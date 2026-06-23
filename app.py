from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"


st.set_page_config(
    page_title="ML Group 1 Music Popularity Predictor",
    page_icon=None,
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    div[data-testid="stMetric"] {
        background: #f6f8fb;
        border: 1px solid #d9e2ec;
        color: #102a43;
        padding: 0.85rem 1rem;
        border-radius: 8px;
    }
    .small-note {
        color: #53606e;
        font-size: 0.92rem;
        line-height: 1.35;
    }
    .warning-box {
        border: 1px solid #d7e2f0;
        background: #eef5fb;
        color: #102a43;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0 1rem 0;
    }
    .warning-box strong {
        color: #0b2545;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    audio_path = ARTIFACT_DIR / "audio_only_ridge.joblib"
    context_path = ARTIFACT_DIR / "all_context_hgb.joblib"
    metadata_path = ARTIFACT_DIR / "metadata.json"
    missing = [p.name for p in [audio_path, context_path, metadata_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing app artifacts: "
            + ", ".join(missing)
            + ". Run `python train_app_models.py` from the app folder."
        )
    return (
        joblib.load(audio_path),
        joblib.load(context_path),
        json.loads(metadata_path.read_text(encoding="utf-8")),
    )


@st.cache_data
def load_test_predictions():
    predictions_path = ARTIFACT_DIR / "test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Missing held-out prediction artifact: test_predictions.csv. "
            "Run `python train_app_models.py` from the app folder."
        )
    predictions = pd.read_csv(predictions_path)
    numeric_cols = [
        "actual_popularity",
        "all_context_prediction",
        "all_context_error",
        "all_context_abs_error",
        "audio_only_prediction",
        "audio_only_abs_error",
    ]
    for col in numeric_cols:
        predictions[col] = pd.to_numeric(predictions[col], errors="coerce")
    return predictions


def clip_prediction(value):
    return float(np.clip(value, 0, 100))


def default(metadata, feature, fallback):
    return float(metadata.get("feature_defaults", {}).get(feature, fallback))


def build_audio_inputs(metadata):
    st.subheader("Audio features")
    c1, c2, c3 = st.columns(3)
    with c1:
        danceability = st.slider("Danceability", 0.0, 1.0, default(metadata, "danceability", 0.55), 0.01)
        energy = st.slider("Energy", 0.0, 1.0, default(metadata, "energy", 0.65), 0.01)
        valence = st.slider("Valence", 0.0, 1.0, default(metadata, "valence", 0.45), 0.01)
        acousticness = st.slider("Acousticness", 0.0, 1.0, default(metadata, "acousticness", 0.25), 0.01)
    with c2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, default(metadata, "instrumentalness", 0.0), 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, default(metadata, "liveness", 0.18), 0.01)
        speechiness = st.slider("Speechiness", 0.0, 1.0, default(metadata, "speechiness", 0.06), 0.01)
        loudness = st.number_input("Loudness (dB)", min_value=-60.0, max_value=5.0, value=default(metadata, "loudness", -7.0), step=0.1)
    with c3:
        tempo = st.number_input("Tempo (BPM)", min_value=40.0, max_value=240.0, value=default(metadata, "tempo", 120.0), step=1.0)
        duration_seconds = st.number_input(
            "Duration (seconds)",
            min_value=30.0,
            max_value=900.0,
            value=max(30.0, default(metadata, "duration_ms", 210000.0) / 1000),
            step=1.0,
        )
        key = st.selectbox("Key", list(range(12)), index=int(np.clip(round(default(metadata, "key", 5)), 0, 11)))
        mode = st.radio("Mode", ["Minor", "Major"], horizontal=True, index=int(default(metadata, "mode", 1)))
        time_signature = st.selectbox("Time signature", [3, 4, 5, 6, 7], index=1)
        explicit = st.checkbox("Explicit lyrics", value=bool(round(default(metadata, "explicit", 0))))

    return {
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": duration_seconds * 1000,
        "key": int(key),
        "mode": 1 if mode == "Major" else 0,
        "time_signature": int(time_signature),
        "explicit": 1 if explicit else 0,
    }


def build_context_inputs():
    st.subheader("Retrospective context")
    st.markdown(
        "<div class='warning-box'><strong>Context mode is retrospective.</strong> "
        "Use it only for existing songs where chart/listening context is already known.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    context = {
        "matched_billboard": 0,
        "best_rank": 0.0,
        "peak_rank": 0.0,
        "max_weeks_on_chart": 0.0,
        "chart_appearances": 0.0,
        "first_chart_month": 0.0,
        "last_chart_month": 0.0,
        "mean_rank_score": 0.0,
        "max_rank_score": 0.0,
        "matched_lastfm": 0,
        "log1p_lastfm_sample_playcount": 0.0,
        "log1p_lastfm_sample_listeners": 0.0,
        "lastfm_sample_user_reach_pct": 0.0,
        "lastfm_scrobble_span_days": 0.0,
    }

    with c1:
        use_billboard = st.checkbox("Song has Billboard Hot 100 context")
        if use_billboard:
            best_rank = st.number_input("Best Billboard rank (1 is best)", 1, 100, 50)
            peak_rank = st.number_input("Peak rank from source", 1, 100, int(best_rank))
            average_rank = st.number_input("Average chart rank", 1, 100, int(best_rank))
            max_weeks = st.number_input("Max weeks on chart", 1, 60, 5)
            appearances = st.number_input("Chart appearances", 1, 60, int(max_weeks))
            first_month = st.number_input("First chart month", 1, 12, 1)
            last_month = st.number_input("Last chart month", 1, 12, int(first_month))
            context.update(
                {
                    "matched_billboard": 1,
                    "best_rank": float(best_rank),
                    "peak_rank": float(peak_rank),
                    "max_weeks_on_chart": float(max_weeks),
                    "chart_appearances": float(appearances),
                    "first_chart_month": float(first_month),
                    "last_chart_month": float(last_month),
                    "mean_rank_score": float(101 - average_rank),
                    "max_rank_score": float(101 - best_rank),
                }
            )

    with c2:
        use_lastfm = st.checkbox("Song has Last.fm sample context")
        if use_lastfm:
            playcount = st.number_input("Last.fm sample playcount", 1, 10000, 10)
            listeners = st.number_input("Last.fm sample listeners (50-user sample)", 1, 50, 2)
            span_days = st.number_input("Scrobble span (days)", 0, 8000, 30)
            context.update(
                {
                    "matched_lastfm": 1,
                    "log1p_lastfm_sample_playcount": float(math.log1p(playcount)),
                    "log1p_lastfm_sample_listeners": float(math.log1p(listeners)),
                    "lastfm_sample_user_reach_pct": float(min(100, listeners / 50 * 100)),
                    "lastfm_scrobble_span_days": float(span_days),
                }
            )
    return context


def make_prediction_frame(audio_inputs, context_inputs, features):
    row = {**audio_inputs, **context_inputs}
    return pd.DataFrame([{feature: row.get(feature, 0.0) for feature in features}])


def prediction_text(audio_pred, selected_pred, mode):
    delta = selected_pred - audio_pred
    if mode == "Audio-only":
        return "This prediction uses audio features only."
    if abs(delta) < 1:
        return "Context changes the estimate by less than one popularity point."
    direction = "higher" if delta > 0 else "lower"
    return f"Retrospective context makes the estimate {abs(delta):.1f} points {direction} than audio-only."


def summarize_prediction_errors(data):
    errors = data["all_context_error"].dropna()
    abs_errors = data["all_context_abs_error"].dropna()
    if data.empty or errors.empty:
        return {"mae": np.nan, "rmse": np.nan, "mean_error": np.nan}
    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_error": float(errors.mean()),
    }


audio_model, context_model, metadata = load_artifacts()
test_predictions = load_test_predictions()

st.title("ML Group 1 Music Popularity Predictor")
st.caption("A transparent Streamlit artifact for the final music popularity project.")

st.markdown(
    "<div class='warning-box'>This app predicts Spotify popularity on a 0-100 scale. "
    "Audio-only mode is suitable for hypothetical songs. Context mode is retrospective because Billboard and Last.fm fields are post-release signals.</div>",
    unsafe_allow_html=True,
)

tab_predict, tab_actual, tab_notes, tab_coverage = st.tabs(
    ["Predict", "Actual vs predicted", "Model notes", "Data coverage"]
)

with tab_predict:
    mode = st.radio("Prediction mode", ["Audio-only", "Retrospective context"], horizontal=True)
    audio_inputs = build_audio_inputs(metadata)
    if mode == "Retrospective context":
        context_inputs = build_context_inputs()
    else:
        context_inputs = {feature: 0.0 for feature in metadata["billboard_features"] + metadata["lastfm_features"]}

    audio_df = make_prediction_frame(audio_inputs, {}, metadata["audio_features"])
    context_df = make_prediction_frame(audio_inputs, context_inputs, metadata["all_context_features"])

    audio_prediction = clip_prediction(audio_model.predict(audio_df)[0])
    selected_prediction = audio_prediction if mode == "Audio-only" else clip_prediction(context_model.predict(context_df)[0])

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected prediction", f"{selected_prediction:.1f} / 100")
    m2.metric("Audio-only comparison", f"{audio_prediction:.1f} / 100")
    m3.metric("Difference", f"{selected_prediction - audio_prediction:+.1f}")

    st.markdown(f"**Interpretation:** {prediction_text(audio_prediction, selected_prediction, mode)}")
    comparison = pd.DataFrame(
        {
            "Model view": ["Audio-only", "Selected mode"],
            "Predicted popularity": [audio_prediction, selected_prediction],
        }
    )
    st.bar_chart(comparison, x="Model view", y="Predicted popularity", height=260)

with tab_actual:
    st.subheader("Held-out actual vs predicted popularity")
    st.markdown(
        "This is the model cross-check: these songs were held out from model training. "
        "Actual popularity is Spotify's real 0-100 popularity label; prediction is the final all-context model output."
    )

    f1, f2, f3 = st.columns(3)
    context_options = ["All"] + sorted(test_predictions["context_group"].dropna().unique().tolist())
    bin_options = ["All"] + list(["0-20", "20-40", "40-60", "60-80", "80-100"])
    selected_context = f1.selectbox("Context group", context_options)
    selected_bin = f2.selectbox("Popularity bin", bin_options)
    sort_mode = f3.selectbox("Show examples", ["Largest errors", "Smallest errors", "Highest actual popularity"])

    filtered = test_predictions.copy()
    if selected_context != "All":
        filtered = filtered[filtered["context_group"].eq(selected_context)]
    if selected_bin != "All":
        filtered = filtered[filtered["popularity_bin"].eq(selected_bin)]

    summary = summarize_prediction_errors(filtered)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Songs shown", f"{len(filtered):,}")
    m2.metric("MAE", "n/a" if np.isnan(summary["mae"]) else f"{summary['mae']:.2f}")
    m3.metric("RMSE", "n/a" if np.isnan(summary["rmse"]) else f"{summary['rmse']:.2f}")
    m4.metric("Mean error", "n/a" if np.isnan(summary["mean_error"]) else f"{summary['mean_error']:+.2f}")

    if filtered.empty:
        st.info("No held-out songs match the selected filters.")
    else:
        plot_df = filtered.sample(min(len(filtered), 1500), random_state=34351).rename(
            columns={
                "actual_popularity": "Actual popularity",
                "all_context_prediction": "Predicted popularity",
            }
        )
        st.scatter_chart(plot_df, x="Actual popularity", y="Predicted popularity", height=320)

        if sort_mode == "Largest errors":
            table = filtered.sort_values("all_context_abs_error", ascending=False)
        elif sort_mode == "Smallest errors":
            table = filtered.sort_values("all_context_abs_error", ascending=True)
        else:
            table = filtered.sort_values("actual_popularity", ascending=False)

        display = table[
            [
                "track_name",
                "artists",
                "actual_popularity",
                "all_context_prediction",
                "all_context_abs_error",
                "audio_only_prediction",
                "audio_only_abs_error",
                "context_group",
                "popularity_bin",
            ]
        ].head(100).copy()
        display = display.rename(
            columns={
                "track_name": "Track",
                "artists": "Artist(s)",
                "actual_popularity": "Actual",
                "all_context_prediction": "Final model predicted",
                "all_context_abs_error": "Final abs error",
                "audio_only_prediction": "Audio-only predicted",
                "audio_only_abs_error": "Audio-only abs error",
                "context_group": "Context group",
                "popularity_bin": "Popularity bin",
            }
        )
        for col in ["Actual", "Final model predicted", "Final abs error", "Audio-only predicted", "Audio-only abs error"]:
            display[col] = display[col].round(2)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.markdown(
            "<p class='small-note'>The table is limited to 100 rows for readability. "
            "The metrics above use every held-out row matching the selected filters.</p>",
            unsafe_allow_html=True,
        )

with tab_notes:
    final_test = metadata["final_test"]
    st.subheader("Model result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out test MAE", f"{float(final_test['test_mae']):.2f}")
    c2.metric("Held-out test RMSE", f"{float(final_test['test_rmse']):.2f}")
    c3.metric("Held-out test R2", f"{float(final_test['test_r2']):.3f}")
    st.markdown(
        "The final report selected Histogram Gradient Boosting using validation results, then evaluated it once on the held-out test set. "
        "The model is useful for retrospective estimation, but the R2 is modest, so predictions should be interpreted as estimates."
    )
    st.dataframe(pd.DataFrame(metadata["model_comparison"]), use_container_width=True, hide_index=True)

with tab_coverage:
    st.subheader("Why context needs caveats")
    st.markdown(
        "Most Spotify songs in the modeling table do not have Billboard or Last.fm context matches. "
        "This is why the app separates audio-only prediction from retrospective context prediction."
    )
    st.dataframe(pd.DataFrame(metadata["context_coverage"]), use_container_width=True, hide_index=True)
    st.markdown(
        "<p class='small-note'>Last.fm values are from a 50-user sample, not global Last.fm totals. Billboard values are from 2022 Hot 100 weekly chart data.</p>",
        unsafe_allow_html=True,
    )
