# ML Group 1 Music Popularity Predictor

This Streamlit app is the user-facing artifact for the final music popularity project.

Update 2.0 adds a held-out actual-vs-predicted comparison tab.

## What it does

- Predicts Spotify popularity on a 0-100 scale.
- Provides an audio-only mode for hypothetical or unreleased songs.
- Provides a retrospective context mode for existing songs with known Billboard and/or Last.fm context.
- Shows the selected prediction beside the audio-only comparison.
- Shows held-out test songs with actual Spotify popularity, final-model prediction, audio-only prediction, and absolute error.
- Includes model notes, test metrics, and context coverage caveats.

## Important interpretation

Context mode is retrospective. Billboard rank, chart appearances, Last.fm playcount, and Last.fm sample listeners are post-release signals. Do not present context mode as a release-time hit predictor.

## Run locally

From this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained model files and held-out prediction comparison file are included in `artifacts/`. If the artifacts are missing, regenerate them from the final notebook outputs with:

```bash
python train_app_models.py
```
