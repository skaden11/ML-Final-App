# Deploy the ML Group 1 Streamlit App

This folder is ready to upload to GitHub and deploy with Streamlit Community Cloud.

## What is included

- `app.py` - the Streamlit app
- `requirements.txt` - Python packages Streamlit Cloud must install
- `artifacts/` - trained models, model metadata, and held-out test predictions
- `README.md` - project-facing app description

## Recommended deployment path

Use Streamlit Community Cloud. It creates a normal shareable web link, so reviewers do not need to extract a zip file or run terminal commands.

## Step 1: Create a GitHub repository

1. Go to <https://github.com>.
2. Sign in or create an account.
3. Click the `+` button in the top-right corner.
4. Choose `New repository`.
5. Repository name suggestion: `ml-group-1-music-popularity-app`.
6. Set visibility to `Public`.
7. Click `Create repository`.

## Step 2: Upload this folder

1. In the new GitHub repository, click `uploading an existing file`.
2. Open this folder on your computer:
   `C:\Users\spenc\Downloads\ML Group 1 Streamlit Deploy Ready`
3. Drag all files and folders from inside it into GitHub:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `DEPLOY_TO_STREAMLIT.md`
   - `artifacts`
4. Wait until the files finish uploading.
5. At the bottom, click `Commit changes`.

Important: upload the contents of the folder, not the zip file itself.

## Step 3: Deploy on Streamlit Community Cloud

1. Go to <https://streamlit.io/cloud>.
2. Sign in with GitHub.
3. Click `Create app` or `New app`.
4. Choose `Deploy a public app from GitHub`.
5. Select your repository: `ml-group-1-music-popularity-app`.
6. Set the branch to `main`.
7. Set the main file path to:
   `app.py`
8. Click `Deploy`.

After deployment, Streamlit will give you a public link similar to:

`https://ml-group-1-music-popularity-app.streamlit.app`

That is the link to submit or share.

## If the app does not load

Check these common issues first:

1. Make sure `artifacts/` was uploaded to GitHub.
2. Make sure `artifacts/` contains:
   - `all_context_hgb.joblib`
   - `audio_only_ridge.joblib`
   - `metadata.json`
   - `test_predictions.csv`
3. Make sure the Streamlit main file path is exactly:
   `app.py`
4. Make sure `requirements.txt` is in the same GitHub level as `app.py`.

## What to say in the final report

The deployed app is the user-facing artifact for the project. It provides an interactive popularity prediction interface and an Update 2.0 actual-vs-predicted evaluation tab. The evaluation tab compares model predictions against actual Spotify popularity on held-out test rows, making the artifact more transparent than a prediction-only demo.
