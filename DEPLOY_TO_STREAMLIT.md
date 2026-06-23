# Deploy the ML Group 1 Streamlit App

This is the deploy-ready package for the Streamlit Community Cloud version of the app.

## Why the previous deployment error happened

The traceback that ended inside `load_artifacts()` means Streamlit successfully found and ran `app.py`, but it did not find one or more files inside the required `artifacts/` folder.

The GitHub repository must have this exact structure:

```text
app.py
requirements.txt
README.md
DEPLOY_TO_STREAMLIT.md
artifacts/
  all_context_hgb.joblib
  audio_only_ridge.joblib
  metadata.json
  test_predictions.csv
```

The `artifacts` folder must be uploaded to GitHub. It cannot stay only on your computer.

## Fast fix for your current Streamlit app

If your GitHub repo already has `app.py`, do this:

1. Open your GitHub repository.
2. Click `Add file`.
3. Click `Upload files`.
4. Open this folder on your computer:
   `C:\Users\spenc\Downloads\ML Group 1 Streamlit Deploy Ready v2`
5. Drag the entire `artifacts` folder into GitHub.
6. Also drag the updated `requirements.txt` and `app.py`.
7. Click `Commit changes`.
8. Go back to Streamlit Cloud.
9. Open the app menu and click `Reboot app`.

That should fix the missing-artifacts error.

## Clean redeploy option

If the repo is messy, use the clean approach:

1. Create a new public GitHub repository.
2. Upload the contents of this folder, not the zip file.
3. Go to <https://streamlit.io/cloud>.
4. Create a new app from that GitHub repository.
5. Set the main file path to:
   `app.py`
6. In advanced settings, use Python 3.12 if Streamlit asks for a Python version.
7. Deploy.

## What not to do

- Do not upload only the zip file.
- Do not upload only `app.py`.
- Do not put `artifacts` inside another nested folder.
- Do not set the Streamlit main file path to a file inside `artifacts`.

## How to tell it is fixed

The app should open with these tabs:

- `Predict`
- `Actual vs predicted`
- `Model notes`
- `Data coverage`

The `Actual vs predicted` tab should show the held-out test comparison with 11,576 songs.
