# Churn Analysis and Customer Intelligence

Streamlit app for managing cancellation reviews and analyzing separate historical customer CSV data.

## Run locally (Windows)

Run these commands from the project directory (the folder containing `app.py` and
`requirements.txt`). If your PowerShell prompt is in the extracted parent folder
instead, enter the project folder first:

```powershell
Set-Location .\Minor_Project
```

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.headless true --server.port 8501
```

Open http://localhost:8501/ (also reachable at http://127.0.0.1:8501/).
The app initializes its SQLite database automatically if it does not exist.

## Demo workspace

Use the **Demo workspace** selector in the sidebar to explore a populated dashboard without creating or changing real records. The selection is retained in the page URL so the demo badge and data separation remain visible after refresh. The demo generates 480 synthetic customer profiles and 21 synthetic cancellation reviews in memory. All demo screens display **DEMO DATA — NOT REAL CUSTOMERS**, and demo exports include that classification in both their filenames and data.

The customer generator uses a fixed seed (`2026`); the churn outcome is sampled with noise from generated tenure, contract, satisfaction, engagement, support-contact, and escalation attributes. Demo cancellation cases use a separate fixed seed and dates relative to the current date. These examples are illustrative, not research evidence or predictions about real customers.

The demo model uses a reproducible stratified train/holdout split. Customer IDs and churn outcomes are excluded from model inputs; cancellation reasons and dates are not model features. Demo cases are read-only and never enter the SQLite case database. While the demo workspace is selected, real CSV upload and case-edit controls are not used. Return to **Real workspace** to resume normal CSV and case workflows. **Reset demo workspace** clears in-memory demo model results and restores the generated starting examples; it does not delete SQLite records or uploaded files.

## Database and backups

By default, review cases are stored at `data\cancellation_cases.sqlite3` under this project directory. To store it elsewhere, set `CANCELLATION_DB_PATH` to an absolute path before starting Streamlit. Back up the SQLite file with the app stopped; retain backups securely because case records can include customer identifiers and comments.

Cancellation-review records are separate from uploaded historical CSV data. Uploaded files are used only for the current app session; they are not added to the case database. Model evaluation uses stratified out-of-fold predictions on the labeled historical dataset. It should not be interpreted as guaranteed future performance. To score, upload a separate CSV containing active customers with the same feature columns used for training.

## Navigation

- **Welcome**: first-launch introduction, four-step workflow, important data/model notes, and navigation cards for the existing tools.
- **Retention Overview**: filtered request, outcome, cancellation, save-rate, revenue, reason, and trend summaries.
- **New Cancellation Review** and **Case History**: create, inspect, update, filter, and export cases.
- **Churn Analysis**: inspect historical churn labels, train/evaluate a model, and optionally score active customers.
- **Data Quality/Settings**: inspect missing values, duplicates, inferred types, and confirm the historical target mapping.

Welcome is the initial page for a new app session and remains available from the sidebar. Its cards use the same in-app navigation as the sidebar links; switching between Real and Demo workspaces does not reset the selected page or uploaded data.

## Private cloud deployment with persistent case storage

This deployment copy supports Supabase Postgres for durable hosted case records while keeping SQLite for local development. Do not deploy the local `data\cancellation_cases.sqlite3` file or upload it to GitHub. Existing local SQLite records are not migrated by this setup; the hosted Supabase table starts empty unless you explicitly migrate records later.

1. Create a private GitHub repository and add this project folder's source files. The included `.gitignore` excludes SQLite data, Python caches, local environments, and Streamlit secrets.
2. Create a Supabase project and run `supabase_schema.sql` once in its SQL Editor.
3. In Supabase, copy the project URL and the service-role key. Store them only in the Streamlit Community Cloud app's **Secrets** settings using the structure in `.streamlit/secrets.toml.example`. Keep `require_remote_db = true`; this prevents a silent fallback to temporary cloud-local SQLite.
4. In Streamlit Community Cloud, create an app from the private GitHub repository with `app.py` as the entry point. Configure the app as private and invite only intended viewers. App access is restricted to invitees; all invitees can use the app's case features and see records in the shared database.
5. Confirm the deployment can connect to Supabase before inviting viewers. Never commit the service-role key or add it to a source file.

The hosted app uses the Supabase service-role credential only from Streamlit's server-side secrets. The database has Row Level Security enabled and does not grant the public `anon` or `authenticated` roles access. Treat the service-role key as an administrator credential. Because the hosted Real workspace allows case creation and all invited viewers can access the same records, invite only trusted people and use synthetic data for demonstrations. Community Cloud local file storage is not guaranteed to persist; hosted cases use Supabase instead.
