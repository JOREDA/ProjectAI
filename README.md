# Navrang Systems — Anonymous Feedback & Attrition Prevention

This Streamlit demo uses a **synthetic dataset**. Do not use it to collect or process real employee feedback.

## Run locally

Install the packages in `requirements.txt`, configure the secrets below, then run:

```powershell
streamlit run app.py
```

For local development, credentials can be supplied through `.env` or `credentials.json`. Both files are ignored by Git; never commit them.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository for this project and upload the project files. Do not upload `.env`, `credentials.json`, `.streamlit/secrets.toml`, `cache/`, `__pycache__/`, or `models/`.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from that repository and set the main file to `app.py`.
3. In the app's **Settings → Secrets**, add:

   ```toml
   groq_api_key = "your Groq API key"
   ombudsperson_passcode = "a new, strong passcode"
   ```

   Do not reuse the local passcode. The app will not start unless both credentials are configured.
4. Deploy the app. Streamlit Community Cloud will provide the public URL.

The app and its sample data are public to anyone with that URL. Keep real employee data and credentials out of the repository and the demo.
