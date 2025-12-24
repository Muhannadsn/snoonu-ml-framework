# Streamlit Cloud Deployment Guide

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at share.streamlit.io)
3. Your data files accessible (or sample data)

## Step 1: Prepare Repository

### Push to GitHub

```bash
# Initialize git (if not already)
git init

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/snoonu-ml-framework.git

# Add files (secrets.toml is automatically ignored)
git add .

# Commit
git commit -m "Initial commit - Snoonu ML Dashboard"

# Push
git push -u origin main
```

### Verify .gitignore

Ensure these sensitive files are NOT committed:
- `.streamlit/secrets.toml`
- `data/*.parquet`
- Any `.env` files

## Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set:
   - **Repository**: `YOUR_USERNAME/snoonu-ml-framework`
   - **Branch**: `main`
   - **Main file path**: `app.py`

## Step 3: Configure Secrets

In Streamlit Cloud dashboard:

1. Click your app's **Settings** (gear icon)
2. Go to **Secrets** section
3. Paste your secrets configuration:

```toml
[credentials]
cookie_name = "snoonu_ml_auth"
cookie_key = "YOUR_SECURE_RANDOM_KEY"
cookie_expiry_days = 7

[credentials.usernames.admin]
name = "Admin User"
email = "admin@snoonu.com"
password = "$2b$12$YSkBJpxcXVEvyOKLESWtS.jRLRHk3DkqDvn/42/Qd7reMcqumaYg2"

[credentials.usernames.analyst]
name = "Data Analyst"
email = "analyst@snoonu.com"
password = "$2b$12$YSkBJpxcXVEvyOKLESWtS.jRLRHk3DkqDvn/42/Qd7reMcqumaYg2"

[preauthorized]
emails = ["team@snoonu.com"]
```

### Generate Secure Cookie Key

```python
import secrets
print(secrets.token_hex(32))
```

### Generate Password Hashes

```python
import bcrypt
password = "your_secure_password"
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
print(hashed)
```

## Step 4: Upload Sample Data

Since data files are gitignored, you have options:

### Option A: Include Sample Data
Create a small sample dataset and commit it:
```python
# Create sample_data.py
df_sample = df.head(10000)
df_sample.to_parquet('data/sample.parquet')
```

### Option B: Cloud Storage
Store data in cloud storage (GCS, S3) and load dynamically.

### Option C: Database
Connect to a database using Streamlit secrets for credentials.

## Security Best Practices

### Cookie Security
- Use a strong, random cookie key (32+ characters)
- Set appropriate expiry (7 days default)
- Cookie is HTTP-only and secure by default on Streamlit Cloud

### Password Policy
- Use strong passwords (12+ characters, mixed case, numbers, symbols)
- All passwords are bcrypt hashed
- Never store plain text passwords

### Access Control
- Create separate accounts for different roles
- Review active sessions periodically
- Rotate cookie keys periodically

### Network Security
Streamlit Cloud provides:
- HTTPS by default
- DDoS protection
- Automatic SSL certificates

## User Accounts

Default accounts (change passwords before deployment!):

| Username | Default Password | Role |
|----------|------------------|------|
| admin    | snoonu2024      | Full access |
| analyst  | snoonu2024      | Data analyst |
| viewer   | snoonu2024      | Read-only |

## Troubleshooting

### "Authentication system not configured"
- Verify secrets are set in Streamlit Cloud dashboard
- Check TOML syntax in secrets

### "Invalid credentials"
- Ensure password hash is correct
- Regenerate hash if needed

### App crashes on load
- Check logs in Streamlit Cloud dashboard
- Verify all dependencies in requirements.txt

## Sharing Your App

Once deployed, share your app URL:
```
https://your-app-name.streamlit.app
```

Users will see a login page and must authenticate before accessing the dashboard.

## Updating the App

Push changes to GitHub - Streamlit Cloud auto-deploys:
```bash
git add .
git commit -m "Update description"
git push
```

## Resource Limits (Free Tier)

- 1 GB RAM
- 1 CPU
- Apps sleep after inactivity
- Unlimited viewers
