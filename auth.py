"""
Authentication Module for Snoonu ML Dashboard
==============================================
Handles user authentication, session management, and access control.
"""

import streamlit as st
import streamlit_authenticator as stauth
from datetime import datetime
import yaml


def get_auth_config():
    """Get authentication configuration from secrets."""
    try:
        # Try to load from Streamlit secrets (works on Cloud and locally)
        credentials = {
            'usernames': {}
        }

        # Load users from secrets
        if hasattr(st, 'secrets') and 'credentials' in st.secrets:
            creds = st.secrets['credentials']

            # Extract usernames
            if 'usernames' in creds:
                for username in creds['usernames']:
                    user_data = creds['usernames'][username]
                    credentials['usernames'][username] = {
                        'name': user_data.get('name', username),
                        'email': user_data.get('email', f'{username}@snoonu.com'),
                        'password': user_data.get('password', '')
                    }

            cookie_name = creds.get('cookie_name', 'snoonu_ml_auth')
            cookie_key = creds.get('cookie_key', 'snoonu_ml_secret_key')
            cookie_expiry = creds.get('cookie_expiry_days', 7)
        else:
            # Fallback for local development without secrets
            credentials = {
                'usernames': {
                    'admin': {
                        'name': 'Admin User',
                        'email': 'admin@snoonu.com',
                        'password': '$2b$12$YSkBJpxcXVEvyOKLESWtS.jRLRHk3DkqDvn/42/Qd7reMcqumaYg2'
                    }
                }
            }
            cookie_name = 'snoonu_ml_auth'
            cookie_key = 'snoonu_ml_secret_key'
            cookie_expiry = 7

        # Get preauthorized emails
        preauthorized = []
        if hasattr(st, 'secrets') and 'preauthorized' in st.secrets:
            preauthorized = list(st.secrets['preauthorized'].get('emails', []))

        return {
            'credentials': credentials,
            'cookie': {
                'name': cookie_name,
                'key': cookie_key,
                'expiry_days': cookie_expiry
            },
            'preauthorized': preauthorized
        }

    except Exception as e:
        st.error(f"Error loading auth config: {e}")
        return None


def init_authenticator():
    """Initialize the authenticator."""
    config = get_auth_config()

    if config is None:
        return None

    try:
        authenticator = stauth.Authenticate(
            credentials=config['credentials'],
            cookie_name=config['cookie']['name'],
            cookie_key=config['cookie']['key'],
            cookie_expiry_days=config['cookie']['expiry_days'],
            auto_hash=False  # Passwords are already hashed
        )
        return authenticator
    except Exception as e:
        st.error(f"Failed to initialize authenticator: {e}")
        return None


def render_login_page():
    """Render a premium dark mode login portal."""
    st.markdown("""
        <style>
        /* Hide default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {display: none;}

        /* Full page dark background */
        .stApp {
            background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #0E1117 100%);
        }

        /* Login container */
        .login-portal {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 80vh;
            padding: 2rem;
        }

        /* Logo section */
        .login-logo-section {
            text-align: center;
            margin-bottom: 3rem;
        }

        .login-logo {
            font-family: 'Altform', sans-serif;
            font-size: 3.5rem;
            font-weight: 900;
            color: #FAFAFA;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }

        .login-logo span {
            color: #FF6B35;
        }

        .login-tagline {
            color: #666;
            font-size: 1rem;
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        /* Decorative elements */
        .login-decoration {
            position: absolute;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,107,53,0.1) 0%, transparent 70%);
            pointer-events: none;
        }

        .login-decoration-1 {
            top: -100px;
            right: -100px;
        }

        .login-decoration-2 {
            bottom: -150px;
            left: -150px;
        }

        /* Style the login form */
        .login-card {
            background: rgba(26, 31, 46, 0.8);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 24px;
            padding: 2.5rem;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.5);
        }

        .login-card h3 {
            color: #FAFAFA !important;
            font-family: 'Altform', sans-serif;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        /* Style Streamlit inputs */
        .login-card .stTextInput > div > div {
            background: #0E1117 !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 12px !important;
        }

        .login-card .stTextInput input {
            color: #FAFAFA !important;
            padding: 0.75rem 1rem !important;
        }

        .login-card .stTextInput input::placeholder {
            color: #666 !important;
        }

        .login-card .stTextInput > div > div:focus-within {
            border-color: #FF6B35 !important;
            box-shadow: 0 0 0 2px rgba(255,107,53,0.2) !important;
        }

        /* Style the submit button */
        .login-card .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #FF6B35 0%, #FF8B5E 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(255,107,53,0.3) !important;
        }

        .login-card .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(255,107,53,0.4) !important;
        }

        /* Footer text */
        .login-footer {
            text-align: center;
            margin-top: 2rem;
            color: #444;
            font-size: 0.85rem;
        }

        .login-footer a {
            color: #FF6B35;
            text-decoration: none;
        }
        </style>

        <div class="login-decoration login-decoration-1"></div>
        <div class="login-decoration login-decoration-2"></div>

        <div class="login-portal">
            <div class="login-logo-section">
                <div class="login-logo">Snoonu<span>ML</span></div>
                <div class="login-tagline">Insight Marketplace</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    return True


def check_authentication():
    """
    Check if user is authenticated.
    Returns: (is_authenticated, username, authenticator)
    """
    authenticator = init_authenticator()

    if authenticator is None:
        st.error("Authentication system not configured properly.")
        st.stop()
        return False, None, None

    # Check if already authenticated (from cookie or previous login)
    authentication_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    # If already authenticated, return immediately without showing login form
    if authentication_status is True and username:
        return True, username, authenticator

    # Not authenticated - show login form
    # Use a container to control visibility
    login_container = st.empty()

    with login_container.container():
        render_login_page()
        authenticator.login(location='main')

    # Re-check authentication status after login attempt
    authentication_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    if authentication_status is False:
        st.error('Username or password is incorrect')
        return False, None, authenticator

    if authentication_status is None:
        st.warning('Please enter your credentials')
        return False, None, authenticator

    # Successfully authenticated - clear login form
    login_container.empty()
    return True, username, authenticator


def render_user_menu(authenticator, username):
    """Render user menu in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Logged in as:** {username}")

        authenticator.logout(button_name='Logout', location='sidebar')


def log_activity(username: str, action: str, details: str = None):
    """Log user activity for audit trail."""
    timestamp = datetime.now().isoformat()

    # Store in session state for this session
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []

    st.session_state.activity_log.append({
        'timestamp': timestamp,
        'username': username,
        'action': action,
        'details': details
    })


def require_auth(func):
    """Decorator to require authentication for a function."""
    def wrapper(*args, **kwargs):
        is_auth, username, auth = check_authentication()
        if is_auth:
            return func(*args, **kwargs)
        else:
            st.stop()
    return wrapper
