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
    """Render the login page."""
    # Custom CSS for login page
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-logo {
            font-size: 2.5rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
        }
        .login-logo span {
            color: #FF6B35;
        }
        </style>
    """, unsafe_allow_html=True)

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
            <div class="login-header">
                <div class="login-logo">Snoonu<span>ML</span></div>
                <p style="color: #888;">Analytics Platform</p>
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
