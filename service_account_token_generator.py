from google.oauth2 import service_account
import google.auth.transport.requests

def generate_access_token(json_key_file, scopes):
    try:
        credentials = service_account.Credentials.from_service_account_file(
            json_key_file, scopes=scopes
        )
        auth_request = google.auth.transport.requests.Request()
        credentials.refresh(auth_request)
        return credentials.token
    except Exception as e:
        raise Exception(f"Error generating access token: {e}")
