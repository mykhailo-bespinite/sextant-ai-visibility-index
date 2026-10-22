import base64

from aeo_index.redact import REDACTED, Redactor, secret_values


def test_configured_secrets_are_redacted():
    redact = Redactor(["k-test-123456", "short"])
    assert redact("key k-test-123456 and short") == f"key {REDACTED} and short"


def test_key_shapes_are_redacted_even_when_not_configured():
    redact = Redactor([])
    for text in (
        "sk-proj-AbCdEfGh12345678",
        "sk-ant-api03-AbCdEfGh12345678",
        "xai-AbCdEfGh12345678",
        "pplx-AbCdEfGh12345678",
        "AIzaSyA-1234567890abcdefghijklmnop",
    ):
        assert redact(f"bad key {text} here") == f"bad key {REDACTED} here"


def test_authorization_credentials_are_redacted():
    redact = Redactor([])
    assert redact("Authorization: Bearer abc.def-123") == f"Authorization: Bearer {REDACTED}"
    assert redact("Basic bG9naW46cGFzcw==") == f"Basic {REDACTED}"


def test_longer_secret_wins_over_a_secret_inside_it():
    redact = Redactor(["abcdefgh", "abcdefgh-ijkl"])
    assert redact("abcdefgh-ijkl") == REDACTED


def test_text_without_secrets_is_unchanged():
    assert Redactor(["k-test-123456"])("Profound leads.") == "Profound leads."


def test_secret_values_from_env_include_the_basic_auth_form():
    env = {
        "OPENAI_API_KEY": "k-test-123456",
        "DATAFORSEO_LOGIN": "login-test-1",
        "DATAFORSEO_PASSWORD": "pw-test-123456",
        "HOME": "/home/x",
    }
    values = secret_values(env)
    assert "k-test-123456" in values and "pw-test-123456" in values and "login-test-1" in values
    assert base64.b64encode(b"login-test-1:pw-test-123456").decode() in values
    assert "/home/x" not in values
