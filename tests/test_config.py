from src.config import load_config


def test_default_embedding_model_is_supported_gemini_model():
    config = load_config()

    assert config.embedding_model == "models/gemini-embedding-001"


def test_default_chat_model_is_current_gemini_model():
    config = load_config()

    assert config.chat_model == "gemini-3.5-flash"
