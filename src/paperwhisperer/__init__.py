"""PaperWhisperer — a RAG application for chatting with PDF documents."""

# Validate TLS against the OS trust store (like the browser does), not only
# certifi's bundle. Needed when antivirus/proxy software intercepts HTTPS with
# a root CA that Windows trusts but certifi doesn't -- otherwise calls to the
# Gemini API fail with CERTIFICATE_VERIFY_FAILED. Injected here at package
# import so it is active before any network client is created.
import truststore as _truststore

_truststore.inject_into_ssl()

# Install the global LLM response cache at startup.
from paperwhisperer.cache import setup_caching  # noqa: E402

setup_caching()

__version__ = "0.1.0"
