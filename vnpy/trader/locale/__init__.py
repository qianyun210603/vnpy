from gettext import translation, NullTranslations
from pathlib import Path


localedir: Path = Path(__file__).parent

translations: NullTranslations = translation("vnpy", localedir=localedir, fallback=True)

_ = translations.gettext
