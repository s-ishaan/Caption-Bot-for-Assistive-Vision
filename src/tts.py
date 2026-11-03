import os
import tempfile
from typing import Optional
from gtts import gTTS


def generate_speech(text: str, output_path: Optional[str] = None, lang: str = "en") -> Optional[str]:
    """Converts a generated caption into speech (audio file) for assistive vision playback.

    Args:
        text: The caption string to speak.
        output_path: Optional destination MP3 file path.
        lang: Language code (default 'en').

    Returns:
        The path to the generated audio file, or None if TTS fails.
    """
    if not text or not text.strip():
        return None

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"[Assistive TTS Warning] Could not generate speech audio: {e}")
        return None
