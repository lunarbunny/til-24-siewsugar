import ffmpeg
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_audio(file_bytes: bytes, sr: int = 16_000) -> np.ndarray:
    """
    Use file's bytes and transform to mono waveform, resampling as necessary
    Parameters
    ----------
    file: bytes
        The bytes of the audio file
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input('pipe:', threads=0)
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run_async(pipe_stdin=True, pipe_stdout=True)
        ).communicate(input=file_bytes)

    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

class ASRManager:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")

    def transcribe(self, audio_bytes: bytes) -> str:
        audio = load_audio(audio_bytes)
        input_features = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt").input_features
        generated_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]
