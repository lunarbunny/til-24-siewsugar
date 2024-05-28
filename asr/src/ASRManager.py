import ffmpeg
import numpy as np
import torch
from transformers import GenerationConfig, WhisperProcessor, WhisperForConditionalGeneration

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
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sr, loglevel="quiet")
            .run_async(pipe_stdin=True, pipe_stdout=True)
        ).communicate(input=file_bytes)

    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

MODEL_CHECKPOINT = "./models/whisper-small.en-til-24/"

class ASRManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ASR] Device: {self.device}")

        self.processor = WhisperProcessor.from_pretrained(MODEL_CHECKPOINT)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT).to(self.device)
        
        # Suppress tokens for numeric values, to get the equivalent word spelled out (e.g. "123" -> "one two three")
        tokenizer=self.processor.tokenizer
        number_tokens = [i for i in range(tokenizer.vocab_size) if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))]
        self.gen_config = GenerationConfig.from_pretrained(MODEL_CHECKPOINT)
        self.gen_config.suppress_tokens += number_tokens

    def transcribe(self, audio_bytes: bytes) -> str:
        audio = load_audio(audio_bytes)
        input_features = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        generated_ids = self.model.generate(input_features, generation_config=self.gen_config)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]
