"""Main beam handler to do tacotron2 inference"""

from io import BytesIO
import base64
import torch
import numpy as np
from scipy.io.wavfile import write

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

device = "cuda"
dst_dir = './vol/'
torch.hub.set_dir(dst_dir)

tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

def to_speech(text: str) -> str:
    """Convert text to audio using tacotron2.
    Args:
        text (str): text to turn to speech.

    Return:
        audio text
    """

    sequences, lengths = utils.prepare_input_sequence([text])
    with torch.no_grad():
        mel, _, _ = tacotron2.infer(sequences, lengths)

        mel = mel.unsqueeze(0)
        mel = torch.nn.functional.interpolate(mel, scale_factor=(1.0, 1.0 / 1.1), mode='bilinear', align_corners=False)
        mel = mel.squeeze(0)

        audio = waveglow.infer(mel)

    buffer_ = BytesIO()
    write(buffer_, 22050, audio[0].data.cpu().numpy())
    return {
        "audio": base64.b64encode(buffer_.getvalue()).decode("ascii")
    }
