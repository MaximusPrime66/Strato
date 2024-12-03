from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback
import io
import base64
import soundfile as sf
from speechbrain.pretrained import Tacotron2, HIFIGAN
from pydantic import BaseModel

# FastAPI application instance
app = FastAPI()

# Add CORS middleware to allow all origins for simplicity (you can restrict it to specific domains later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. You can replace "*" with specific domains for more security.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Global variable to hold the Tacotron2 model
tacotron_model = None
hifi_gan_model = None

# Model initialization
def init_model():
    global tacotron_model, hifi_gan_model
    try:
        # Load Tacotron2 and HIFIGAN models
        tacotron_model = Tacotron2.from_hparams(
            source="speechbrain/tts-tacotron2-ljspeech", savedir="pretrained_model"
        )
        hifi_gan_model = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_model"
        )
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Failed to load models: {e}")
        traceback.print_exc()

# Initialize models on startup
init_model()

# Pydantic model for the request body
class SpeechRequest(BaseModel):
    text: str
    voice: str = "default"  # Add the voice field with default value

# Endpoint for synthesizing speech
@app.post("/synthesize")
async def synthesize_speech(request: SpeechRequest):
    try:
        # Validate the input text
        if not request.text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        print(f"Text to synthesize: {request.text}")
        
        # Synthesize speech
        result = tacotron_model.encode_text(request.text)
        print(f"Result from encode_text: {result}")  # Debugging: print the result

        # Handle the result from tacotron_model.encode_text
        if isinstance(result, tuple) and len(result) == 3:
            mel_output, mel_length, alignment = result
        else:
            mel_output = result
            mel_length = None
            alignment = None
        
        # Decode the mel spectrogram with HIFIGAN
        waveform = hifi_gan_model.decode_batch(mel_output)  # Only get waveform

        # Convert waveform to wav format
        buffer = io.BytesIO()
        sf.write(buffer, waveform.squeeze().cpu().numpy(), 22050, format="wav")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Return audio as base64-encoded string
        return {"audio": audio_base64}
    
    except Exception as e:
        print(f"Speech synthesis error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")
