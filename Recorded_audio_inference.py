import whisper
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
from scipy.io.wavfile import write
from IPython.display import Audio, display
from tensorflow_tts.inference import AutoProcessor, TFAutoModel
import argparse
import os

# Step 1: ASR with Whisper (German/French to English transcription)
def asr_with_whisper(audio_file_path, tflite_model_path, language="de"):
    # Load TFLite model with GPU delegate
    try:
        from tensorflow.lite.python.interpreter import load_delegate
        gpu_delegate_path = './libtensorflowlite_gpu_delegate.so'

        if not os.path.exists(gpu_delegate_path):
            raise FileNotFoundError(f"GPU delegate not found at path: {gpu_delegate_path}")

        gpu_delegate = load_delegate(gpu_delegate_path, {
            'inference_preference': 2,  # Fast inference mode
        })

        interpreter = tf.lite.Interpreter(
            model_path=tflite_model_path,
            experimental_delegates=[gpu_delegate]
        )
        print("GPU delegate enabled.")
    except Exception as e:
        print(f"Error loading GPU delegate: {e}")
        print("Falling back to CPU.")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]['index']
    output_tensor = interpreter.get_output_details()[0]['index']

    # Create a tokenizer for Whisper (German or French language for input)
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language=language)

    # Start ASR inference
    print(f'Calculating mel spectrogram from {audio_file_path}...')
    mel_from_file = whisper.audio.log_mel_spectrogram(audio_file_path)
    input_data = whisper.audio.pad_or_trim(mel_from_file, whisper.audio.N_FRAMES)
    input_data = np.expand_dims(input_data, 0)

    print("Invoking interpreter for ASR ...")
    interpreter.set_tensor(input_tensor, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_tensor)

    # Convert tokens to text
    print("Converting tokens ...")
    transcribed_text = ""
    for token in output_data:
        token[token == -100] = wtokenizer.eot
        transcribed_text = wtokenizer.decode(token)
    print("Transcribed text:", transcribed_text)

    return transcribed_text

# Step 2: TTS with FastSpeech (English)
def tts_with_fastspeech(transcribed_text):
    # Initialize TTS models for English
    melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")
    fastspeech = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech-ljspeech-en")
    processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech-ljspeech-en")

    # Prepare input text for TTS
    input_text_tts = transcribed_text.strip()
    if not input_text_tts:
        print("Transcribed text is empty. Using default text.")
        input_text_tts = "This is a default fallback text."

    # Convert text to sequence
    input_ids = processor.text_to_sequence(input_text_tts.lower())
    input_ids_tensor = tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0)

    speaker_ids = tf.convert_to_tensor([0], dtype=tf.int32)
    speed_ratios = tf.convert_to_tensor([1.0], dtype=tf.float32)

    # Generate audio
    print("Generating audio using TTS ...")
    try:
        mel_before, mel_after, duration_outputs = fastspeech.inference(
            input_ids=input_ids_tensor,
            speaker_ids=speaker_ids,
            speed_ratios=speed_ratios,
        )
        audio_after = melgan(mel_after)[0, :, 0]

        # Play audio
        display(Audio(data=audio_after.numpy(), rate=22050))

        # Save audio as WAV file
        output_audio_path = "output_audio.wav"
        write(output_audio_path, 22050, audio_after.numpy())
        print(f"Audio saved as '{output_audio_path}'")

    except Exception as e:
        print(f"An error occurred during TTS inference: {e}")

# Main execution with Command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio transcription and synthesis.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (MP3 or WAV)")
    parser.add_argument("language", type=str, choices=["de", "fr"], help="Language of the audio file: 'de' for German, 'fr' for French")

    args = parser.parse_args()

    audio_file_path = args.audio_file
    language = args.language
    tflite_model_path = './models/whisper-base.tflite'

    # ASR inference (German/French to English transcription)
    start_time = timer()
    transcribed_text = asr_with_whisper(audio_file_path, tflite_model_path, language)
    print(f"ASR Inference took {timer() - start_time:.2f} seconds")

    # TTS inference (English speech generation)
    start_time = timer()
    tts_with_fastspeech(transcribed_text)
    print(f"TTS Inference took {timer() - start_time:.2f} seconds")
