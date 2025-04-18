from faster_whisper import WhisperModel

model_size = "small"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe("kobieta.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))