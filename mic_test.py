import speech_recognition as sr

r = sr.Recognizer()

try:
    with sr.Microphone(device_index=1) as source:
        print("Using Realtek Microphone Array...")
        print("Calibrating background noise...")
        r.adjust_for_ambient_noise(source, duration=2)

        print("Speak now: What is the time now?")
        audio = r.listen(source, timeout=10, phrase_time_limit=8)

    print("Recognizing...")
    result = r.recognize_google(audio, language="en-IN")
    print("SUCCESS:", result)

except Exception as e:
    print("ERROR:", repr(e))
