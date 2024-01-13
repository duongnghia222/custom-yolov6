import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# Using the default microphone as the audio source
with sr.Microphone() as source:
    print("Please say something:")
    # Adjust the recognizer sensitivity to ambient noise
    r.adjust_for_ambient_noise(source, duration=3)
    # Listening for the first phrase and extracting it into audio data
    print("Now")
    audio = r.listen(source)

try:
    # Using Google Web Speech API to recognize audio
    print("Google Speech Recognition thinks you said:")
    print(r.recognize_google(audio))
except sr.UnknownValueError:
    # API was unable to understand the audio
    print("Google Speech Recognition could not understand the audio")
except sr.RequestError as e:
    # API was unreachable or unresponsive
    print(f"Could not request results from Google Speech Recognition service; {e}")


try:
    # Use Sphinx for offline speech recognition
    print("Sphinx thinks you said:")
    print(r.recognize_sphinx(audio))
except sr.UnknownValueError:
    print("Sphinx could not understand the audio")
except sr.RequestError as e:
    print(f"Sphinx error; {e}")