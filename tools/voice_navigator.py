import pyttsx3


def test_pyttsx3(text):
    try:
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Set properties (optional)
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

        # Test phrase
        test_text = "Hello, this is a pyttsx3 test."

        # Say the test phrase
        engine.say(text)

        # Wait for the speech to finish
        engine.runAndWait()

        print("Test completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


