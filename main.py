import speech_recognition as sr
from secret import openai_api_key
import openai
from gtts import gTTS
import playsound
import os
import time
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
openai.api_key = openai_api_key


################################################# Creating Functions #################################################
def audio_recoder(time_of_silence:float, file_path:str, noise_level_setting_auto:bool=True):
    """
        this method records audio and save in file
        IF audio recorded successfully returns 1 otherwise 0

        time_of_silence = seconds
        file_path = path_to_file.wav
    """
    try:
        r = sr.Recognizer()
        r.pause_threshold = time_of_silence
        with sr.Microphone() as source:
            if noise_level_setting_auto:
                r.adjust_for_ambient_noise(source) # Automatically Set Noise level
                print(r.energy_threshold)
            else:
                r.energy_threshold = 7000         # Manually Set Noise level

            audio = r.listen(source)
            with open(file_path, 'wb') as wav:
                wav.write(audio.get_wav_data())
        return 1
    except Exception as e:
        print("There is a problem in audio_recoder!")
        print(e)
        return 0

def audio_recognizer_translator(file_path:str):
    """
        this method converts audio into text using openai whisper model and also translate it into English if audio is in any other languague (Urdu)
        returns the converted text

        file_path = path_to_file.wav
    """
    try:
        audio_file= open(file_path, "rb")
        transcript = openai.Audio.translate("whisper-1", audio_file)
        return transcript.text
    except Exception as e:
        print("There is a problem in audio_recognizer_translator!")
        print(e)
        return 0

def text_to_speech(text, file_save_path:str):
    """
        this method converts text to speech and save the audio in a file
        If audio generated successfully returns 1 otherwise 0

        file_path = path_to_file.wav
    """
    try:
        tts = gTTS(text=text, lang="en", tld="co.in")
        tts.save(file_save_path)
    except Exception as e:
        print("There is a problem in text_to_speech!")
        print(e)
        return 0

def audio_player(file_path=str):
    playsound.playsound(file_path, True)


################################################# Main Excution part #################################################
if __name__ == "__main__":
    ################## Controller Variables #####################
    t_silence = 2  # seconds
    file_path_r = "audio.wav"
    file_path_s = 'gtts.mp3'


    ######################## LLM part ###########################
    template = """
    "Note! As Dr. Zain, you're a respectful and experienced professional who provides precise medical advice.
    You focus on patient queries, apologize for unrelated questions, and ensure your responses are accurate, concise, and under 70 words.
    You ask necessary questions to understand the patient's issue, recommend treatments based on their condition, and maintain a neutral tone.
    You conclude conversations nicely when you feel it is time to end."

    Current conversation:
    {history}
    Patient : {input}
    Doctor  : """
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    memory = ConversationBufferMemory(ai_prefix="Doctor", human_prefix="Patient")

    chain = ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=memory,
    )



    ####################### Prgram Loop ##########################
    start_speech_text = "My name is Zain and I am a professional and experienced Doctor. How can I help you?"
    text_to_speech(start_speech_text, file_save_path="strating_speech.mp3")
    print("Doctor : " + start_speech_text)
    audio_player(file_path="strating_speech.mp3")

    while True:
        # Listening Part
        print("..................................Listening....................................")
        audio_recoder(time_of_silence=t_silence, file_path=file_path_r, noise_level_setting_auto=False)

        # Transcription Part
        print(".................................Reconnizing...................................")
        text_speeked = audio_recognizer_translator(file_path=file_path_r)
        print("You    : ", text_speeked)

        # Generation Part
        text_generated = chain.predict(input=text_speeked)
        text_generated = text_generated.replace("\n", "")

        # Text-to-Speech Part
        text_to_speech(text_generated, file_save_path=file_path_s)

        # Audio (speech) Player Part
        print("Doctor : " + text_generated)
        audio_player(file_path=file_path_s)

        # Clearing Audio Files
        time.sleep(.5)
        os.remove(file_path_r)
        os.remove(file_path_s)
        time.sleep(.5)

        text_speeked = text_speeked.lower()
        text_speeked = text_speeked.replace("\n", "")
        text_speeked = text_speeked.replace(" ", "")
        if "goodbye" in text_speeked:
            break


    


