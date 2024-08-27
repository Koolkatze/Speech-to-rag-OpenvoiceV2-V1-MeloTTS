import pyaudio
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from sentence_transformers import SentenceTransformer, util
import argparse
import wave
from zipfile import ZipFile
import langid
import openai
from openai import OpenAI
import time
import speech_recognition as sr
from faster_whisper import WhisperModel
import os
import pygame

PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

model_size = "medium"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
# Define the name of the log file
chat_log_filename = "chatbot_conversation_log.txt"
class VoiceService:

    def __init__(self):
        self._ckpt_converter = 'modules/OpenVoice/checkpoints_v2/converter'
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._output_dir = 'outputs_v2'

        self._tone_color_converter = ToneColorConverter(f'{self._ckpt_converter}/config.json', device=self._device)
        self._tone_color_converter.load_ckpt(f'{self._ckpt_converter}/checkpoint.pth')

        os.makedirs(self._output_dir, exist_ok=True)

    def open_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def get_relevant_context(self, user_input, vault_embeddings, vault_content, model, top_k=3):
        """
        Retrieves the top-k most relevant context from the vault based on the user input.
        """
        if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
            return []

        # Encode the user input
        input_embedding = model.encode([user_input])
        # Compute cosine similarity between the input and vault embeddings
        cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
        # Adjust top_k if it's greater than the number of available scores
        top_k = min(top_k, len(cos_scores))
        # Sort the scores and get the top-k indices
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
        # Get the corresponding context from the vault
        relevant_context = [vault_content[idx].strip() for idx in top_indices]
        return relevant_context

    def chatgpt_streamed(self, user_input, system_message, conversation_history, bot_name, vault_embeddings,
                         vault_content, model):
        """
        Function to send a query to OpenAI's GPT-3.5-Turbo model, stream the response, and print each full line in yellow color.
        """
        # Get relevant context from the vault
        relevant_context = self.get_relevant_context(user_input, vault_embeddings, vault_content, model)
        # Concatenate the relevant context with the user's input
        user_input_with_context = user_input
        if relevant_context:
            user_input_with_context = "\n".join(relevant_context) + "\n\n" + user_input
        messages = [{"role": "system", "content": system_message}] + conversation_history + [
            {"role": "user", "content": user_input_with_context}]
        temperature = 1
        streamed_completion = client.chat.completions.create(
            model="local-model",
            messages=messages,
            stream=True
        )
        self.full_response = ""
        line_buffer = ""
        for chunk in streamed_completion:
            delta_content = chunk.choices[0].delta.content
            if delta_content is not None:
                line_buffer += delta_content
                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]:
                        print(NEON_GREEN + line + RESET_COLOR)
                        self.full_response += line + '\n'
                    line_buffer = lines[-1]
        if line_buffer:
            print(NEON_GREEN + line_buffer + RESET_COLOR)
            self.full_response += line_buffer
        return self.full_response

    def transcribe_with_whisper(self, audio_file_path):
        # Load the model
        segments, info = whisper_model.transcribe(audio_file_path, beam_size=5)
        self.transcription = ""
        for segment in segments:
            self.transcription += segment.text + " "
        # Transcribe the audio
        self.result = self.transcription.strip()
        return self.transcription.strip()

    # Function to record audio from the microphone and save to a file
    def record_audio(self, file_path):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []

        print("Recording...")

        try:
            while True:
                data = stream.read(1024)
                frames.append(data)
        except KeyboardInterrupt:
            pass

        print("Recording stopped.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()

    def user_chatbot_conversation(self):
        conversation_history = []
        system_message = self.open_file("chatbot1.txt")
        # Load the sentence transformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Load the initial content from the vault.txt file
        vault_content = []
        if os.path.exists("vault.txt"):
            with open("vault.txt", "r", encoding="utf-8") as vault_file:
                vault_content = vault_file.readlines()
        # Create embeddings for the initial vault content
        vault_embeddings = model.encode(vault_content) if vault_content else []
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        while True:
            audio_file = "temp_recording.wav"
            self.record_audio(audio_file)
            user_input = self.transcribe_with_whisper(audio_file)
            os.remove(audio_file)  # Clean up the temporary audio file
            if user_input.lower() == "exit":  # Say 'exit' to end the conversation
                break
            elif user_input.lower().startswith(
                    ("print info", "Print info")):  # Print the contents of the vault.txt file
                print("Info contents:")
                if os.path.exists("vault.txt"):
                    with open("vault.txt", "r", encoding="utf-8") as vault_file:
                        print(NEON_GREEN + vault_file.read() + RESET_COLOR)
                else:
                    print("Info is empty.")
                continue
            elif user_input.lower().startswith(("delete info", "Delete info")):  # Delete the vault.txt file
                confirm = input("Are you sure? Say 'Yes' to confirm: ")
                if confirm.lower() == "yes":
                    if os.path.exists("vault.txt"):
                        os.remove("vault.txt")
                        print("Info deleted.")
                        vault_content = []
                        vault_embeddings = []
                        vault_embeddings_tensor = torch.tensor(vault_embeddings)
                    else:
                        print("Info is already empty.")
                else:
                    print("Info deletion cancelled.")
                continue
            elif user_input.lower().startswith(("insert info", "insert info")):
                print("Recording for info...")
                audio_file = "vault_recording.wav"
                record_audio(audio_file)
                vault_input = transcribe_with_whisper(audio_file)
                os.remove(audio_file)  # Clean up the temporary audio file
                with open("vault.txt", "a", encoding="utf-8") as vault_file:
                    vault_file.write(vault_input + "\n")
                print("Wrote to info.")
                # Update the vault content and embeddings
                vault_content = open("vault.txt", "r", encoding="utf-8").readlines()
                vault_embeddings = model.encode(vault_content)
                vault_embeddings_tensor = torch.tensor(vault_embeddings)
                continue
            print(CYAN + "Tú:", user_input + RESET_COLOR)
            conversation_history.append({"role": "user", "content": user_input})
            print(PINK + "Glitch:" + RESET_COLOR)
            self._chatbot_response = self.chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot", vault_embeddings_tensor, vault_content, model)
            conversation_history.append({"role": "assistant", "content": self._chatbot_response})
            self._prompt1 = self._chatbot_response
            start_time = time.time()
            VoiceService.openvoice(self, " ") # FOR CLONED SIMPLE VOICE
            end_time = time.time()
            print(f"Openvoice Execution Time: {end_time - start_time}")
            print("OpenVoice ended")
            # VoiceService.openvoice_v2(self)
            # VoiceService.melotts2(self, " ") # FOR SIMPLE VOICE

    def user_chatbot_conversation2(self):
        conversation_history = []
        system_message = self.open_file("chatbot1.txt")
        # Load the sentence transformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Load the initial content from the vault.txt file
        vault_content = []
        if os.path.exists("vault.txt"):
            with open("vault.txt", "r", encoding="utf-8") as vault_file:
                vault_content = vault_file.readlines()
        # Create embeddings for the initial vault content
        vault_embeddings = model.encode(vault_content) if vault_content else []
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        while True:
            audio_file = "temp_recording.wav"
            self.record_audio(audio_file)
            user_input = self.transcribe_with_whisper(audio_file)
            os.remove(audio_file)  # Clean up the temporary audio file
            if user_input.lower() == "exit":  # Say 'exit' to end the conversation
                break
            elif user_input.lower().startswith(
                    ("print info", "Print info")):  # Print the contents of the vault.txt file
                print("Info contents:")
                if os.path.exists("vault.txt"):
                    with open("vault.txt", "r", encoding="utf-8") as vault_file:
                        print(NEON_GREEN + vault_file.read() + RESET_COLOR)
                else:
                    print("Info is empty.")
                continue
            elif user_input.lower().startswith(("delete info", "Delete info")):  # Delete the vault.txt file
                confirm = input("Are you sure? Say 'Yes' to confirm: ")
                if confirm.lower() == "yes":
                    if os.path.exists("vault.txt"):
                        os.remove("vault.txt")
                        print("Info deleted.")
                        vault_content = []
                        vault_embeddings = []
                        vault_embeddings_tensor = torch.tensor(vault_embeddings)
                    else:
                        print("Info is already empty.")
                else:
                    print("Info deletion cancelled.")
                continue
            elif user_input.lower().startswith(("insert info", "insert info")):
                print("Recording for info...")
                audio_file = "vault_recording.wav"
                record_audio(audio_file)
                vault_input = transcribe_with_whisper(audio_file)
                os.remove(audio_file)  # Clean up the temporary audio file
                with open("vault.txt", "a", encoding="utf-8") as vault_file:
                    vault_file.write(vault_input + "\n")
                print("Wrote to info.")
                # Update the vault content and embeddings
                vault_content = open("vault.txt", "r", encoding="utf-8").readlines()
                vault_embeddings = model.encode(vault_content)
                vault_embeddings_tensor = torch.tensor(vault_embeddings)
                continue
            print(CYAN + "Tú:", user_input + RESET_COLOR)
            conversation_history.append({"role": "user", "content": user_input})
            print(PINK + "Glitch:" + RESET_COLOR)
            self._chatbot_response2 = self.chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot", vault_embeddings_tensor, vault_content, model)
            conversation_history.append({"role": "assistant", "content": self._chatbot_response2})
            self._prompt2 = self.full_response
            #self._chatbot_response2
            start_time = time.time()
            VoiceService.openvoice_v2(self)
            end_time = time.time()
            print(f"OpenVoiceV2 Execution Time: {end_time - start_time}")
            print("OpenVoiceV2 ended")

    def openvoice_v2(self):
        reference_speaker = 'modules/OpenVoice/resources/glitch2.mp3'  # This is the voice you want to clone
        target_se, audio_name = se_extractor.get_se(reference_speaker, self._tone_color_converter, vad=True)

        texts = {
            #'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
            #'EN': "Did you ever hear a folk tale about a giant turtle?",
            'ES': f"{self._prompt2}"
            #'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
            #'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
            #'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
            #/*'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
        }

        src_path = f'{self._output_dir}/tmp.wav'

        # Speed is adjustable
        speed = 1.3

        for language, text in texts.items():
            model = TTS(language='ES', device=self._device)
            speaker_ids = model.hps.data.spk2id

            for speaker_key in speaker_ids.keys():
                speaker_id = speaker_ids[speaker_key]
                speaker_key = speaker_key.lower().replace('_', '-')
                print(speaker_key)
                source_se = torch.load(f'modules/OpenVoice/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=self._device)
                model.tts_to_file(text, 3, src_path, speed=speed)
                save_path = f'{self._output_dir}/output_v2_{speaker_key}and{language}.wav'

                # Run the tone color converter
                encode_message = "@MyShell"
                self._tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=save_path,
                    message=encode_message)
                
                play_audio =  self.play(save_path)
                play_audio
                # hear_me = VoiceService.record_audio()
                  
        #if len(self._conversation_history) > 20:
           # self._conversation_history = self._conversation_history[-20:]

    def openvoice(self, text):

        reference_speaker = 'modules/OpenVoice/resources/glitch2.wav'  # This is the voice you want to clone
        target_se, audio_name = se_extractor.get_se(reference_speaker, self._tone_color_converter, vad=True)
        source_se = torch.load(f'modules/OpenVoice/checkpoints_v2/base_speakers/ses/es.pth',
                               map_location=self._device)

        save_path = f'{self._output_dir}/output.mp3'

        src_path = self.melotts2(text, standalone=False)

        # Run the tone color converter
        encode_message = "@MyShell"
        self._tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message)
        self.play(save_path)
    def melotts2(self,text, standalone=True):

        texts = {
           # 'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
           # 'EN': "Did you ever hear a folk tale about a giant turtle?",
            'ES': f"{self._prompt1}"
           # 'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
           # 'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
           # 'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
           # 'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
        }

        for language, text in texts.items():
            model = TTS(language="ES", device=self._device, ckpt_path="modules/MeloTTS/MeloTTS-Spanish/checkpoint.pth", config_path="modules/MeloTTS/MeloTTS-Spanish/config.json")

        src_path = f'{self._output_dir}/tmp.wav'

        # Speed is adjustable
        speed = 1.3
        model.tts_to_file(text, 3, src_path, speed=speed)

        return self.play(src_path) if standalone else src_path

    def play(self, temp_audio_file):

        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        # os.remove(temp_audio_file)

# Start the conversation in main