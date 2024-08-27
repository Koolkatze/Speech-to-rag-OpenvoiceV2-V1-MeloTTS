from voice import VoiceService
import time

vs = VoiceService()
print("OpenVoice Starting...")
start_time = time.time()
#vs.openvoice_v2()
vs.user_chatbot_conversation()
end_time = time.time()
print(f"OpenVoice Execution Time: {end_time - start_time}")
print("OpenVoice_v2 ended")


