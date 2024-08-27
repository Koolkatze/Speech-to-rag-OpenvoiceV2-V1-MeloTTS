# Speech-to-rag-OpenvoiceV2-V1-MeloTTS
Speech to Speech with RAG you can use any version of Openvoice in much more languajes even on version 1 with my code. Just implemented all and works perfectly (Don´t use a virtual envoirment if you want it to run quick).

REQUIREMENTS:

Windows 10/11
Python 3.10 https://www.python.org/downloads/release/python-3100/
CUDA Toolkit 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64 Select windows version and .exe(local).
ffmpeg installed https://phoenixnap.com/kb/ffmpeg-windows or pip install ffmpeg
NVIDIA GPU (will prob work with only CPU too)
microphone
local LLM setup (default is LM studio but working on OLlama to use WEB UI)
You might need Pytorch (https://pytorch.org/) (Included in HOW TO INSTALL)
If an ERROR like this occurs: "Could not load library cudnn_ops_infer64_8.dll. Error code 126" Please make sure cudnn_ops_infer64_8.dll is in your library path!" go to https://github.com/Purfview/whisper-standalone-win/releases/tag/libs download "cuBLAS.and.cuDNN_CUDA11_win_v2.zip take all the files inside .zip (.dll) and move them to your PC's C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
If an ERROR like this occurs: "Could not load library cublas64_12.dll.": Go to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin, take cublas64_11.dll make a copy of it and rename it cublas64_12.dll

HOW TO INSTALL:

0. Use Allways Windosws PowerShell Terminal

1. pip install ffmpeg
2. git clone https://github.com/Koolkatze/Speech-to-rag-OpenvoiceV2-V1-MeloTTS.git
3. cd Speech-to-rag-OpenvoiceV2-V1-MeloTTS
4. pip install -r requirements.txt
5. pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
6. mkdir modules
7. cd modules
8. download MeloTTS from https://github.com/myshell-ai/MeloTTS.git
9. download OpenVoice from https://github.com/myshell-ai/OpenVoice.git
10. cd MeloTTS
11. pip install -r requirements.txt
12. pip install -e .
13. python -m unidic download
14. download desired languages for MeloTTS from https://huggingface.co/myshell-ai
15. extract folder to Speech-to-rag-OpenvoiceV2-V1-MeloTTS/modules/MeloTTS
16. cd Speech-to-rag-OpenvoiceV2-V1-MeloTTS/modules/Openvoice
17. pip install requirements.txt
18. pip install -e .
19. download checkpoints from https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
20. extract .zip to Speech-to-rag-OpenvoiceV2-V1-MeloTTS/modules/Openvoice
21. In voice.py set your reference PATHs 
22. OpenVoice/checkpoints_v2 on line 34
23. Your-Voice-to-Clone.mp3 on line 281
24. Your path to modules/OpenVoice/checkpoints_v2/base_speakers/ses/ on line 307
25. Your-Voice-to-Clone.mp3 on line 326
26. Your path to modules/OpenVoice/checkpoints_v2/base_speakers/ses/any_of_existent_accents.pth on line 328
27. Your path to modules/MeloTTS/MeloTTS-[prefered language]/checkpoint.pth on line 357
29. Your path to modules/MeloTTS/MeloTTS-[prefered language]/config.json on line 357

30. start LM studio server (or similar) in your PC. you can change to other program by substituting its localhost in: http://localhost:1234/v1 line 28 for any other local LLM host.
31. Edit chatbot1.txt to create a Chat's Character personality.
32. Edit vault.txt to create Chats Knowledge about yourself (or user).
34. Edit main.py and add a # before vs.user_chatbot_conversation() to turn off Openvoice-v1 or add a # vs.user_chatbot_conversation2() to turn off OpenVoice-v2
35. run: python main.py

You can see how much time each one lasts and use the one that suits better your needs.

All OpenVoices will clone the reference voice you introduce at start.

ROADMAP:

pip install ffmpeg
Run a docker in the program to try and use LMStudio instead of OLlama (possible alternative).
Change LMStudio for OLlama to use its Web UI.
Read OLlamas output or Chatbot's answer inside OLlama and stream the text string to Frames by Brilliant Labs by using Brilliant Labs NOA Assistant and OLlama Web UI sharing the string info.
Using all the sensors inside Frames by Brilliant Labs (Camera, Movement/Gravity, Tap Buttons) to control and share info with OLlama and enhance the chatting experience.
Implementing video stream through the glasses camera to the preferred LLM inside OLlama or LMStudio (with Docker) to make a ChatGPT type of chatting with any opensource model.
