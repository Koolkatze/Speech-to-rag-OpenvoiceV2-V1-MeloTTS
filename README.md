# Speech-to-rag-OpenvoiceV2-V1-MeloTTS

LOCAL LLM, UNCENSORED, SPEECH-TO-SPEECH RAG, FREE AI WITH VOICE CLONING AND CHARACTER (personality) CREATION:

Speech to Speech with RAG you can use any version of Openvoice in much more languajes even on version 1 with my code. Just implemented all and works perfectly (Don´t use a virtual envoirment if you want it to run quick).

(This README is tested and works perfectly fine)

PATHS are inserted as relative PATHS not your "C:\" 

REQUIREMENTS:

1. Windows 10/11
2. Python 3.10 https://www.python.org/downloads/release/python-3100/
3. CUDA Toolkit 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64 Select windows version and .exe(local).
4. ffmpeg installed https://phoenixnap.com/kb/ffmpeg-windows or pip install ffmpeg
5.. NVIDIA GPU (will prob work with only CPU too)
6. microphone
7. local LLM setup (default is LM studio but working on OLlama to use WEB UI)
8. You might need Pytorch (https://pytorch.org/) (Included in HOW TO INSTALL)
9. If an ERROR like this occurs: "Could not load library cudnn_ops_infer64_8.dll. Error code 126" Please make sure cudnn_ops_infer64_8.dll is in your library path!" go to https://github.com/Purfview/whisper-standalone-win/releases/tag/libs download "cuBLAS.and.cuDNN_CUDA11_win_v2.zip take all the files inside .zip (.dll) and move them to your PC's C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
10. If an ERROR like this occurs: "Could not load library cublas64_12.dll.": Go to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin, take cublas64_11.dll make a copy of it and rename it cublas64_12.dll

HOW TO INSTALL:

0. Use Allways Windosws PowerShell Terminal
- /usr/local/bin/python -m pip install --upgrade pip
1. pip install ffmpeg
2. prepare yourself...
3. git clone https://github.com/Koolkatze/Speech-to-rag-OpenvoiceV2-V1-MeloTTS.git
4. cd Speech-to-rag-OpenvoiceV2-V1-MeloTTS
5. pip install -r requirements.txt
6. Take a shower...
7. mkdir modules
8. cd modules
9. download MeloTTS from https://github.com/myshell-ai/MeloTTS.git
10. download OpenVoice from https://github.com/myshell-ai/OpenVoice.git
11. cd MeloTTS
12. pip install -r requirements.txt
13. pip install -e .
14. python -m unidic download
15. download desired languages for MeloTTS from https://huggingface.co/myshell-ai
16. extract folder to Speech-to-rag-OpenvoiceV2-V1-MeloTTS/modules/MeloTTS
17. cd Speech-to-rag-OpenvoiceV2-V1-MeloTTS/modules/Openvoice
18. pip install -r requirements.txt
19. pip install -e .
20. download checkpoints from https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
21. extract .zip to Speech-to-rag-OpenvoiceV2-V1-MeloTTS/modules/Openvoice
22. pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
23. In voice.py set your reference PATHs 
24. OpenVoice/checkpoints_v2 on line 34
25. Your-Voice-to-Clone.mp3 on line 281
26. Your path to modules/OpenVoice/checkpoints_v2/base_speakers/ses/ on line 307
27. Your-Voice-to-Clone.mp3 on line 326
28. Your path to modules/OpenVoice/checkpoints_v2/base_speakers/ses/any_of_existent_accents.pth on line 328
29. Your path to modules/MeloTTS/MeloTTS-[prefered language]/checkpoint.pth on line 357
30. Your path to modules/MeloTTS/MeloTTS-[prefered language]/config.json on line 357
31. start LM studio server (or similar) in your PC. you can change to other program by substituting its localhost in: http://localhost:1234/v1 line 28 for any other local LLM host.
32. Edit chatbot1.txt to create a Chat's Character personality.
33. Edit vault.txt to create Chats Knowledge about yourself (or user).
34. Edit main.py and add a # before vs.user_chatbot_conversation() to turn off Openvoice-v1 or add a # vs.user_chatbot_conversation2() to turn off OpenVoice-v2
35. run: python main.py

You can see how much time each one lasts and use the one that suits better your needs.

All OpenVoices will clone the reference voice you introduce at start.

ROADMAP:

1. Change LMStudio for OLlama to use its Web UI or use a Docker file to use LMStudio from anywhere.
2. Read OLlamas output or Chatbot's answer inside OLlama and stream the text string to Frames by Brilliant Labs by using Brilliant Labs NOA Assistant and OLlama Web UI sharing the string info.
3. Using all the sensors inside Frames by Brilliant Labs (Camera, Movement/Gravity, Tap Buttons) to control and share info with OLlama and enhance the chatting experience.
4. Implementing video stream through the glasses camera to the preferred LLM inside OLlama or 5. LMStudio (with Docker) to make a ChatGPT type of chatting with any opensource model.
