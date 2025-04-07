import socket
from concurrent.futures import ThreadPoolExecutor
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from GPT_SoVITS.inference_fcn import synthesize
from openai import OpenAI
import os
from dashscope.audio.asr import *
import dashscope
from http import HTTPStatus
import time
import librosa
import soundfile


# 配置LLM的API
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# TTS模型预加载
GPT_model_path = 'GPT_weights\paimon_e20.ckpt'
SoVITS_model_path = 'SoVITS_weights\paimon_e100_s121900.pth'
output_path = '.\output'
print("Loading GPT weights and SoVits weights.")
change_gpt_weights(gpt_path=GPT_model_path)
change_sovits_weights(sovits_path=SoVITS_model_path)
print("Loaded.")
# 第一次语音生成为生成缓存
synthesize("这是第一段测试文本", 1)
print("预启动已完成")

# LLM情绪识别
def predict_emotion(text):
    try:
        completion = client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': '请你作为一个情绪分析助手，当用户输入一段话时，请根据这段话的内容识别其中蕴含的最主要情绪，并从所给的情绪词语中用一个最恰当的情绪词语来描述它。现在有如下情绪词语：害怕、生气、失落、好奇、戏谑、悲伤。这些词语分别对应数字1、2、3、4、5、6。请确保只返回一个最能代表这段话情绪的词语的对应数字。示例：- 用户输入：“我刚刚赢得了比赛！”- 输出：“4” 现在，请对以下内容进行分析：'},
                {'role': 'user', 'content': text}
                ]
        )
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"错误信息：{e}")
    return completion.choices[0].message.content

# 生成发送内容
def send_voice(sentence, senti):
    print("Generating Voice.")
    synthesize(sentence, senti) # 该方法调用的get_tts_wav修改speed=1为0.9
    with open('./output/output.wav', 'rb') as f:
        senddata = f.read()
    senddata += b'?!'
    senddata += b'%i' % senti
    return senddata

# 断句
def is_sentence_end(char):
    return char in ['。', '？', '！', '……']

# 在线ASR
dashscope.api_key = "sk-xxx"
def _receive_file(client_socket):
    file_data = b''
    while True:
        data = client_socket.recv(1024)
        # print(data)
        client_socket.send(b'sb')
        if data[-2:] == b'?!':
            file_data += data[0:-2]
            break
        if not data:
            print('Waiting for WAV...')
            continue
        file_data += data
    return file_data
def fill_size_wav():
    with open('input.wav', "r+b") as f:
        # Get the size of the file
        size = os.path.getsize('input.wav') - 8
        # Write the size of the file to the first 4 bytes
        f.seek(4)
        f.write(size.to_bytes(4, byteorder='little'))
        f.seek(40)
        f.write((size - 28).to_bytes(4, byteorder='little'))
        f.flush()
def process_voice():
    # stereo to mono
    fill_size_wav()
    y, sr = librosa.load('input.wav', sr=None, mono=False)
    y_mono = librosa.to_mono(y)
    y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)
    soundfile.write('input.wav', y_mono, 16000)
    recognition = Recognition(model='paraformer-realtime-v2',
                          format='wav',
                          sample_rate=16000,
                          # “language_hints”只支持paraformer-realtime-v2模型
                          language_hints=['zh'],
                          callback=None)
    result = recognition.call('input.wav')
    if result.status_code == HTTPStatus.OK:
        # 调用 get_sentence 方法并处理返回的结果
        sentences = result.get_sentence()
        if isinstance(sentences, list) and len(sentences) > 0:
            sentence = sentences[0]
            text = sentence.get('text', '')
            print(text)
            return text
        else:
            print('Error: Result does not contain any sentences.')

def handle_client_connection(client_socket):
    try:
        with open('./paimon4.txt', 'r', encoding='utf-8') as pf:
            sys_prompt = pf.read()
        messages = [{'role': 'system', 'content': sys_prompt}]
        while True:
            file = _receive_file(client_socket)
            with open('input.wav', 'wb') as f:
                f.write(file)
                print('WAV file received and saved.')
            ask_text = process_voice()
            print("用户问题: ", ask_text)
            messages.append({"role": "user", "content": ask_text})
            # print("发送前的message: ", messages)
            response = client.chat.completions.create(
                model="qwen-plus",  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
                )
            sentence = ""
            assistant_output = ""
            for chunk in response:
                if chunk.choices[0].finish_reason == 'stop':
                    messages.append({"role": "assistant", "content": assistant_output})
                    print("回复记录: ", assistant_output)
                    time.sleep(3)
                    client_socket.sendall(b'stream_finished')
                    print("已发送stream_finished")
                    break
                inform = chunk.choices[0].delta
                content = inform.content
                content = content.replace('\n', '').replace('-', '').replace('*', '').replace(' ', '')
                sentence += content
                assistant_output += content
                while True:
                    sentence_end_index = -1
                    for i, char in enumerate(sentence):
                        if is_sentence_end(char):
                            sentence_end_index = i
                            break
                    if sentence_end_index != -1:
                        emotion = int(predict_emotion(sentence[:sentence_end_index+1]))
                        senddata = send_voice(sentence[:sentence_end_index+1], emotion)
                        client_socket.sendall(senddata)
                        sentence = sentence[sentence_end_index+1:]
                    else:
                        break
            # print("所有聊天记录: ", messages)
    except Exception as e:
        print(f"Error handling client connection: {e}")
    finally:
        client_socket.close()

host = socket.gethostbyname(socket.gethostname())
def start_server(host=host, port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10240000)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")
    # 多线程连接
    with ThreadPoolExecutor(max_workers=5) as executor:
        try:
            while True:
                client_socket, addr = server_socket.accept()
                print(f"Accepted connection from {addr}")
                temp_char_name = 'character_paimon'
                client_socket.sendall(b'%s' % temp_char_name.encode())
                executor.submit(handle_client_connection, client_socket)
        except KeyboardInterrupt:
            print("Server is shutting down.")
        finally:
            server_socket.close()

if __name__ == '__main__':
    start_server()
