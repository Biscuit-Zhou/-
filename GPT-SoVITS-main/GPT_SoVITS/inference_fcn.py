import os
import soundfile as sf
from inference_webui import get_tts_wav

output_path = 'output'

def synthesize(target_text, number):
    ref_texts = {
        1: ("这…应该说西风教会的修女见多识广吗…", "ref2/n1.wav"), 
        2: ("居然是这样！他真的骗了我们，那个大坏蛋！", "ref2/n2.wav"),
        3: ("对不起，把麻烦引到了你的身上。", "ref2/n3.wav"),
        4: ("那就是阿贝多吗？为什么他会待在这种地方？", "ref2/n4.wav"), 
        5: ("哼哼，你已经吓不住我们啦！才不是那个味道呢！", "ref2/n5.wav"), 
        6: ("唔…不是我们要找的东西，白高兴一场。", "ref2/n6.wav")
    }
    if number in ref_texts:
        ref_text, ref_audio_path = ref_texts[number]
    else:
        raise ValueError(f"Invalid reference number {number}. Please provide a valid number.")
    # Synthesize audio
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path, 
                                   prompt_text=ref_text, 
                                   prompt_language='中文', 
                                   text=target_text, 
                                   text_language='中文', top_p=1, temperature=1)
    
    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")
