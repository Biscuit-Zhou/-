# 是派蒙！（《数字生命》服务端）

## 🚀 项目简介

本项目是对[极客湾《数字生命》](https://github.com/zixiiu/Digital_Life_Server)服务端的深度二次开发，旨在优化原有功能并减少计算耗时。通过重构代码结构、增强使用性能和提升兼容性，为体验《数字生命》提供更高效的支持。

本项目需搭配[《数字生命》原版客户端](https://github.com/QSWWLTN/DigitalLife)使用

## 🛠️ 快速开始

**测试通过的环境**
- Python 3.9，PyTorch 2.3.0，CUDA 12.1，Windows 10
- Python 3.9，PyTorch 2.3.0，CPU，Windows 10
- Python 3.10，PyTorch 2.5.1，CPU，Ubuntu 22.04 LTS

### 0. 准备

强烈建议使用带 CUDA 11.8 及以上的设备以加速语音合成，项目使用大模型服务平台百炼，可根据需要替换为其他厂商或自行部署本地推理

### 1: 部署

0. 安装Miniconda3（可选）（推荐）

1. 安装[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/tree/20240821v2?tab=readme-ov-file) （测试版本均为v2），按照其项目附带的README文档进行部署安装

2. 将本项目与GPT-SoVITS文件夹合并

3. 补充安装第三方库

```bash
    cd GPT-SoVITS-main
    pip install -r requirements_add.txt
```

### 2: 使用

1. 修改 setup_gpu.py

- line 18  配置LLM的API
> 当前选用 qwen-plus
 
- line 65  配置在线ASR的API
> 当前选用 paraformer-realtime-v2

2. 激活你的工作区（例如conda环境）

3. 运行项目文件

```bash
    python setup_gpu.py
```
显示

### 注意

Linux下如果出现键值'中文'报错，在GPT_SoVITS的inference_webui.py文件get_tts_wav函数中，将这两行代码

```bash
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
```

修改为

```bash
    prompt_language = i18n(prompt_language)
    prompt_language = dict_language[prompt_language]
    text_language = i18n(text_language)
    text_language = dict_language[text_language]
```

## 📢 联系我们
- 作者邮箱：2760933054@qq.com
