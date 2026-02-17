![image](./thumbnail.png)
An AI pipeline that tracks and reports what you do in front of your computer everyday. All data processed locally.

## 1. Setup
This setup guide is written for Windows platform. **For users using MacOS or other systems, please read "If you're using other OS..." before getting started**
#### 1.1 Clone the repo
```git clone https://github.com/Merxon22/FlowTrace.git```
#### 1.2 Setup virtual environment
1. This project is developed with **Python 3.12**. First, create a virtual environment with `python3 -m venv ./venv` in the project's root directory. Then, do `pip install -r requirements.txt`.
3. **IMPORTANT:** If you want to change the way this venv is created (e.g., use another name, or use anaconda), you also have to make changes in `./scripts/main.bat` and `./scripts/auto_screen_capture.bat`. Both batch files call the python program located in the virtual environment's folder, so make sure to adapt the path to your case.

#### 1.3 Install [llama server](https://github.com/ggml-org/llama.cpp)
1. For this part, primarily refer to [llama.cpp's GitHub](https://github.com/ggml-org/llama.cpp). On Windows, Llama Server can be easily installed using `winget install llama.cpp` (Reference [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md))
2. Verify your installation using `where llama-server`

#### 1.4 Choose your AI models
1. This project used the **Qwen3** model family, but you can choose any other VLM/LLM model that is in **[GGUF](https://en.wikipedia.org/wiki/Llama.cpp#GGUF_file_format)** format. Qwen3 was chosen because:
    1. **Great performance**
    2. **Has officially-released gguf versions:** more reliable, less hassle
    3. **Has models in different sizes:** most people can probably find the right model that runs on their machine.
2. This project uses the VLM (`Qwen3VL-4B-Instruct-Q4_K_M.gguf`) and its `mproj` file (`mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf`) found **[here](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/blob/main/Qwen3VL-4B-Instruct-Q4_K_M.gguf)**. The `mproj` file is necessary for a VLM model to perform visual task.
3. This project uses the LLM (`Qwen3-8B-Q4_K_M.gguf`) found **[here](https://huggingface.co/Qwen/Qwen3-8B-GGUF/tree/main)**.
4. The models run well on an RTX 3060 12GB. You can refer to the "FAQ" section regarding how to choose the model that runs best on you machine.
#### 1.5 Setup the config file
Finally, configure the following variables in `./scripts/config.json`:
- `vlm_model_path`, `vlm_proj_path`, `llm_model_path`: relative path to your downloaded models

You can leave the remaining settings as default. Alternatively, you can adjust these settings based on your habbit:
- `capture_interval`: Time interval between each screen capture (in seconds)
- `start_of_day`: Which hour is considered as the "start time" of the day (assuming people might stay up late after 12 a.m.)

#### 🎉🎉And you're done!🎉🎉

## 2. Running the pipeline
The full pipeline consists of two parts: a screen capturer and a report generator

#### 2.1 Screen Capturer
- Manually run `./scripts/auto_screen_capture.bat` to start a console window that continuosly captures your active window. Screenshots will be saved to `./screenshots` folder
- **Suggested:** Create a shortcut for this batch file and place it somewhere easily reachable for quick start (e.g. Desktop)
- **Suggested:** Or, use Window's `shell:startup` command to open up a folder, and put your shortcut there to automatically run the script during computer startup.

#### 2.2 Report Generator
- Simply run `./scripts/main.bat` to generate a report for your **previous day**. E.g., if you have `START_OF_DAY` as 6 and call this batch file on Feb 4 after 6:00 a.m., it will generate a report based on the screenshots taken between Feb 3 6:00 a.m. and Feb 4 5:59 a.m. **The report will be saved as a PDF on your desktop**. You can modify the save path in `./scripts/main.py`
- **Tip:** Alternatively, call the batch file with argument like `--date %Y-%M-%D` to generate a report for any given date. E.g., `--date 2026-02-03` will generate a report based on the screenshots taken between Feb 3 6:00 a.m. and Feb 4 5:59 a.m. (as long as you have screenshots for that day)

## 3. If you're using other OS...
FlowTrace is develoepd on Windows but should also run on other OS. The setup is the same, but you might need to pay attention to these during setup:
- Install llama server using the method specified for your own system (refer to their main GitHub page)
- Run commands (`./scripts/auto_screen_capture.bat` and `./scripts/main.bat`) are written as **batch files**, which won't run on systems like MacOS. You have to convert them to other formats like **shell script**.

## 4. Potential FAQs
- **Will taking screenshot every 1 minute take up a lot of disk space?**
No, images are resized to be smaller than 1920x1080 and are stored in JPEG format. **A typical image takes up only 100Kb to 200Kb of space**, meaning that every 60 images (1 hour of use time) take up less than 10Mb on average.
- **Can I run this pipeline without a GPU?**
Yes. Theoretically, llama server should handle the hardware setup so our script does not have to be concerned about that. 
    - **Will it be slower?**
    Probably, but if your CPU has an NPU unit, then the NPU will be helping out.
    - **And (potentially) good news:**
    When you run a model on CPU, the model is stored on **RAM** instead of **VRAM**. Most computers' RAM (8G, 16G, 32G...) is bigger than most consumer-level GPU's VRAM (assuming most people don't have GPUs like RTX5090 or H200). So, running on CPU means that you can likely run bigger models. Without a GPU, maybe try out **MoE models like "30B-A3B"**; they are the models that require larger VRAM/RAM but runs faster.
- **How to choose the right model?** 
    The model selection in Setup guide runs well on a RTX 3060 12GB GPU, and takes about 20 minutes to process a 10-hour-long period (about 600 screenshots). To find a model that fits best on your machine:
    - Hugging face provides a rough estimate of how much VRAM (or RAM) each model takes
    - The **Q4** quantization model is recommended: great balance between accuracy and VRAM efficiency.
    - If a model takes up too much VRAM/RAM, **try look for smaller quantized models first** (Q8 > Q6 > Q5 > Q4...), but it's not recommended to go below Q4 (it loses too much accuracy). Then, try look for a model with smaller parameter size (8B > 4B > 2B...)
    - If a model runs too slow on your machine, try look for models with a smaller parameter size
    - **If you have a lot of VRAM, try MoE models like "30B-A3B"**. They run much faster.
- **Will the screenshots be automatically deleted after generating a report?**
    At the current stage, this project does not delete past screenshots because:
    - They do not take up a lot of space
    - With historical screenshots, you are free to develop even crazier algorithms like "Compare my current week to past week" or "Store my activity history in a vector database so I can apply a search algorithm when needed"

    **Note:** If you want to delete the screenshots manually, don't forget the `./outputs` folder. Those are condensed versions of your activity log stored in text format (generated by VLM and LLM). Delete them to completely wipe out your trace.
- **Don't you feel uncomfortable having an algorithm that monitors your actions?**
Not really. FlowTrace was developed as a tool to perform better self-introspection, not as a "deadline pusher" that critiques user's activities. In fact, I purposefully prompted the models to **"honestly identify non-productivity activities (including leisure)"** and **"summarize the day objectively without judging the user**. Whether to use the report as a way to push themselve become more productive, or simply use it to understand what they're doing everyday, is up to the user to decide.
