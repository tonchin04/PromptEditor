from cProfile import label
import glob
import os
import json
import csv

from tqdm import tqdm
import gradio
import huggingface_hub
import numpy
import onnxruntime
import pandas
import PIL.Image
from googletrans import Translator

from src import dbimutils, sort

translator = Translator()

path = ""

tagger_model_repos = [
  "SmilingWolf/wd-v1-4-moat-tagger-v2",
  "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
  "SmilingWolf/wd-v1-4-convnext-tagger-v2",
  "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
  "SmilingWolf/wd-v1-4-vit-tagger-v2",
]

loaded_models = {
  "SmilingWolf/wd-v1-4-moat-tagger-v2" : None,
  "SmilingWolf/wd-v1-4-swinv2-tagger-v2" : None,
  "SmilingWolf/wd-v1-4-convnext-tagger-v2" : None,
  "SmilingWolf/wd-v1-4-convnextv2-tagger-v2" : None,
  "SmilingWolf/wd-v1-4-vit-tagger-v2" : None,
}

thresholds = {
  "tags" : 0.4,
  "character_tags" : 0.5
}

danbooru_dict_path = "./tags.csv"

def loadImage(path):
  fileTypes = ('jpg', 'png', 'gif')
  images = []
  for type in fileTypes:
    images += glob.glob(path + '\\*.' + type)
  images = sorted(images, key=sort.natural_keys)
  global image_list, image_count
  image_list = []
  image_count = len(images)
  for path in images:
    basename = os.path.basename(path)
    image_list.append(basename)
  print("loaded images : " + str(image_count))
  return images

def loadModel(repo) -> onnxruntime.InferenceSession:
  path = huggingface_hub.hf_hub_download(repo_id=repo, filename="model.onnx", cache_dir="cache")
  model = onnxruntime.InferenceSession(path)
  return model

def changeModel(model_name):
  global loaded_models
  loaded_models[model_name] = loadModel(model_name)
  return loaded_models[model_name]

def loadLabels() -> list[str]:
  path = huggingface_hub.hf_hub_download(
      repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2", filename="selected_tags.csv", cache_dir="cache"
  )
  df = pandas.read_csv(path)

  tag_names = df["name"].tolist()
  rating_indexes = list(numpy.where(df["category"] == 9)[0])
  general_indexes = list(numpy.where(df["category"] == 0)[0])
  character_indexes = list(numpy.where(df["category"] == 4)[0])
  return tag_names, rating_indexes, general_indexes, character_indexes

def predict(
  image_dir,
  image_name,
  selected_model,
  tags_threshold,
  character_tags_threshold
  ):

  tag_names, rating_indexes, general_indexes, character_indexes = loadLabels()

  global loaded_model

  if selected_model is None:
    selected_model = "SmilingWolf/wd-v1-4-moat-tagger-v2"

  model = loaded_models[selected_model]

  if model is None:
    loadModel(selected_model)
    changeModel(selected_model)
    model = loaded_models[selected_model]

  _, height, width, _ = model.get_inputs()[0].shape

  img_path = os.path.join(image_dir, image_name)

  with PIL.Image.open(img_path) as image:
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = numpy.asarray(image)

    image = image[:, :, ::-1]

    image = dbimutils.make_square(image, height)
    image = dbimutils.smart_resize(image, height)
    image = image.astype(numpy.float32)
    image = numpy.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > tags_threshold]
    general_res = dict(general_res)

    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_tags_threshold]
    character_res = dict(character_res)

    detected_tags = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    tags = (
      ", ".join(list(detected_tags.keys()))
      .replace("_", " ")
      .replace("(", "\\(")
      .replace(")", "\\)")
    )

    # if os.path.isfile(danbooru_dict_path):
    #   with open(danbooru_dict_path, 'r', encoding='UTF-8') as f:
    #     tag_csv = csv.reader(f)
    #     tag_dict = {rows[0]: rows[1] for rows in tag_csv}
    #     translated_tags = tags
    #     for old, new in tag_dict.items():
    #       translated_tags = translated_tags.replace(old, new)

    # else:
    #   translated_tags = None

  return tags, rating, general_res, character_res

def interrogate_all(image_dir, image_list, selected_model, tags_threshold, character_tags_threshold):

  for image_path in tqdm(image_list):
    image_name = os.path.basename(image_path[0])
    interrogate_output, _, _, _ = predict(image_dir, image_name, selected_model, tags_threshold, character_tags_threshold)
    image_name_wo_ext = os.path.splitext(image_name)[0]
    with open(os.path.join(image_dir, (image_name_wo_ext + '.txt')), 'w', encoding='UTF-8') as f:
      f.write(interrogate_output)
  print('Done.')
  return

def replacePrompt(prompt):
  return prompt

def getSelectIndex(evt: gradio.SelectData, image_dir):
  filename = getFilename(evt.index)
  prompt = getPrompt(filename, image_dir)
  global image_count
  pages = str(evt.index + 1) + "/" + str(image_count)
  return [filename, prompt, pages]

def getFilename(index):
  filename = image_list[index]
  return filename

def getPrompt(filename, image_dir):
  canReadPrompt = os.path.isfile(os.path.join(image_dir, (os.path.splitext(filename)[0] + '.txt')))
  if canReadPrompt:
    with open(os.path.join(image_dir, (os.path.splitext(filename)[0] + '.txt')), 'r', encoding='UTF-8') as f:
      prompt = f.read()
  else:
    prompt = ""
  return prompt

def savePrompt(filename, image_dir, prompt):
  with open(os.path.join(image_dir, (os.path.splitext(filename)[0] + '.txt')), 'w', encoding='UTF-8') as f:
    f.write(prompt)
    print("Save Prompt: " + os.path.join(image_dir, (os.path.splitext(filename)[0] + '.txt')))
  return

def translatePrompt(prompt):
  translatedPrompt = translator.translate(prompt, dest='ja').text
  return translatedPrompt

def detectAutoTranslate(is_auto_translate, prompt):
  if is_auto_translate:
    return translatePrompt(prompt)
  return ""

def clearTranslatedPrompt():
  return ""

with gradio.Blocks() as prompt_editor:
  gradio.Markdown("# Prompt Editor v1.1")
  with gradio.Tab("Tagger"):
    with gradio.Row():
      image_dir = gradio.Textbox(label="Image Directory Path", scale=5, max_lines=1, value=path)
      reload_button = gradio.Button(value="Load", variant="secondary", scale=1)

    with gradio.Row():
      images = gradio.Gallery(show_label=False, columns=5, container=True, scale=3)

      with gradio.Column(scale=2):
        with gradio.Row():
          selected = gradio.Textbox(label="Selected Image", max_lines=1, scale=4, interactive=False)
          pages = gradio.Textbox(label="Number", scale=1)
        with gradio.Row():
          previous_button = gradio.Button(value="Previous", variant="secondary")
          next_button = gradio.Button(value="Next", variant="secondary")

        prompt = gradio.Textbox(label="Prompt", show_copy_button=True)
        ja_prompt = gradio.Textbox(label="Translated Prompt", show_copy_button=True)

        with gradio.Row():
          translate_button = gradio.Button(value="Translate", variant="secondary", scale=2)
          is_auto_translate = gradio.Checkbox(label="Auto", scale=1)

        with gradio.Row():
          save_button = gradio.Button(value="Save", variant="primary")
          revoke_button = gradio.Button(value="Revoke", variant="secondary")


    gradio.Markdown("## Interrogate")
    with gradio.Row():
      with gradio.Column():
        with gradio.Row():
          selected_model = gradio.Dropdown(label="Interrogate Model", choices=tagger_model_repos, value="SmilingWolf/wd-v1-4-moat-tagger-v2", scale=3, interactive=True)
          interrogate_button = gradio.Button(value="Interrogate")
        with gradio.Row():
          tags_threshold = gradio.Slider(label="Tags Threshold", minimum=0, maximum=1, value=thresholds["tags"])
          character_tags_threshold = gradio.Slider(label="Character Tags Threshold", minimum=0, maximum=1, value=thresholds["character_tags"])
      with gradio.Column():
        interrogate_output = gradio.Textbox(label="Interrogate Output", show_copy_button=True)
        with gradio.Row():
          replace_prompt = gradio.Button(value="Replace Prompt", variant="primary")
          interrogate_all_button = gradio.Button(value="Interrogate All Images", variant="secondary")

    with gradio.Row():
      with gradio.Column():
        rating = gradio.Label(label="Rating")
        character_tags = gradio.Label(label="Character Tag")
      tags = gradio.Label(label="Tags")

    image_count = gradio.State()
    image_list = gradio.State()

    reload_button.click(fn=loadImage, inputs=image_dir, outputs=images)
    images.select(fn=getSelectIndex, inputs=image_dir, outputs=[selected, prompt, pages])
    selected.change(fn=clearTranslatedPrompt, outputs=ja_prompt)
    selected.change(fn=detectAutoTranslate, inputs=[is_auto_translate, prompt], outputs=ja_prompt)
    prompt.change(fn=detectAutoTranslate, inputs=[is_auto_translate, prompt], outputs=ja_prompt)
    translate_button.click(fn=translatePrompt, inputs=prompt, outputs=ja_prompt)
    revoke_button.click(fn=getPrompt, inputs=[selected, image_dir], outputs=prompt)
    save_button.click(fn=savePrompt, inputs=[selected, image_dir, prompt])
    selected_model.change(fn=changeModel, inputs=selected_model)
    selected.change(fn=predict, inputs=[image_dir, selected, selected_model, tags_threshold, character_tags_threshold], outputs=[interrogate_output, rating, tags, character_tags])
    interrogate_button.click(fn=predict, inputs=[image_dir, selected, selected_model, tags_threshold, character_tags_threshold], outputs=[interrogate_output, rating, tags, character_tags])
    interrogate_all_button.click(fn=interrogate_all, inputs=[image_dir, images, selected_model, tags_threshold, character_tags_threshold])
    replace_prompt.click(fn=replacePrompt, inputs=interrogate_output, outputs=prompt)

  with gradio.Tab("Settings"):
    gradio.Markdown("## Tagger")
    gradio.Markdown("Default Path:")


if __name__ == "__main__":
  prompt_editor.launch(inbrowser=True)
