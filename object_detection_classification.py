#@title Imports and function definitions

# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import cv2
import math

from PIL import Image as im
import os
from subprocess import Popen,PIPE

# from keras.utils import get_file
# import keras.utils as image
from keras.preprocessing import image

import pathlib

from tensorflow.keras.optimizers import Adam


# For measuring the inference time.
import time


# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())




def load_detector():
  print("load detector called")
  module_handle = "C:/Users/mkpre/acps/faster_rcnn_openimages_v4_inception_resnet_v2_1.tar/faster_rcnn_openimages_v4_inception_resnet_v2_1"

  detector = hub.load(module_handle).signatures['default']
  return detector

def load_predictor():
  print("load predictor called ")
  model_predict = tf.keras.models.load_model('C:/Users/mkpre/acps/classification_model.h5',compile=False)
  # model_predict.compile(optimizer=Adam(1e-5),
  #                       loss='categorical_crossentropy',
  #                       metrics=['accuracy'])
  model_predict.summary()
  return model_predict

def display_image(image):
  # fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)
  


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img



def run_detector(detector, path):
  img = load_img(path)

  image = cv2.imread(path)
  width,height,dim = image.shape

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

 
  result = {key:value.numpy() for key,value in result.items()}
  # print(type(result["detection_scores"]),len(result["detection_scores"]))
  # print(type(result["detection_boxes"]),len(result["detection_boxes"]))
  # print(type(result["detection_class_entities"]),len(result["detection_class_entities"]))

  
  # print(result["detection_class_entities"][0])
  # print(type(result["detection_class_entities"][0]))

  obj_name = bytes("Fruit", 'utf-8')
  banana_name = bytes("Banana", 'utf-8')
  orange_name = bytes("Orange", 'utf-8')
  apple_name = bytes("Apple", 'utf-8')
  strawberry_name = bytes("Strawberry",'utf-8')
  guava_name = bytes("Guava", 'utf-8')
  jujube_name = bytes("Jujube", 'utf-8')
  pomegranate_name = bytes("Pomegranate", 'utf-8')
  grape_name = bytes("Grapes", 'utf-8')

  Fruit_list = [obj_name, 
                banana_name, 
                orange_name, 
                apple_name, 
                strawberry_name, 
                guava_name, 
                jujube_name,
                pomegranate_name, 
                grape_name, 
              ]

  # print(obj_name)
  # print(type(obj_name))
  # print("Found %d objects." % len(result["detection_scores"]))
  # print("Inference time: ", end_time-start_time)

  o_scores = []
  o_boxes = []
  o_entities = []


  
  indices = []
  for i in range(len(result["detection_class_entities"])):
    if result["detection_class_entities"][i] in Fruit_list and result["detection_scores"][i] >= 0.40:
      indices.append(int(i))
      # print(result["detection_boxes"][i],result["detection_scores"][i],result["detection_class_entities"][i])
      ymin,xmin,ymax,xmax = result["detection_boxes"][i]

      y1,x1,y2,x2 = math.ceil(ymin*height),math.ceil(xmin*width),math.ceil(ymax*height),math.ceil(xmax*width)
      crop = image[y1:y2, x1:x2]
      # cv2_imshow(crop)
      o_scores.append(result["detection_scores"][i])
      o_boxes.append(result["detection_boxes"][i])
      o_entities.append(result["detection_class_entities"][i])


  o_scores = np.array(o_scores)
  o_boxes = np.array(o_boxes)
  o_entities = np.array(o_entities)
  # print("Maximum index : ",scores.index(max(scores)))

  if(len(o_boxes) == 0):
    print("No fruit found in Image !!")
    return np.array([])
  
  xmin,ymin,xmax,ymax = o_boxes[0]
  x1,y1,x2,y2 = math.ceil(ymin*height),math.ceil(xmin*width),math.ceil(ymax*height),math.ceil(xmax*width)
  crop = image[y1:y2, x1:x2]
  print("image shape ",image.shape)
  print("ymin xmin ymax xmax ",ymin,xmin,ymax,xmax)
  print("y1 x1 y2 x2 ",y1,x1,y2,x2)
  print("Crop shape ",crop.shape)
  cv2.imwrite("croped_image"+str(y1)+str(x1)+str(y2)+str(x2)+".jpg",crop)
  
  to_show_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
  det_fruit = im.fromarray(to_show_crop)
  det_fruit.show()
  image_with_boxes = draw_boxes(img.numpy(),o_boxes,o_entities,o_scores)
  to_save_i_w_b = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
  cv2.imwrite("input_image"+str(y1)+str(x1)+str(y2)+str(x2)+".jpg",to_save_i_w_b)
  i_w_b = im.fromarray(image_with_boxes)
  i_w_b.show()
  display_image(image_with_boxes)
  print("Detected "+ str(o_entities[0])+ " Image ")
  return crop





# tensorflow_graph = tf.saved_model.load("my_model")
# x = np.random.uniform(size=(4, 32)).astype(np.float32)
# predicted = tensorflow_graph(x).numpy()



def click_and_detect():
  print("click and detect called ")
  image_names = ['freshapple.jpg']
  for image_name in image_names:
    # something = input("Press button")
    # if(something == 'q'):
    #   proc = Popen(['ssh','acps@192.168.42.79'],stdin = PIPE)
    #   proc.communicate('acps')
    #   # os.system("ssh acps@192.168.42.79")
    #   # os.system("acps")
    #   os.system("sudo su")
    #   os.system("init 0")
    #   break 
    # os.system("python C:/Users/mkpre/acps/cl2.py")

    
    fruit_image = run_detector(detector, 'C:/Users/mkpre/acps/'+image_name)
    if(fruit_image.shape[0] == 0):
      return False
    image_name = []
    image_conf = []
    predict_result = []

    img = fruit_image
    # img = cv2.imread('C:/Users/mkpre/acps/freshbanana.jpg')
    
    img = cv2.resize(img,(150,150))
    # img = image.load_img(path, color_mode="rgb", target_size=(150, 150), interpolation="nearest")
    # imgplot = plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    images = np.vstack([img])
    classes = model_predict.predict(images, batch_size=10)

    max = np.amax(classes[0])
    if np.where(classes[0] == max)[0] == 0:
      
      image_conf.append(max)
      predict_result.append('Fresh Apple')
    elif np.where(classes[0] == max)[0] == 1:
      
      image_conf.append(max)
      predict_result.append('Fresh Banana')
    elif np.where(classes[0] == max)[0] == 2:
      
      image_conf.append(max)
      predict_result.append('Fresh Orange')
    elif np.where(classes[0] == max)[0] == 3:
      
      image_conf.append(max)
      predict_result.append('Rotten Apple')
    elif np.where(classes[0] == max)[0] == 4:
      
      image_conf.append(max)
      predict_result.append('Rotten Banana')
    else:
      
      image_conf.append(max)
      predict_result.append('Rotten orange')

    print(predict_result)

print("loading detector ........")
detector = load_detector()
print("Detector Loaded Successfully !")
print("Loading Predictor .......")
model_predict = load_predictor()
print("Predictor Loaded Successfully !")