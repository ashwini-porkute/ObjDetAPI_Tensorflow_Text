### Steps to create Object detection model using tensorflow object detection API.

Reference Links:
------------------
https://gilberttanner.com/blog/creating-your-own-objectdetector/

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-models-install

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#download-pre-trained-model

https://medium.com/iuc-bilgisayar-kulubu/training-custom-object-detector-step-by-step-4ef47bc2cb03

STEPS:
==========
1. Create Project directory:
- Create a project folder named "Object_Detection_tensorflow" and create a virtual environment named "objectdet" by using the below command.
$ python3 -m venv ./objectdet
$ source objectdet/bin/activate

2. Clone the tensorflow object detection repository:
$ git clone https://github.com/tensorflow/models


3. Download system specific protobuf zip folder from "https://github.com/protocolbuffers/protobuf/releases"
ex: protoc-23.1-linux-aarch_64.zip extract and copy it to path so that final tree structure of the project is like below:
		.
		|-- README.md
		|-- models
		|-- objectdet
		|-- requirements.txt
		`-- protoc-23.1-linux-aarch_64


if protoc is not installed then installed it with:
$ sudo snap install protobuf
$ protoc --version
$ gedit ~/.bashrc
add protoc file path in PATH variable (: separated). ex: PATH="$PATH:/usr/bin:/home/ashwini/my_git_projects/Object_Detection_tensorflow/protoc-23.1-linux-aarch_64/bin

and then goto models/research folder and run below:

$ ./protoc object_detection/protos/*.proto --python_out=.

4. Install the tensorflow object detection api and also verifying the installation:
$ cp object_detection/packages/tf2/setup.py .
$ python -m pip install .
(below command to verify the tensorflow object detection api installation)
$ python object_detection/builders/model_builder_tf2_test.py

if below error is encountered:
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
Downgrade the protobuf package to 3.20.x or lower.
Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

Then type below command:
$ pip install --upgrade "protobuf<=3.20.1"


5. Create dataset:
- Downloaded the SVT scene text dataset from kaggle "https://www.kaggle.com/datasets/nageshsingh/the-street-view-text-dataset"
- copy and extract the zip folder into "SVT_Text_dataset" folder.
		.
		|-- README.md
		|-- SVT_Text_dataset
		|-- models
		|-- objectdet
		|-- requirements.txt
		`-- protoc-23.1-linux-aarch_64
- copy the 2 xmls into "xmls/" folder created manually as shown below:
		.
		|-- img
		|   |-- 00_00.jpg
		|   |-- 00_01.jpg
		|   |-- 00_02.jpg
		|-- xml_to_csv_svt.py
		`-- xmls
		    |-- test.xml
		    `-- train.xml

- run "xml_to_csv_svt.py" script, this will create 2 csv files as required.
$ python xml_to_csv_svt.py

6. create tf_record format of generated dataset:
- create a folder named "tf_records" under models/research folder.

- Run below command from models/research directory to generate train tfrecord file:
$ python generate_tfrecord.py --csv_input=/home/ashwini/my_git_projects/Object_Detection_tensorflow/SVT_Text_dataset/train_labels.csv  --output_path="tf_records/train.record" --image_dir="/home/ashwini/my_git_projects/Object_Detection_tensorflow/SVT_Text_dataset"

- Run below command from models/research directory to generate test tfrecord file:
$ python generate_tfrecord.py --csv_input=/home/ashwini/my_git_projects/Object_Detection_tensorflow/SVT_Text_dataset/test_labels.csv  --output_path="tf_records/test.record" --image_dir="/home/ashwini/my_git_projects/Object_Detection_tensorflow/SVT_Text_dataset"

Note: change the line no 34 according to the dataset labels.(as it is text detection problem the label is "text")

7. Training the model using transfer learning:
- Create "training" folder under models/research/object_detection folder.

- before training, we need to create a label map and a training configuration file.
label map maps an id to a name.
Note: The id number of each item should match the id specified in the generate_tfrecord.py file, and id:0 is reserved for the background.
here, after creating labelmap.pbtxt file copy it inside "training" folder which was created earlier.


- Download the pre-trained "SSD ResNet50 V1 FPN 640x640" model (as tensorflow version>=2 outdates the ssd mobilenet v1 coco model)using below command:
$ wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

after downloading the pretrained model, extract and copy the same in "training" folder and also copy the config of that model to "training" folder and change it according to your use case.
for changes of config file, refer: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#download-pre-trained-model


- Start training with below command:
$ python model_main_tf2.py --model_dir=training/my_ SSD_ResNet50_V1_FPN --pipeline_config_path=training/my_ SSD_ResNet50_V1_FPN.config

Note: above training will take too much time on low end configuration systems

above training will create the checkpoints under --model_dir folder.

8. Visualization of training on tensorboard:
9.  Exporting the checkpoint to ob file.
10. Converting pb to lite format i.e tflite for embedded or android devices.



