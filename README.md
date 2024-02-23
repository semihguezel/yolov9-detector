# yolov9-detector

# Usage
Clone the repository with the following command:  <br />
```bash
$ git clone https://github.com/semihguezel/yolov9-detector.git
```

Then, navigate to the yolov9-detector directory: <br /> 
```bash
$ cd yolov9-detector
```

Create a folder called "models" with the following command: <br />
```bash
$ mkdir models <br />**
```

Inside the "models" folder, you can add your pretrained models. Alternatively, you can use the default models available at <a 
  href="https://github.com/WongKinYiu/yolov9" 
  target="_blank">
 Yolov9 GitHub. </a>
To download models, go to the Performance section and click on the model names.
<p>
  <img src="https://github.com/semihguezel/yolov9-detector/assets/71209213/d88c6202-6f3e-4811-acb8-2b1566a1b503" width="350" title="Download Models"> <br />
</p>

After placing the model, add a file called "label.txt," which includes class IDs and names. Classes must be stored as key-value pairs, as shown here: "0: Person."

Here is a visual representation of how the "label.txt" file should look:
<p>
  <img src="https://github.com/semihguezel/yolov9-detector/assets/71209213/0f9aba9b-4aa4-4af4-a102-5739f626cb04" width="360" title="Example of how the label.txt file should look"> <br />
</p>

After that, go into the base directory and install the required packages by running the following command where the "requirements.txt" file is located:
```bash
$ pip install -r requirements.txt
```

Before running the project, fill in the necessary fields in the configuration file for the project to run successfully. The project currently accepts only video sources, so don't forget to set the "video_path" parameter inside the configuration file. Also, make sure to fill in the "model_path" and "label_path" parameters.

<p>
  <img src="https://github.com/semihguezel/yolov9-detector/assets/71209213/21ad3d55-56e0-4d86-baf4-533b57fef4da" width="360" title="Configuration file"> <br />
</p>

Lastly, navigate to the main folder and run "yolov9_detector.py" with the following command: <br />
```bash
$ python yolov9_detector.py
```

Note: This repository includes files from the official implementation of YOLOv9, which can be found <a 
  href="https://github.com/WongKinYiu/yolov9" 
  target="_blank">
 here </a>. The official repository may be updated with the latest changes that might not be included here.
