# Video annotation tool for SAM project

A tool to quickly annotate bounding box using a semi-automated process. The user has to annotate the first and the last 
frame of a sequence, then the intermediate bounding boxes are interpolated using a pre-trained detector (like YOLO)
and a specified trajectory of the object (eg. linear).

## Usage

### Installation

Requires Python 3+ version.

```
pip install -r requirements.txt 
```

### Run

```
python main.py --video video.mp4  --output-path ./data --path-yolo-model ./models --path-yolo-data ./runs/detect [--input-annot ./video.json] [--execute-yolo "True"]
```
where:  
--path-yolo-data is the path where results from YOLO execution are saved. This path is used for both, 1) uploading previous YOLO executions and 2) saving YOLO executions carried out during the execution of the app. YOLO results are saved when shorcut 't' is executed.  

--execute-yolo is a flag (only activated when "True" (str value)) that indicates if YOLOv7 is executed and saved (for the full video) during the app start. This speeds up the execution of interpolation (shortcut 'a') during the app execution. Results are saved inside the path indicated in "--path-yolo-data", inside the "labels" folder.  

PS. YOLO data is different from annotation data. YOLO data is the result of executing YOLO in the video, which is used as base for the interpolation process. Annotation data is the result of annotation process (carried out manually), which includes results from YOLO data, but only for the objects of interest (objects marked with a box during annotations). 

### Shortcuts

<ul>
  <li>Left Click : Annotate a new bounding box => After drawing two boxes, the labeler pop-up window is displayed.</li>
  <li>Enter : Pop the labeler window. </li>
  <li>Right Click : Edit an existing box </li>
  <li>Right Click + Backspace : Remove a box from a sequence </li>
  <li>Space bar : Play mode </li>
  <li>a : generate interpolated boxes </li>
  <li>+/- : change at a finer scale the frame step </li>
  <li>s : save the annotation in a json file </li>
  <li>t : save the annotation in YOLO format (txt files, one file per frame), as well as YOLO detection results </li>
  <li>q : exit </li>
</ul> 