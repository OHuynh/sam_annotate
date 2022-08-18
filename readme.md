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
python main.py --video video.mp4
```

### Shortcuts

<ul>
  <li>Left Click : Annotate a new bounding box => After drawing two boxes, the labeler pop-up window is displayed.</li>
  <li>Enter : Pop the labeler window. </li>
  <li>Right Click : Edit an existing box </li>
  <li>Right Click + Backspace : Remove a box from a sequence </li>
  <li>Space bar : Play mode </li>
  <li>a : generate interpolated boxes </li>
  <li>+/- : change at a finer scale the frame step </li>
  <li>q : exit </li>
</ul> 