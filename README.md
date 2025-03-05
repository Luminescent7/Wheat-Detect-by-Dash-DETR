# Wheat-Detect-by-Dash-DETR

A wheat detection platform based on [*Dash*](https://github.com/plotly/dash) and [*Ultralytics*](https://github.com/ultralytics/ultralytics).  
The system allows users to upload images in real time, and dynamically adjust detection parameters such as confidence and IoU threshold through the slider, so that the original image and detection results are displayed side by side, so that users can visually compare the differences before and after detection.  In addition, the system also provides a detailed table display of the detection information and the function of downloading the image of the detection result. 

## Usage

1. Clone this repo:
```
git clone https://github.com/Luminescent7/Wheat-Detect-by-Dash-DETR.git
cd Wheat-Detect-by-Dash-DETR
```

2. Create a fresh venv (with `conda` or `virtualenv`) and activate it:
```
conda create -n dash-detr python=3.8
conda activate dash-detr
```

3. Install the requirements:
```
pip install -r requirements.txt
```

4. Start the app:
```
python app.py
```

5. Try the app at `localhost:8050`!

## Statement
The inspiration for this project is from [*dash-detr*](https://github.com/plotly/dash-detr/tree/master)
