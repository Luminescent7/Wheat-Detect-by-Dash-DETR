from dash import Dash,Input, Output, State, dcc, html
import base64, io
from PIL import Image
import numpy as np
from model import detect  # 引入上面定义的 detect 函数
from flask import send_file
app = Dash(__name__)

# 布局：上传组件、显示图片的区域、参数调整滑块、按钮等
app.layout = html.Div([
    
    html.H2("Wheat Image Detection"),  # 标题
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and Drop or ', html.A('Select a Wheat Image')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        accept='image/*'
    ),
    # 原始图像和检测结果图像并排显示
    html.Div([
        html.Div([
            html.H5("Original Image"),
            html.Img(id='image-original', style={'maxWidth': '100%', 'maxHeight': '500px'})
        ], style={'flex': '1', 'padding': '10px'}),
        html.Div([
            html.H5("Detection Result"),
            html.Img(id='image-annotated', style={'maxWidth': '100%', 'maxHeight': '500px'})
        ], style={'flex': '1', 'padding': '10px'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    # 置信度和 IoU 阈值调整滑块
    html.Label("Confidence Threshold:"),
    dcc.Slider(id='confidence-slider', min=0, max=1, step=0.05, value=0.5,
               marks={0: '0', 0.5: '0.5', 1: '1'}, tooltip={"placement": "bottom"}),
    html.Br(),
    html.Label("IoU Threshold:"),
    dcc.Slider(id='iou-slider', min=0, max=1, step=0.05, value=0.5,
               marks={0: '0', 0.5: '0.5', 1: '1'}, tooltip={"placement": "bottom"}),
    html.Br(),
    # 按钮：运行检测 和 下载结果
    html.Button("Run Detection", id='btn-detect', n_clicks=0),
    html.Button("Download Result Image", id='btn-download'),
    dcc.Download(id='download-image'),
    # 显示检测信息（类别和置信度）
    html.Div(id='detection-info', style={'whiteSpace': 'pre-wrap', 'marginTop': '10px'}),
    # 隐藏存储组件，用于存储检测结果图像的数据（供下载用）
    dcc.Store(id='annotated-image-store')
])
from dash import callback_context  # 需要引入 dash.callback_context

@app.callback(
    [Output('image-original', 'src'),
     Output('image-annotated', 'src'),
     Output('detection-info', 'children'),
     Output('annotated-image-store', 'data')],
    [Input('upload-image', 'contents'),
     Input('btn-detect', 'n_clicks')],
    [State('confidence-slider', 'value'),
     State('iou-slider', 'value')],
    prevent_initial_call=True
)
def handle_image_upload_and_detection(contents, n_clicks, conf_value, iou_value):
    """
    处理图片上传和模型推理：
    - 如果用户上传图片，显示原始图片并清空检测结果
    - 如果用户点击检测按钮，运行模型检测并更新检测结果
    """
    ctx = callback_context  # 获取触发来源
    if not ctx.triggered:
        return None, None, "", None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]  # 获取触发的组件ID

    # 添加输入验证
    if not (0 <= conf_value <= 1) or not (0 <= iou_value <= 1):
        return None, None, "Invalid threshold values", None
        
    # 添加文件大小限制
    if contents and len(contents) > 10 * 1024 * 1024:  # 10MB
        return None, None, "File too large", None

    # 1️⃣ **如果是上传图片**
    if trigger_id == "upload-image":
        if contents is None:
            return None, None, "", None
        return contents, None, "", None  # 清空检测结果，显示原图

    # 2️⃣ **如果是点击 "Run Detection" 按钮**
    if trigger_id == "btn-detect":
        if contents is None:
            return None, None, "No image uploaded", None
        
        # 解码 base64 图片
        content_type, content_string = contents.split(',')
        image_data = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 运行模型推理
        annotated_img, detections = detect(image, conf=conf_value, iou=iou_value)
        if annotated_img is None:
            return contents, None, "Detection failed or no output.", None
        
        # 将检测结果转换为 base64 以便显示
        annotated_rgb = annotated_img[..., ::-1]
        buffered = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        annotated_src = "data:image/png;base64," + img_b64

        # 处理检测信息（类别和置信度）
        info_elements = []
        if len(detections) == 0:
            info_elements.append(html.P("No objects detected."))
        else:
            for det in detections:
                class_name = det['class']
                conf_score = det['confidence'] * 100
                info_elements.append(html.P(f"{class_name}: {conf_score:.1f}%"))

        return contents, annotated_src, info_elements, annotated_src  # 返回检测结果

    return None, None, "", None  # 如果意外情况，返回默认值


# 回调3：点击下载按钮时，提供带检测标注的图像文件下载
@app.callback(
    Output('download-image', 'data'),
    [Input('btn-download', 'n_clicks')],
    [State('annotated-image-store', 'data')],
    prevent_initial_call=True
)
def download_image(n_clicks, annotated_data):
    if n_clicks is None or annotated_data is None:
        return None
    
    # 提取 base64 部分并解码为二进制图像数据
    header, b64_data = annotated_data.split(',', 1)
    image_bytes = base64.b64decode(b64_data)
    
    # 将二进制图像数据转换为 base64 字符串
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # 返回 base64 编码后的数据和文件名，触发下载
    return dict(content=f"data:image/png;base64,{b64_image}", filename="detection_result.png")


if __name__ == '__main__':
    app.run_server(debug=True)
