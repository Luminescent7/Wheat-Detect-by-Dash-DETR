from dash import Dash, Input, Output, State, dcc, html, callback_context
import dash_bootstrap_components as dbc  # 新增的导入
import base64, io
from PIL import Image
import numpy as np
from model import detect  # 引入上面定义的 detect 函数
from flask import send_file

# 使用 Bootstrap 主题创建应用
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 定义统一的样式
common_style = {
    'width': '220px',         # 固定宽度，避免在大屏幕上过分拉伸
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'solid',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px',
    'fontSize': '18px',
    # 'fontWeight': 'bold',
}
# 替换原来的 app.layout 定义
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("小麦图片检测", className="text-center my-4"), width=12)),
    dbc.Col(
        html.Hr(style={'borderColor': '#ccc', 'borderWidth': '4px','margin-top':'0px','margin-bottom':'0px'}),
        width=12
    ),
    dbc.Row([
        
        dbc.Col(
            dcc.Upload(
                id='upload-image',
                children=html.Div(['拖拽或 ', html.A('选择一张小麦图片')]),
                style={**common_style,
                'backgroundColor': '#effaff',##B7CECE
                'color': '#1C0F13',
                'borderColor': '#E2E2E2',
                },
                accept='image/*',
            ), 
            width='auto'
        ),
        
        dbc.Col(
            dbc.Button(
                "执行推理", 
                id='btn-detect', 
                className="me-2",
                style={**common_style,
                'backgroundColor': '#f3fbf1',##B7CECE
                'color': '#1C0F13',
                'borderColor': '#E2E2E2',
                },
            ), 
            width='auto'
        ),
        
        dbc.Col(
            dbc.Button(
                "下载结果图片", 
                id='btn-download', 
                style={**common_style,
                'backgroundColor': '#fff6f1',##B7CECE
                'color': '#1C0F13',
                'borderColor': '#E2E2E2',
                },
            ), 
            width='auto'
        ),
        dcc.Download(id='download-image'),
        # 隐藏存储组件，用于存储检测结果图像的数据（供下载用）
        dcc.Store(id='annotated-image-store')
    ], className="mb-4",justify="center"),

    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("原图片",
                style={
                    'textAlign': 'center',
                    'fontSize': '16px',
                    'fontWeight': 'bold'
                }),
                dbc.CardBody(html.Img(id='image-original', 
                style={
                    'width': '100%',
                    'maxHeight': '400px',
                    'minHeight': '400px',
                    }))
            ], className="mb-4")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("检测结果",
                style={
                    'textAlign': 'center',
                    'fontSize': '16px',
                    'fontWeight': 'bold'
                }),
                dbc.CardBody(html.Img(id='image-annotated', 
                style={
                    'width': '100%',
                    'maxHeight': '400px',
                    'minHeight': '400px',
                    }))
            ], className="mb-4")
        ], md=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("置信度阈值:",style={'textAlign': 'center',}),
            dcc.Slider(id='confidence-slider', min=0, max=1, step=0.05, value=0.5,
                       marks={0: '0', 0.5: '0.5', 1: '1'}, tooltip={"placement": "bottom"})
        ], md=6),
        dbc.Col([
            html.Label("IoU 阈值:",style={'textAlign': 'center',}),
            dcc.Slider(id='iou-slider', min=0, max=1, step=0.05, value=0.5,
                       marks={0: '0', 0.5: '0.5', 1: '1'}, tooltip={"placement": "bottom"})
        ], md=6)
    ], className="mb-4"),
    
    
    
    
    
    dbc.Row([
        dbc.Col(
            html.Div(id='detection-info', style={
                'marginTop': '10px',
                'height': '300px',       # 固定高度
                'width':'800px',
                'overflowY': 'auto',      # 启用竖向滚动
                'marginLeft': 'auto',     # 关键：左右外边距为 auto
                'marginRight': 'auto'     # 关键：左右外边距为 auto
            }),
            width=12,
        )
    ]),
    
    
    
], 
fluid=True,
style={
'marginTop': '10px',
'marginBottom': '10px',
'paddingLeft': '10px',   # 左右留白
'paddingRight': '10px',
'fontFamily': '华文中宋',
#'backgroundColor': '#E5E5E5',
})

# 回调函数部分保持不变
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
    ctx = callback_context
    if not ctx.triggered:
        return None, None, "", None
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if not (0 <= conf_value <= 1) or not (0 <= iou_value <= 1):
        return None, None, "Invalid threshold values", None
        
    if contents and len(contents) > 10 * 1024 * 1024:  # 10MB
        return None, None, "File too large", None

    if trigger_id == "upload-image":
        if contents is None:
            return None, None, "", None
        return contents, None, "", None

    if trigger_id == "btn-detect":
        if contents is None:
            return None, None, "No image uploaded", None
        
        content_type, content_string = contents.split(',')
        image_data = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        annotated_img, detections = detect(image, conf=conf_value, iou=iou_value)
        if annotated_img is None:
            return contents, None, "Detection failed or no output.", None
        
        annotated_rgb = annotated_img[..., ::-1]
        buffered = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        annotated_src = "data:image/png;base64," + img_b64
        print("Stored Base64 Data:", annotated_src[:100])  # 仅打印前100个字符

        # 构建检测信息表格
        if len(detections) == 0:
            table_body = html.Tbody([
                html.Tr([html.Td("No objects detected", colSpan=2)])
            ])
        else:
            rows = []
            for det in detections:
                class_name = det['class']
                conf_score = det['confidence'] * 100
                rows.append(html.Tr([html.Td(class_name,style={'textAlign': 'center',}), html.Td(f"{conf_score:.1f}%",style={'textAlign': 'center',})]))
            table_body = html.Tbody(rows)
        
        detection_table = dbc.Table(
            [html.Thead(html.Tr([html.Th("类别",style={'textAlign': 'center',}), html.Th("置信度",style={'textAlign': 'center',})])), table_body],
            bordered=True, hover=True, responsive=True, striped=True
        )

        return contents, annotated_src, detection_table, annotated_src

    return None, None, "", None

@app.callback(
    Output('download-image', 'data'),
    [Input('btn-download', 'n_clicks')],
    [State('annotated-image-store', 'data')],
    prevent_initial_call=True
)
def download_image(n_clicks, annotated_data):
    if n_clicks is None or annotated_data is None:
        return None
    
    # annotated_data 通常形如 "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAALUCAIA..."
    if not annotated_data.startswith("data:image/png;base64,"):
        # 如果你只处理 PNG，可简单判断；若需兼容其他类型，需更灵活的处理
        return None
    
    # annotated_data: "data:image/png;base64,iVBOR..."
    prefix, b64_str = annotated_data.split(',', 1)
    # 将其解码为二进制文件
    image_bytes = base64.b64decode(b64_str)

    # 直接返回 Dash 提供的 helper，对文件名和二进制数据进行打包
    return dcc.send_bytes(
        io.BytesIO(image_bytes).getvalue(),  # 传入 bytes
        "detection_result.png"              # 指定下载文件名
    )
    
if __name__ == '__main__':
    app.run_server(debug=True)
