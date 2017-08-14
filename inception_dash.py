# imports ----------------------------------------------------------------------
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import flask

from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications import imagenet_utils
from keras.preprocessing import image
import tensorflow as tf
import os

# initialize -------------------------------------------------------------------

app = dash.Dash()

app.css.append_css({
    'external_url': (
        'https://cdn.rawgit.com/chriddyp/0247653a7c52feb4c48437e1c1837f75'
        '/raw/a68333b876edaf62df2efa7bac0e9b3613258851/dash.css'
    )
})

# background_color = '#f2f2f2'
background_color = '#ffffff'

# intialize the images ---------------------------------------------------------

img_dir   = os.path.abspath('sample_images')
file_names = os.listdir(img_dir)
img_names  = list(map(lambda x: x.split('.')[0], file_names))
static_route = '/static/'

# intialize keras model --------------------------------------------------------

model = InceptionV3(weights='imagenet')
graph = tf.get_default_graph()

# layout -----------------------------------------------------------------------

app.layout = html.Div([

    # left column
    html.Div([

        # header text
        dcc.Markdown(
            '''
            ### ImageNet Classification with Keras

            The ImageNet Challenge is a image classification competition that pits 
            machine learning models against each other. Select from an image below to 
            see what the neural network model predicts. 
            This dashboard is built with Python and [Plotly Dash](https://plot.ly/products/dash/). 
            Classification is done using [Keras](https://keras.io/applications/) and the InceptionV3 model. 
            ***
            '''.replace('  ', ''), 
            className='container',
            containerProps={'style': 
                {'maxWidth': '650px', 
                'padding-top': '5%',
                 'background' : background_color}}
        ), 

        # file select
        dcc.Dropdown(
            id='img-selector',
            options=[{'label': i, 'value': j} for i, j in zip(img_names, file_names)],
            value=file_names[0]
        )
    ],

    # left column div style
    style={'width': '48%', 'display': 'inline-block', 'background' : background_color}
    ),

    # right column
    html.Div([

        # image display
        html.Img(
            id = 'img-square',
            style={'height':  '300px', 
                   'width':   '300px', 
                   'display': 'block', 
                   'margin':  'auto', 
                   'padding-top': '5%'}
        ),

        # graph display
        dcc.Graph(
            id='top5-graph'
        )

    ],

    style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'background': background_color}
    )

],

style={'background': background_color}

)

# backend ----------------------------------------------------------------------

# update bar plot
@app.callback(
     dash.dependencies.Output('top5-graph',   'figure'),
    [dash.dependencies.Input( 'img-selector', 'value')]
)
def update_graph(name):

    # load and preprocess image
    file_path = os.path.join(img_dir, name)

    img = image.load_img(file_path, target_size = (299, 299))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = img[None,:]

    # predict on new image 
    global graph
    with graph.as_default():
        preds = model.predict(img)
   
    preds = decode_predictions(preds, top=5)[0]

    pred_names   = list(map(lambda x: x[1], preds))
    pred_softmax = list(map(lambda x: x[2], preds))

    # return new graph
    return {'data': [{'x': pred_names, 'y': pred_softmax, 'type': 'bar'}],
            'layout': {'plot_bgcolor':  background_color, 'paper_bgcolor': background_color, 'yaxis': {'range': [0, 1]}}}

# update image
@app.callback(
     dash.dependencies.Output('img-square',   'src'),
    [dash.dependencies.Input( 'img-selector', 'value')]
)
def update_img(value):
    return os.path.join(static_route, value)

# static image route
@app.server.route('{}<image_path>.jpg'.format(static_route))
def serve_image(image_path):
    image_name = '{}.jpg'.format(image_path)
    return flask.send_from_directory(img_dir, image_name)

# start server -----------------------------------------------------------------

if __name__ == '__main__':
    app.run_server()