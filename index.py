def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from app import server
from app import app

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from apps import index_page
from apps.text_analysis1 import app1



app.layout = html.Div([
	dcc.Location(id='url', refresh=False),
	html.Div(id='page_content')
	])

@app.callback(
	Output('page_content', 'children'),
	Input('url', 'pathname')
)
def display_page(pathname):
	print(pathname)
	if pathname == '/app/text_analysis':
		return app1.app1_viz
	else:
		return index_page.layout


if __name__ == '__main__':
    app.run_server(debug=False, port=5500)

