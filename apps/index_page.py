import dash_html_components as html
import dash_core_components as dcc

layout = html.Div([
	html.Hr(style = {'background': '', 'width': "100%"}, className='mt-1 mb-1 p-0'),
    html.H2('NLP', className='my-3'),
    dcc.Link('Text Analysis - NER, SUMMARY, WORDCLOUD', href='/app/text_analysis'),
	], className = 'm-3'),
	