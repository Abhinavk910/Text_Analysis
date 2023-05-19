#from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from app import app
import spacy
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from gensim.summarization import summarize
import re
import requests
from bs4 import BeautifulSoup
import operator
import io
from matplotlib import pyplot as plt
import base64
import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from string import punctuation
from collections import Counter
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS
from spacy_download import load_spacy

#my_model = T5ForConditionalGeneration.from_pretrained('assets/text_analysis1/model/pretrained_model/')
#tokenizer = T5Tokenizer.from_pretrained('assets/text_analysis1/model/tokenizer/')

# nlp = spacy.load("en_core_web_sm")
nlp = load_spacy("en_core_web_sm")
df=pd.read_excel('assets/text_analysis1/data/my_dictionary.xlsx')
df=df.sort_values(by='len')

text_color=[['#05ADD4', '#0486DA','#012273', '#1F1741', '#e2e5d7']]
tc=0
def bs(link):
    response=requests.get(link)
    soup=BeautifulSoup(response.text, "html.parser")
    return soup
def newvalue(OldValue, num):
    OldMax = num[0]
    OldMin = num[-1]
    NewMax = 50
    NewMin = 15
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue
def newvalue2(OldValue, key):
    OldMax = 50
    OldMin = 15
    NewMax = 0.87
    NewMin = 0.26
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = round((((OldValue - OldMin) * NewRange) / OldRange) + NewMin, 2)
    length = len(key.strip())
    avg_len = NewValue
    spance_taken_in_x = round(avg_len / 26 * length, 2)
    return spance_taken_in_x
def to_sub(x):
    
    pos = ['f', 'i', 'j', 'l', 't']
    neg = ['m', 'w']
    count = 0
    for i in x:
        if i in pos:
            count -= 0.007
        elif i in neg:
            count += 0.01
    return count
def hhhh(x):
    return newvalue2(x['ulter_size'], x['word']) + 0.02 + to_sub(x['word'])

def getcumsum(data, x, y):
    if y == 'color':
        return data[data.word == x][y]
    else:
        return float(data[data.word == x][y])
def ad(x, add):
    try:
        suma = x['space_take'] + add[x['group']]
    except:
        suma = x['space_take']
    return suma
def  getvocab(link = 'nil', data = 'nil', from_data = True):
    if from_data:
        visible_text = data
    else:
      
        soup = bs(link)
        # visible_text = soup.getText()

        visible_text = (" ").join([i.text for i in soup.findAll('p')] + [i.text for i in soup.findAll('li')])

    text = re.sub('[&?!\"\"”)(“\©,-//|\n\t{[\]\:=;\r ]+'," ", visible_text)
    text = re.sub('[^a-zA-Z]', " ", text)
    text = re.sub('wg\w+', "", text)
    text = text.lower()
    text = re.sub('wiki\w+', "", text)
    text = re.findall('[a-z\-\'\"]+', text)
    text = [word for word in text if word.lower() not in STOP_WORDS and len(word) != 1]
    doc = nlp(" ".join(text))
    text = [token.lemma_ for token in doc]
    vocab = {}
    for word in text:
        if word not in vocab.keys():
            vocab[word] = 1
        else:
            vocab[word] += 1

    vocab = sorted(vocab.items(), key = operator.itemgetter(1), reverse = True)[:50]
    vocab = {key:value for key,value in vocab}
    num = list(vocab.values())
    vocab = sorted(vocab.items())
    vocab = {key:value for key,value in vocab}


    return vocab,num 
def get_data(vocab, num):  
    data = pd.DataFrame([vocab], ).T.reset_index()
    data.columns = ['word', 'fref']
    data['ulter_size'] = [newvalue(i, num = num) for i in list(data.fref)]
    data['space_take'] = data.apply(lambda x: hhhh(x), axis = 1)
    initial_x = 0.0
    maxvalue = 0.95
    lastvalue = initial_x
    newcum = []
    group = []
    gg = 0
    group_sub = {}
    group_no = {}
    i = 0
    for row in data.itertuples():
        thisvalue =  row[4] + lastvalue
        if thisvalue > maxvalue:
            group_no[gg] = i
            i = 1
            group_sub[gg] = maxvalue - oldthisvalue
            gg += 1
            thisvalue = row[4]
            newcum.append( thisvalue )
            lastvalue = thisvalue
            group.append(gg)
            continue
        i += 1
        newcum.append( thisvalue )
        lastvalue = thisvalue
        group.append(gg)
        oldthisvalue = thisvalue
    group_no[gg] = i
    group_sub[gg] = maxvalue - oldthisvalue
    data['cumsum'] = newcum
    data['group'] = group
    add = {i:group_sub[i]/group_no[i]  for i in range(len(group_no))}
    del add[list(add.keys())[-1]]
    data['space_take_new'] = data.apply(lambda x: ad(x, add), axis = 1)
    maxvalue = 1
    lastvalue = initial_x
    newcumnew = []
    group = []
    gg = 0
    i = 0
    for row in data.itertuples():
        thisvalue =  row[7] + lastvalue
        if thisvalue > maxvalue:
            gg += 1
            thisvalue = row[7]
            newcumnew.append( thisvalue )
            lastvalue = thisvalue
            group.append(gg)
            continue
      # gg += 1
        newcumnew.append( thisvalue )
        lastvalue = thisvalue
        group.append(gg)
        oldthisvalue = thisvalue

    data['cumsum_new'] = newcumnew
    data['group_new'] = group
    data['color'] = data['ulter_size'].apply(lambda x: text_color[tc][0]   if x < 25 else text_color[tc][1] if x < 35 else text_color[tc][2] if x < 45 else text_color[tc][3] )
    return data

def entname(name):
    return html.Span(name, style={
        "font-size": "0.8em",
        "font-weight": "bold",
        "line-height": "1",
        "border-radius": "0.35em",
        "text-transform": "uppercase",
        "vertical-align": "middle",
        "margin-left": "0.5rem"
    })

def entbox(children, name, color):
    if name not in ['PERSON', 'ORG','GPE', "EVENT",'WORK_OF_ART', 'LAW', 'LOC','PRODUCT']:


        change = html.Mark(children, style={
            "background": color,
            "padding": "0.45em 0.6em",
            "margin": "0 0.25em",
            "line-height": "1",
            "border-radius": "0.35em",
        })
    else:
        change=dbc.Button(children,href="https://www.google.com/search?q="+"+".join(children[:-1]), target="_blank", style={
            "background": color,
            "padding": "0.45em 0.6em",
            "margin": "0 0.25em",
            "line-height": "1",
            "border-radius": "0.35em",
        })
    return change


def entity(children, name):
    if type(children) is str:
        children = [children]

    children.append(entname(name))
    color={'PERSON':'#AA9CFC', 'ORG':'#7AECEC', 'DATE':'#BFE1D9', 'GPE':"#FECA74", 'ORDINAL':'blue',
           'CARDINAL':'#E4E7D2', "EVENT":'#F0D0FF', 'WORK_OF_ART':'magenta', 'LAW':"#FF8197", 'TIME':'#BFE1D9',
          "LOC": "#FF9560", 'PRODUCT':'#BFEEB9'}
    return entbox(children, name, color[name])

set_sum=set(df.Word.unique())

def make_button(placement):
    return dbc.Button(
        f"{placement}",href="https://www.google.com/search?q="+placement, target="_blank",
        id=f"tooltip-target-{placement}",
        className="mx-1",
        n_clicks=0,
    )

def make_tooltip(placement, meaning):
    return dbc.Tooltip(
        f"{meaning}",
        target=f"tooltip-target-{placement}",
        placement="top",
    )

def make_final_tootip(word):
    return [make_button(word)]+[make_tooltip(word, df[df.Word==word].Meaning.values[0])]
    
def render(doc, drop_value):
    children = []
    last_idx = 0
    for ent in doc.ents:
        text_no = doc.text[last_idx:ent.start_char]
        for text_sp in text_no.split(" "):
            text_sp2=re.sub('[^a-zA-Z]', " ", text_sp).strip()
            if text_sp2 in set_sum:
                children += make_final_tootip(text_sp2.strip())
            else:
                children.append(text_sp)
                children.append(" ")

        if ent.label_ not in drop_value:
            children.append(doc.text[ent.start_char:ent.end_char])
                
        else:

            children.append(
                entity(doc.text[ent.start_char:ent.end_char], ent.label_))
        last_idx = ent.end_char
    children.append(doc.text[last_idx:])
    return children

def draw(data, min_height=""):
    return dbc.Card([
                    dbc.CardBody([
                        data
                    ], className='h-100')
                ], className='h-100', style={'min-height':min_height})

def drawtextarea():
    
    data=html.Div([
                   dbc.Textarea(className="m-0 px-3 py-2",
                    placeholder="Paste text here or click button to extract text form Website", 
                    id='text-nlp1',
                                     style={'background':'#F1F1F1','min-height':"160px"})
                ], style={'textAlign': 'center'}, className='h-100') 
    return draw(data,  min_height="150px")

def wordclodarea():
    
    data=html.Div([
                   dcc.Loading(
                                id="loadingnew-1",
                                type="default",
                                children= html.Img(id='wordcloud-nlp1', style={'max-width': '100%', 'height': 'auto'}), # img element
                                    )
                ], style={'textAlign': 'center'}, className='h-100') 
    return draw(data, min_height="300px")

def drawbutton(idx, name):
    data=dbc.Button(name,id='button-'+idx+'-nlp1', color="primary", className="mr-1")
    return draw(data)

def ner(name):
    data=dcc.Loading(
                                id="loadingnew-2-"+name,
                                type="default",
                                children= html.P(children=[''],id='text-'+name+'-nlp1')
    )
    return draw(data, min_height="300px")
    
def multidropdown():
    data = dcc.Dropdown(id='drop-nlp1',
                options=[
                    {'label': i, 'value': i} for i in [ 'CARDINAL', 'DATE',"EVENT",'GPE','LAW', 'LOC','ORG','ORDINAL',
                                                       'PERSON','PRODUCT', 'TIME','WORK_OF_ART',
                                                         ]],
                value=['PERSON', 'ORG','DATE','GPE', "EVENT",'WORK_OF_ART', 'LAW', 'LOC','PRODUCT'],
                multi=True,style={'font-size':'10px'}
            )  
    return draw(data)

def modelbuild(idx, header="Header", body="this is body"):
    return dbc.Modal(
            [
                dbc.ModalHeader(header),
                dbc.ModalBody(body),
                dbc.ModalFooter(
                    dbc.Button(
                        "Extract!!", id="close-"+idx+"-nlp1", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="modal-"+idx+"-nlp1",
            size="lg",
            is_open=False,
        )

def table():
    data = html.Div([
        dash_table.DataTable(
                        id='table-nlp-1',
                        columns=[{"name": i, "id": i, "selectable": True} for i in ['Word','Freq']],
                        page_size=5,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        page_action="native",
                        page_current= 0,
                        style_cell={'whiteSpace': 'normal','height': 'auto'},
                        
                        
    )],id='table_hidden_nlp1',hidden=True)
    return data


def sigledropdown():
    data = dcc.Dropdown(id='drop2-nlp1',
                options=[
                    {'label': i, 'value': i} for i in [ 'Gensim', 'Spacy','LexRank','LSA', 'Luhn', 'KL-Sum']],
                value='Gensim',
                multi=False,style={'font-size':'10px'}
            )  
    return draw(data)


app1_viz=html.Div([
    dbc.Card([
        dbc.CardHeader(html.H2("Text Analytics"), style={'text-align':'center','font-size':'20px', 'color':"#F1F1F1",'background':'#1C4E80',
                                                    'min-height':'8vh'}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawtextarea()
                ], md=12, lg=10, className='my-2 h-100',style={'background':''}),
                dbc.Col([
                    drawbutton(idx='upload_text_from_web',name= 'Extract text from Website'),
                    modelbuild(idx='upload_text_from_web',
                               header='Extract text from Website',
                               body=dbc.Input(id="input-upload_text_from_web-nlp1",
                                              placeholder="Type site link... Ex:-https://abhinavk910.github.io/ak.github.io", type="text")
                              )
                ], md=12, lg=2, className='my-2 d-flex flex-column flex-grow-1 w-100 ',style={'background':''}), 
            ], align='center', justify='between', className='h-100'),
            dbc.Row([
                dbc.Col([
                    wordclodarea()
                ], md=12, lg=10, className='my-2 h-100',style={'background':''}),
                dbc.Col([
                    drawbutton(idx='wordcloud',name='Generate WordCloud'),
                    table()
                ], md=12, lg=2, className='my-2 d-flex flex-column flex-grow-1 w-100 ',style={'background':''}), 
            ], align='center', justify='between', className='h-100'),
            dbc.Row([
                dbc.Col([
                    ner('ner')
                ], md=12, lg=10, className='my-2 h-100',style={'background':''}),
                dbc.Col([
                    drawbutton(idx='ner', name='Named Entity Recognition'),
                    multidropdown(),
                ], md=12, lg=2, className='my-2 d-flex flex-column flex-grow-1 w-100 ',style={'background':''}), 
            ], align='center', justify='between', className='h-100'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    ner('summary_gensim')
                ], md=12, lg=10, className='my-2 h-100',style={'background':''}),
                dbc.Col([
                    drawbutton(idx='summary_gensim', name='Generate Summary'),
                    sigledropdown(),
                ], md=12, lg=2, className='my-2 d-flex flex-column flex-grow-1 w-100 ',style={'background':''}), 
            ], align='center', justify='between', className='h-100'),
            html.Br(),
        ], className="h-100"),
        
    ],style={'background':'#F1F1F1'}, className='m-0'),
        html.Div([
            html.Div([
                html.P([
                    'Created by ',html.A('Abhinav', href='http://www.linkedin.com/in/abhinavk910', target="_blank",style={'color': '#0069D9'}),
                    html.A(' Kumar', href="https://twitter.com/abhinavk910", target="_blank",style={'color': '#0069D9'})],
                    style={'color': "black"}, className='m-0'),
                html.P([
                    'Tool: ',
                    html.Span('Plotly', style={'color': '#0069D9'}),
                ], style={'color': "black"}, className='m-0')
            ], className='')
    ], style={'background': '', 'min-height': '100px', "text-align": "center"}, className='pt-4 pl-4 m-auto')])



def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


app.callback(
    Output("modal-upload_text_from_web-nlp1", "is_open"),
    [Input("button-upload_text_from_web-nlp1", "n_clicks"), Input("close-upload_text_from_web-nlp1", "n_clicks")],
    [State("modal-upload_text_from_web-nlp1", "is_open")],
)(toggle_modal)


@app.callback(
    Output("text-nlp1", "value"),
    [Input("close-upload_text_from_web-nlp1", "n_clicks")],
    [State("input-upload_text_from_web-nlp1", "value")],
)
def extract_text_nlp1(n_click, value):
    if n_click:
        try:
            soup = bs(value)
            visible_text = (" ").join([i.text for i in soup.findAll('p')])# + [i.text for i in soup.findAll('li')])
            visible_text=re.sub("[^a-zA-Z.0-9]", " ", visible_text)
            return visible_text
        except Exception as e:

        	print(e)
        	return "Type Something......."
    return PreventUpdate



@app.callback(
    [Output('text-ner-nlp1', "children")],
    [Input("button-ner-nlp1", "n_clicks")],
    [State("text-nlp1", "value"), State('drop-nlp1', 'value')],
)
def toggle_popover(n_click, input_text, drop_value):
    if n_click:
        doc = nlp(input_text)
        return [html.Div(
                children=render(doc, drop_value), style={"line-height": "3"}
            ) ]
    else:
        return  [html.Div(
                children=['Named Entity Recognition'], style={"line-height": "3"}
                ) ]


@app.callback(
    [Output('text-summary_gensim-nlp1', "children")],
    [Input("button-summary_gensim-nlp1", "n_clicks")],
    [State("text-nlp1", "value"), State('drop2-nlp1', 'value')],
)
def toggle_popover(n_click, input_text, drop_value):
    if n_click:
        if drop_value=='Gensim':

            return [html.Div(
                    children=summarize(input_text), style={"line-height": "3"}
                ) ]
        # elif drop_value=='t5-decoder':
        #     # Concatenating the word "summarize:" to raw text
        #     text = "summarize:" + input_text

        #     input_ids=tokenizer.encode(text, return_tensors='pt', max_length=512)

        #     summary_ids = my_model.generate(input_ids, max_length=1000, min_length=250)

        #     t5_summary = tokenizer.decode(summary_ids[0])
        #     return [html.Div(
        #             children=t5_summary, style={"line-height": "3"}
        #         ) ]
        elif drop_value=='KL-Sum':
            parser=PlaintextParser.from_string(input_text,Tokenizer('english'))
            kl_summarizer=KLSummarizer()
            kl_summary=kl_summarizer(parser.document,sentences_count=3)
            summary = [str(sentence) for sentence in kl_summary]
            return [html.Div(
                    children=summary, style={"line-height": "3"}
                )]
        
        elif drop_value=='Luhn':
            parser=PlaintextParser.from_string(input_text,Tokenizer('english'))
            luhn_summarizer=LuhnSummarizer()
            luhn_summary=luhn_summarizer(parser.document,sentences_count=3)
            summary = [str(sentence) for sentence in luhn_summary]
            return [html.Div(
                    children=summary, style={"line-height": "3"}
                )]
        
        elif drop_value=='LSA':
            parser=PlaintextParser.from_string(input_text,Tokenizer('english'))
            lsa_summarizer=LsaSummarizer()
            lsa_summary= lsa_summarizer(parser.document,3)
            summary = [str(sentence) for sentence in lsa_summary]
            return [html.Div(
                    children=summary, style={"line-height": "3"}
                )]
        elif drop_value=='LexRank':
            my_parser = PlaintextParser.from_string(input_text,Tokenizer('english'))
            lex_rank_summarizer = LexRankSummarizer()
            lexrank_summary = lex_rank_summarizer(my_parser.document,sentences_count=2)
            summary = [str(sentence) for sentence in lexrank_summary]
            return [html.Div(
                    children=summary, style={"line-height": "3"}
                )]

        else:
            # Extractive Summarization:
            doc2=nlp(input_text)
            len(list(doc2.sents))

            keyword = []
            stopwords = list(STOP_WORDS)
            pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
            for token in doc2:
                if(token.text in stopwords or token.text in punctuation):
                    continue
                if(token.pos_ in pos_tag):
                    keyword.append(token.text)



            freq_word = Counter(keyword)



            max_freq = Counter(keyword).most_common(1)[0][1]
            for word in freq_word.keys():  
                    freq_word[word] = (freq_word[word]/max_freq)

            sent_strength={}
            for sent in doc2.sents:
                for word in sent:
                    if word.text in freq_word.keys():
                        if sent in sent_strength.keys():
                            sent_strength[sent]+=freq_word[word.text]
                        else:
                            sent_strength[sent]=freq_word[word.text]
            summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
            final_sentences = [ w.text for w in summarized_sentences ]
            summary = ' '.join(final_sentences)
            return [html.Div(
                    children=summary, style={"line-height": "3"}
                )]
        
    else:
        return [html.Div(
                    children=['Summary'], style={"line-height": "3"}
                )]

@app.callback(
    [Output('wordcloud-nlp1', 'src'), Output('table-nlp-1', 'data'), Output('table_hidden_nlp1', 'hidden')],
    Input('button-wordcloud-nlp1', 'n_clicks'),
    [State("text-nlp1", "value")]
)
def get_word_cloud(nclick, input_text):
    if nclick:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams['figure.dpi'] = 200
        vocab, num = getvocab(data=input_text, from_data=True)
        data_vocab = get_data(vocab, num)
        initial_x = 0.00
        initial_y = 0.90
        y_step_down = 0.17
        buf = io.BytesIO() # in-memory files
        fig, ax = plt.subplots(1,1,figsize=(15, 5))
        background_color = text_color[tc][4]
        fig.patch.set_facecolor(background_color)
        spance_taken_in_x = initial_x
        old = 0 
        for key, value in vocab.items():
            new = getcumsum(data_vocab, key, 'cumsum_new')
            sizef = getcumsum(data_vocab, key, 'ulter_size')
            if new < old:
                spance_taken_in_x = initial_x
                initial_y -= y_step_down
            ax=plt.text(spance_taken_in_x, initial_y, key.strip(), size = sizef ,c = list(getcumsum(data_vocab, key, 'color'))[0],
                     ha = 'left', va = 'baseline', wrap = True)
            spance_taken_in_x = new
            old = new
        plt.axis('off')
        plt.savefig(buf, format = "png") # save to the above file object
        data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
        exp = data_vocab.iloc[:,:2]
        exp.columns = ['Word','Freq']
        exp.sort_values(by='Freq',ascending=False, inplace=True)

        return ["data:image/png;base64,{}".format(data),exp.to_dict('records'), False]
    else:
        return [" ", [{'Word': 'Adjective','Freq':1}], True]
    
