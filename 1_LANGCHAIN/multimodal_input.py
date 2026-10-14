from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = init_chat_model('google_genai:gemini-3.8-flash')

message={
    'role':'user',
    'content':[
        {'type':'text',
         'text':'Describe the contents of this image'},
         {'type':'image',
                  'url':'https://img1.hscicdn.com/image/upload/f_auto,t_ds_wide_w_800,q_50/lsci/db/PICTURES/CMS/371400/371467.6.jpg'}

    ]
}

response = model.invoke([message])


chain = model | StrOutputParser()

text_output = chain.invoke([message])

print(text_output)