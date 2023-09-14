import streamlit as st
from langchain.llms import OpenAI
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.stateful_button import button
import pandas as pd
import openai
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import base64
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import (
    TextAnalyticsClient,
    AnalyzeHealthcareEntitiesAction,
    RecognizePiiEntitiesAction,
)
import os
from azure.search.documents import SearchClient
from azure.appconfiguration.provider import (
    load,
    SettingSelector
)
import requests
import json
import openai
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Connect to key vault
keyVaultName = "aisnomed-kv"
KVUri = f"https://{keyVaultName}.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

################   FUNCTIONS  ######################
def get_session_state():
    session = st.session_state
    if "checkbox_state" not in session:
        session.checkbox_state = {}
    return session

# Define a function to update the checkbox state
def update_checkbox_state(row):
    session_state = get_session_state()
    # print(f"Session state before update: {session_state}")
    session_state.checkbox_state[row["Suggest Codes"]] = row["AorR"]
    # print(f"Session state after update: {session_state}")

# Define a function to get the checkbox state
def get_checkbox_state():
    session_state = get_session_state()
    print(f"Get Checkbox Session state: {session_state}")
    if "checkbox_state" not in session_state:
        session_state.checkbox_state = {}
    return session_state.checkbox_state

# Define a function to clear the checkbox state
def clear_checkbox_state():
    session_state = get_session_state()
    if "checkbox_state" in session_state:
        del session_state.checkbox_state

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Authenticate the text analytics for health client using key and endpoint 
def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=ta_credential)
    return text_analytics_client

# Authenticate the search client using key and endpoint 
def authenticate_search_client():
    s_credential = AzureKeyCredential(searchkey)
    search_client = SearchClient(
            endpoint=searchendpoint, 
            index_name=searchindex,
            credential=s_credential)
    return search_client

###########################################################

# Get Keys
key = client.get_secret("textanalyticskey").value
endpoint = client.get_secret("textanalyticsep").value

searchkey = client.get_secret("cogsearchkey").value
searchendpoint = client.get_secret("cogsearchep").value
searchindex = client.get_secret("cogsearchindex").value

openaideployment = client.get_secret("azureopenaideployment").value
openai.api_type = "azure"
openai.api_base = client.get_secret("azureopenaiep").value
openai.api_version = "2023-05-15"
openai.api_key = client.get_secret("azureopenaikey").value

###########################################################


st.set_page_config(page_title="🩺 Medical Billing")
st.header('🩺 Medical Billing Codes Quickstart 🚑')

#Call sidebar function to implement the sidebar
st.set_option('deprecation.showPyplotGlobalUse', False)


temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1
)
tokens = st.sidebar.slider(
    "Maximum Tokens",
    min_value=64,
    max_value=2048,
    value=2048,
    step=64
)

#Side Bar code
with st.sidebar:
  st.markdown(
    "## How to use\n"
    "1. Enter your medical short hand or clinical note into the top box 📝\n"  # noqa: E501
    "2. Optional: Adjust the temperature and tokens to change the output 🌡\n"
    "3. Press submit 👆🏽\n"
    "4. Accept the suggested MBS billing codes ✅\n"
  )
  st.markdown("---")
  st.markdown("# About")
  st.markdown(
      "This accelerator was designed for Australian Healthcare customers"
      " to leverage AI for the generation of medical billing codes."
  )
  st.markdown(
      "This tool is a work in progress. "
      "You can contribute to the project on [GitHub](https://github.com/courtney-withers/OpenAI_MBSCodes) "  # noqa: E501
      "with your feedback and suggestions💡"
  )
  st.markdown("Made by the Australian Microsoft Healthcare Team")
  st.markdown("---")

##################################################################################
# Define the function to call the OpenAI API and expand the shorthand summary
def shorthand_expander(text, temperature=temperature, tokens=tokens):
    system_prompt = """
    You are an expert Australian medical practitioner and your role is to review a medical shorthand and elaborate it into a full clinical note.
    - Be succient and clear in your response.
    - Ensure there is sufficient information
    - If the input text is not a shorthand, then leave the text as is.
    Now please review the user's shorthand:
    """
    messages = []
    messages.append({"role" : "system", "content" : system_prompt})
    messages.append({"role" : "user", "content" : text})
    response = openai.ChatCompletion.create(
                    engine=openaideployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=tokens,
                    n=1)
    answer = response.choices[0].message.content
    return answer

# Define the function to call the OpenAI API and generate the billing codes based on full clinical notes

def generate_codes(question, temperature=temperature, tokens=tokens):

    system_prompt = """
    Review the clinical note provided, marked up with suggested SNOMED codes and MBS codes, reformat the existing codes and add any additional suggestions or corrections. 
    - Do not provide irrelevant codes. If there are no relevant codes, say "there are no relevant codes".
    Use the following format for your response:
    SNOMED CODE: 
    SNOMED TERM:
    SNOMED CODE: 
    SNOMED TERM:
    SNOMED CODE: 
    SNOMED TERM:
    MBS CODE:
    MBS TERM:
    MBS CODE:
    MBS TERM:
    MBS CODE:
    MBS TERM:
    MBS CODE:
    MBS TERM:
    
    Clinical Note::
    """

    messages = []

    messages.append({"role" : "system", "content" : system_prompt})

    messages.append({"role" : "user", "content" : question})

    response = openai.ChatCompletion.create(
                    engine=openaideployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=tokens,
                    n=1)
    answer = response.choices[0].message.content
    return answer

# Define the function to split the codes generated by OpenAI into SNOMED and MBS codes
def split_codes(data):
    snomed_codes = []
    snomed_terms = []
    mbs_codes = []
    mbs_terms = []
    for line in data.split('\n'):
        if line.startswith('SNOMED CODE:'):
            snomed_codes.append(line.split(': ')[1])
        elif line.startswith('SNOMED TERM:'):
            snomed_terms.append(line.split(': ')[1])
        elif line.startswith('MBS CODE:'):
            mbs_codes.append(line.split(': ')[1])
        elif line.startswith('MBS TERM:'):
            mbs_terms.append(line.split(': ')[1])
    return snomed_codes, snomed_terms, mbs_codes, mbs_terms

def click_button():
    st.session_state.clicked = True

def exp_button():
    st.session_state.export = True

##########   SESSION STATE     #############

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

if 'output_codes' not in st.session_state:
    #Initialize output_codes as empty
    st.session_state.output_codes = ""

#Initialize a variable to note the run time
if 'runtime' not in st.session_state:
    st.session_state.runtime = 1

if 'export' not in st.session_state:
    st.session_state.export = False

if 'selected' not in st.session_state:
    #Initialize selected state as an empty data frame
    st.session_state.selected = pd.DataFrame()

#################   MAIN    ###################
fulltext = ''
with st.form('my_form'):
  shorthandtext = st.text_area('Paste in a Medical shorthand:')
  submit_button = st.form_submit_button(label='Expand and Generate Codes')
  if submit_button:
    fulltext = shorthand_expander(shorthandtext)
    text = fulltext
    click_button()
  text = st.text_area('Here is the full clinical note', fulltext)
  
  if st.session_state.clicked & st.session_state.runtime == 1:
    #prep to call TA4H
    outputstr=""
    mbsoutputstr=""
    documents = [text]
    client = authenticate_client()
    searchclient = authenticate_search_client()

    if text is not None:
        #analyse with TA4H
        poller = client.begin_analyze_healthcare_entities(documents)
        result = poller.result()
        docs = [doc for doc in result if not doc.is_error]
        for idx, doc in enumerate(docs):
            for entity in doc.entities:
                outputstr = outputstr + format(entity.text)
                if entity.category == "TreatmentName":
                    searchresults = searchclient.search(search_text=entity.text)
                    for result in searchresults:
                        mbsoutputstr = mbsoutputstr + "[[MBS: " + result["ItemNum"] +"]]\r\n"
                        break
                    if entity.data_sources is not None:
                      for data_source in entity.data_sources:
                        if (data_source.name == "SNOMEDCT_US"):
                          outputstr = outputstr + " [[SNOMED: " + data_source.entity_id + "]]"
        outputstr = outputstr + "\r\n" + mbsoutputstr 

    openaitxt = text + " \r\n " + outputstr
    output_codes = generate_codes(openaitxt)
    st.session_state.output_codes = output_codes    
    st.text_area("Output", output_codes, height=150)

    #Set the runtime to 0 so that the code doesn't run again
    st.session_state.runtime = 0


    
st.markdown(
"#### SNOMED Code Output  \n"
"Click the checkboxes to accept the relevant SNOMED codes"
)

################## SNOMED CODES ############################
if st.session_state.clicked:
    grid_key = 'snomed_key'
    snomed_codes, snomed_terms, mbs_codes, mbs_terms = split_codes(st.session_state.output_codes)
    snomed_data = {
    'SNOMED CODES': snomed_codes,
    'SNOMED TERMS': snomed_terms
    }
    df = pd.DataFrame(snomed_data)

    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_column("SNOMED CODES", width=50)
    gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridOptions = gd.build()

    grid_table = AgGrid(
        df, 
        width='100%',
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED,
        fit_columns_on_grid_load=True,
    )

    selected_rows = grid_table['selected_rows']


##################    MBS CODES   ############################ 
st.markdown(
"#### MBS Code Selection  \n"
"Click the checkboxes to accept the relevant MBS codes"
)
#PRINTING RETURNED MBS CODES
#Here we want to print a list of the retunred MBS Codes
if st.session_state.clicked:
    grid_key1 = 'mbs_key'
    snomed_codes, snomed_terms, mbs_codes, mbs_terms = split_codes(st.session_state.output_codes)
    MBS_data = {
        'MBS CODES': mbs_codes,
        'MBS TERMS': mbs_terms
    }
    
    #Create the data frame with the data 
    df1 = pd.DataFrame(MBS_data)

    #Set up the Grid
    gd1 = GridOptionsBuilder.from_dataframe(df1)

    #Configure the column settings to fit on the page an enable grouping
    gd1.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gd1.configure_column("MBS CODES", width=50)

    #Configure the selection mode to allow multiple selections and use checkboxes
    gd1.configure_selection(selection_mode='multiple', use_checkbox=True)

    gd1.configure_grid_options(domLayout='normal')
    gridOptions1 = gd1.build()

    grid_table1 = AgGrid(
        df1, 
        width='100%',
        gridOptions=gridOptions1,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED,
        fit_columns_on_grid_load=True,
    )

    df1 = grid_table1['data']
    selected_rows1 = grid_table1['selected_rows']
    selected_df = pd.DataFrame(selected_rows1).apply(pd.to_numeric, errors='coerce')

    st.session_state.selected = selected_df

    #Write out the selected grid table - DEBUGGING 
    # st.write(grid_table1['selected_rows'])

#Debugging statements 
print("The Clicked sessions state is ",st.session_state.clicked)
print("The run time sessions state is ",st.session_state.runtime)

##### Download the selected rows ########

#Save the selected rows in the session state
export_rows=st.session_state.selected

#Pass the session state 
csv = convert_df(export_rows)

st.download_button(label="Download data as CSV",data=csv,file_name='codes.csv',mime='text/csv')














# Extra Code

# export = st.button('Download Selected Rows', on_click=exp_button)

# if export:
#     # selected_rows1 = grid_table['selected_rows']
#     # selected_rows2 = grid_table1['selected_rows']
#     # selected_df = st.dataframe(selected_rows1 + selected_rows2)
#     # print(selected_df)
#     export_rows=st.session_state.selected
#     csv = export_rows.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="selected_rows.csv">Download Selected Rows</a>'
#     st.markdown(href, unsafe_allow_html=True)



#This is the container which will contain the SNOWMED code response
# with stylable_container(
#     key="container_with_border",
#     css_styles="""
#         {
#             border: 1px solid rgba(49, 51, 63, 0.2);
#             border-radius: 0.5rem;
#             padding: calc(1em - 1px)
#         }
#         """,
# ):




#This is the container which will contain the MBS code response
# with stylable_container(
#     key="container_with_border",
#     css_styles="""
#         {
#             border: 1px solid rgba(49, 51, 63, 0.2);
#             border-radius: 0.5rem;
#             padding: calc(1em - 1px)
#         }
#         """,
# ):










# # Apply the update_checkbox_state function to each row in the DataFrame
# data_df.apply(update_checkbox_state, axis=1)


# # Get the checkbox state
# checkbox_state = get_checkbox_state()

# # Create an array with the checkbox values
# checkbox_array = [checkbox_state.get(string, False) for string in random_MBS]

# # Display the checkbox array
# st.write(checkbox_array)

# ######## TEST CODES #################

# data = {
#     'MBS': random_MBS,
#     'Procedure': procedure_array
# }

# df = pd.DataFrame(data)
# gd = GridOptionsBuilder.from_dataframe(df)
# gd.configure_selection(selection_mode='multiple', use_checkbox=True)
# gridoptions = gd.build()

# grid_table = AgGrid(df, height=250, gridOptions=gridoptions,
#                     update_mode=GridUpdateMode.SELECTION_CHANGED)

# st.write('## Selected')
# selected_row = grid_table["selected_rows"]
# st.dataframe(selected_row)