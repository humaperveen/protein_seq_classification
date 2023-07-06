#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import regex
import plotly.express as px
from collections import Counter

# # set background image
import base64

@st.cache_data()  
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file): 
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


# load the saved model
model = joblib.load('pdb_lr_model.joblib')
labelencoder = joblib.load('label_encoder1.joblib')

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
         

def get_code_freq(seq):
 
  codes_dict= Counter(seq)
  
  st.write(f'Total unique codes: {len(codes_dict.keys())}')

  df = pd.DataFrame({'Code': list(codes_dict.keys()), 'Freq': list(codes_dict.values())})
  return df.sort_values('Freq', ascending=False).reset_index()[['Code', 'Freq']]

def plot_code_freq(df):
  fig = px.bar(df, x='Code', y='Freq', title = "Code frequency", text_auto=True, labels={'Freq':'Frequency'},
               color='Code', height=600, width = 1000)
  fig.update_layout(showlegend=False, title_x=0.5)
  st.plotly_chart(fig)


# predict function for protein sequence
def predict_seq(seq):
  if seq.lower():
    seq = seq.upper()
    # remove uncommon amino acids
    seq =regex.sub(r"[XUZOB]", "", str(seq))
    # Preprocess the input seq using the loaded CountVectorizer
    seq_dtm = model['vect'].transform([seq])

    # Make predictions
    result = model['lr'].predict(seq_dtm)
    pred = labelencoder['le'].inverse_transform(result)
    score = model['lr'].predict_proba(seq_dtm) #[:,1]
    ids = score.argmax(1).item()
    confidence = score[0,ids]
  else:
    # remove uncommon amino acids
    seq =regex.sub(r"[XUZOB]", "", str(seq))
    # Preprocess the input seq using the loaded CountVectorizer
    seq_dtm = model['vect'].transform([seq])

    # Make predictions
    result = model['lr'].predict(seq_dtm)
    pred = labelencoder['le'].inverse_transform(result)
    score = model['lr'].predict_proba(seq_dtm) #[:,1]
    ids = score.argmax(1).item()
    confidence = score[0,ids]
  return (pred[0]), confidence #(score[0])

# Streamlit app code

# use the full page instead of a narrow central column
st.set_page_config(layout = "wide")
set_png_as_page_bg('background_10.png')
st.markdown("<h1 style='text-align: center; color: black;'>Protein Sequence Classification</h1>", unsafe_allow_html=True)
st.write("Protein sequence classification is based on Natural Language Processing (NLP). The protein data set retrieved from Research Collaboratory for Structural Bioinformatics (RCSB) Protein Data Bank (PDB). Model is trained on scikit-learn Logistic Regression classifier (weighted f1-score: 0.73 and macro f1-score: 0.61 with the accuracy 74%) for 95 classes of protein.")
st.write("**95 protein classes are as follow:** allergen, antibiotic/hydrolase, antibiotic/transferase, antimicrobial protein, apoptosis, biosynthetic protein, biotin-binding protein, blood clotting, calcium-binding protein, cell adhesion, cell cycle, cell invasion, chaperone, contractile protein, cytokine, de novo protein, dna binding protein, electron transport, electron transport/oxidoreductase, electron transport/photosynthesis, endocytosis, endocytosis/exocytosis, flavoprotein, fluorescent protein, gene regulation, growth factor/hormone, hormone, hormone receptor, hydrolase, hydrolase /o-glycosyl, hydrolase inhibitor, hydrolase/hydrolase inhibitor, hydrolase/immune system, hydrolase/inhibitor, hydrolase/o-glycosyl, hydrolase/protein binding, hydrolase/signaling protein, hydrolase/transferase, hydrolase/transport protein, immune system, immune system/viral protein, immune system/virus, immunoglobulin, inhibitor/transferase, isomerase, isomerase/isomerase inhibitor, lectin, ligase, ligase/ligase inhibitor, lipid binding protein, lipid transport, luminescent protein, lyase, lyase/lyase inhibitor, membrane protein, membrane protein/transport protein, metal binding protein, metal transport, motor protein, nuclear protein, oxidoreductase, oxidoreductase/oxidoreductase inhibitor, oxygen storage, oxygen storage/transport, oxygen transport, peptide binding protein, phosphotransferase, photosynthesis, plant protein, protein binding, protein transport, receptor, replication, ribosomal protein, ribosome, rna binding protein, signaling protein, signaling protein/transferase, splicing, structural genomics/unknown function, structural protein, sugar binding protein, toxin, transcription, transcription regulation, transcription regulator, transcription/transcription inhibitor, transcription/transferase, transferase, transferase/transferase inhibitor, translation, transport protein, unknown function, viral protein, and virus.")

col1, col2 = st.columns(2, gap = 'large')

col1.header("Enter the protein sequence here:")
# Input text from the user
user_input = col1.text_area(label = " ", value="", height=100, label_visibility='collapsed', key="mytext")


col2.header("Select the protein sequence here:")
option = col2.selectbox(' ',
('SMIKQRTLKNIIRATGVGLHSGEKVYLTLKPAPVDTGIVFCRTDLDPVVEIPARAENVGETTMSTTLVKGDVKVDTVEHLLSAMAGLGIDNAYVELSASEVPIMDGSAGPFVFLIQSAGLQEQEAAKKFIRIKREVSVEEGDKRAVFVPFDGFKVSFEIDFDHPVFRGRTQQASVDFSSTSFVKEVSRARTFGFMRDIEYLRSQNLALGGSVENAIVVDENRVLNEDGLRYEDEFVKHKILDAIGDLYLLGNSLIGEFRGFKSGHALNNQLLRTLIADKDAWEVVTFEDARTAPISYMRPAAAV', 'MEHTIAVIPGSFDPITYGHLDIIERSTDRFDEIHVCVLKNSKKEGTFSLEERMDLIEQSVKHLPNVKVHQFSGLLVDYCEQVGAKTIIRGLRAVSDFEYELRLTSMNKKLNNEIETLYMMSSTNYSFISSSIVKEVAAYRADISEFVPPYVEKALKKKFK', 'TIKEMPQPKTFGELKNLPLLNTDKPVQALMKIADELGEIFKFEAPGRVTRYLSSQRLIKEACDESRFDKNLSQALKFVRDFAGDGLFTSWTHEKNWKKAHNILLPSFSQQAMKGYHAMMVDIAVQLVQKWERLNADEHIEVPEDMTRLTLDTIGLCGFNYRFNSFYRDQPHPFITSMVRALDEAMNKLQRANPDDPAYDENKRQFQEDIKVMNDLVDKIIADRKASGEQSDDLLTHMLNGKDPETGEPLDDENIRYQIITFLIAGHETTSGLLSFALYFLVKNPHVLQKAAEEAARVLVDPVPSYKQVKQLKYVGMVLNEALRLWPTAPAFSLYAKEDTVLGGEYPLEKGDELMVLIPQLHRDKTIWGDDVEEFRPERFENPSAIPQHAFKPFGNGQRACEGQQFALHEATLVLGMMLKHFDFEDHTNYELDIKETLTLKPEGFVVKAKSKKIPLGGIPSPSTEQSAKKV', 'AVCCICNDGECQNSNVILFCDMCNLAVHQECYGVPYIPEGQWLCRRCLQSPSRAVDCALCPNKGGAFKQTDDGRWAHVVCALWIPEVCFANTVFLEPIDSIEHIPPARWKLTCYICKQRGSGACIQCHKANCYTAFHVTCAQQAGLYMKMEPVRETGANGTSFSVRKTAYCDIHTPP', 'MNELVDTTEMYLRTIYDLEEEGVTPLRARIAERLDQSGPTVSQTVSRMERDGLLRVAGDRHLELTEKGRALAIAVMRKHRLAERLLVDVIGLPWEEVHAEACRWEHVMSEDVERRLVKVLNNPTTSPFGNPIPGLVELGV', 'IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM', 'GLSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSKHPGDFGADAQGAMTKALELFRNDIAAKYKELGFQG', 'GSHMNQCPEHSQLTTLGVDGKEFPEVHLGQWYFIAGAAPTKEELATFDPVDNIVFNMAAGSAPMQLHLRATIRMKDGLCVPRKWIYHLTEGSTDLRTEGRPDMKTELFSSSCPGGIMLNETGQGYQRFLLYNRSPHPPEKCVEEFKSLTSCLDSKAFLLTPRNQEACELSNN', 'GDPIADMIDQTVNNQVNRSLTALQVLPTAANTEASSHRLGTGVVPALQAAETGASSNASDKNLIETRCVLNHHSTQETAIGNFFSRAGLVSIITMPTTGTQNTDGYVNWDIDLMGYAQLRRKCELFTYMRFDAEFTFVVAKPNGELVPQLLQYMYVPPGAPKPTSRDSFAWQTATNPSVFVKMTDPPAQVSVPFMSPASAYQWFYDGYPTFGEHLQANDLDYGQCPNNMMGTFSIRTVGIEKSPHSITLRVYMRIKHVRAWIPRPLRNQPYLFKTNPNYKGNDIKCTSTSRDKITTL',
'MVDVGGKPVSRRTAAASATVLLGEKAFWLVKENQLAKGDALAVAQIAGIMAAKQTSALIPLCHPIPLDRVAVSLELVEPGWRVVVTATCVASGRTGVEMEALTAASLAALALYDMCKAVTRDIVIQDVRLLSKTGG', 'MGSSHHHHHHSSGLVPRGSHMNATIREILAKFGQLPTPVDTIADEADLYAAGLSSFASVQLMLGIEEAFDIEFPDNLLNRKSFASIKAIEDTVKLILDGKEAA', 'MRIGYGEDSHRLEEGRPLYLCGLLIPSPVGALAHSDGDAALHALTDALLSAYGLGDIGLLFPDTDPRWRGERSEVFLREALRLVEARGAKLLQASLVLTLDRPKLGPHRKALVDSLSRLLRLPQDRIGLTFKTSEGLAPSHVQARAVVLLDG', 'RIIPVKINEGDVVHRSIEEYIRRNSLKGGIITGIGGLMEAVIGFYSPESKTYLEKRIKSSGSVIEVASLQGNYLVKRNGEVSIHIHVVAGFENTTVAGHLIHGTAKPMLEVFLIEIGE', 'VLDVACGTCDVAMEARNQTGDAAFIIGTDFSPGMLTLGLQKLKKNRRFATIPLVCANALALPFQSTHFDAVLIAFGIRNIMDRKGALKQFHDALKPGG', 'VPRGPGYDNPAYQGCAITGAQVGSLRTPGPTYLQTTYEYSRSNLWRNFGVVIAFTVLYILVTSFGSEVFNFTNSGGGALEFKRSKSAKNK', 'GAVRVDVSGGLGTDAMVVSSYLNTDKSLVTVIVNADNQDRDISLAISGGQPAGAVSVYETSAEHDLAPVRNAGADGRLAVKKQSIVTI', 'DVPPYVMAQGNHARPFGVNLEGLKRRGFDKPTMHVIRNIYKMLYRSGKTLEEVLPEIEQIAETDSAISFFVEFFKRSTRGII', 'KDHTTLVAAVKAAGLVPTLESKGPFTVFAPTNAAFGKLPAGTVDNLVKPENKATLTKILTYHVVPGKLEASDLTDGKKLKTAEGEELTVKKMDGKTWIVDAKGGTSMVTISNVNQSNGVIHVVDTVLMP', 'DTVWLVWFCIQIPVILCVDIVDKYPAWLCANPGAPLHALHRFRQWYIATHNDPVVQWTPATHPLLSNGGGSWVPLFFWIELVFTLPTVLYAVYRLGFVRGRNRAPLGGTTGPLELLLLVYALETALTTAVCIHNISYWDPSIYSSAQKNTFRFQLMGPWLAMPSLLFLDMYSR', 'AANKADLASDENIEALKVLGAVPTIAAGELALKSAAHAKILRYLPGDSSFAPIEGAKLSAPQVKALTMIAEHMKKFGSTGVQEILNKIVFEDIGMIVVYPVED', 'SIALVWFIKLCTSEEAKSMVAGLQIVFSNNTDLSSDGKLPVLILDNGTKVSGYVNIVQFLHKNICTSKYEKGTDYEEDLAIVGKKDRLLEYSLLNYVDVEISRLTDYQLFLNTKNYNEYTKKLFSKLLYFPMWYNTPLQLR'), 0, key='myoption')


# Create a predict button
if col1.button("Predict", key = "1"):  
    if user_input.isalpha():
        res, conf_score = predict_seq(user_input)        

        st.header("Prediction")

        # Display the predicted protein class
        st.subheader("Class: " + str(res))
        
        # Display the confidance score
        st.subheader("Confidence score: {} %" .format(str(round(conf_score*100, 2))))

        
        # Generate chart for the input sequence
        seq_code_df = get_code_freq(user_input)
        # Display chart
        plot_code_freq(seq_code_df)
    elif user_input.isalnum() or user_input.isnumeric():
        col1.write("Please enter alphabetic protein sequence.")
    
if col2.button("Predict", key = "2"):
    if option:
        res, conf_score = predict_seq(option)
    st.header("Prediction")

    # Display the predicted protein class
    st.subheader("Class: " + str(res))
    
    # Display the confidance score
    st.subheader("Confidence score: {} %" .format(str(round(conf_score*100, 2))))

    # Generate chart for the input sequence
    seq_code_df = get_code_freq(option)
    # Display chart
    plot_code_freq(seq_code_df)

