import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="H&M Akıllı Stilist", page_icon="❤️", layout="centered")
st.title("H&M Akıllı Stil Asistanı")
st.write("Sadece neye ihtiyacın olduğunu yaz, sana isteklerine en uygun kıyafetleri bulayım🪄")

@st.cache_resource
def yapay_zekayi_yukle():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def veriyi_hazirla():
    df_tum = pd.read_csv('articles.csv')
    df = df_tum.sample(n=10000, random_state=42)
    secilen_sutunlar = ['article_id', 'prod_name', 'product_group_name', 'colour_group_name', 'detail_desc']
    df = df[secilen_sutunlar].dropna()
    return df

@st.cache_data
def vektorleri_olustur(df, _model):
    return _model.encode(df['detail_desc'].tolist())

with st.spinner("Yapay zeka modeli ve veri seti başlatılıyor, lütfen bekleyiniz..."):
    model = yapay_zekayi_yukle()
    df = veriyi_hazirla()
    vektorler = vektorleri_olustur(df, model)

st.success("Sistem Hazır! Hadi arama yapalım.")

st.markdown("### Ne tarz bir kıyafet arıyorsun?")
arama_metni = st.text_input("Aramak istediğin stili İngilizce olarak buraya yaz:", placeholder="Örn: Something cozy to wear at home while reading a book")

if arama_metni:
    arama_vektoru = model.encode(arama_metni)
    benzerlik_skorlari = util.cos_sim(arama_vektoru, vektorler)[0]
    en_iyi_5_indeks = torch.topk(benzerlik_skorlari, k=5).indices.tolist()

    st.markdown("---")
    st.subheader("✨ İşte Senin İçin Seçtiğim Ürünler:")
    
    for i in en_iyi_5_indeks:
        urun_adi = df.iloc[i]['prod_name']
        kategori = df.iloc[i]['product_group_name']
        aciklama = df.iloc[i]['detail_desc']
        skor = benzerlik_skorlari[i].item()
        
        sol_kolon, sag_kolon = st.columns([1, 3])
        
        with sol_kolon:
            kategori_yazi = str(kategori).replace(' ', '+')
            guvenilir_foto_url = f"https://placehold.co/300x400/f8f9fa/6c757d?text=Gorsel+Bekleniyor\\n{kategori_yazi}"
            st.image(guvenilir_foto_url, use_container_width=True)
                
        with sag_kolon:
            st.markdown(f"#### 👕 {urun_adi}")
            st.caption(f"**Kategori:** {kategori} | **Yapay Zeka Uyum Skoru:** %{int(skor*100)}")
            st.write(f"📝 {aciklama}")
            st.info(" ")