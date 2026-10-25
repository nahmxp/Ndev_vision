from io import BytesIO
import streamlit as st
from PIL import Image, ImageDraw
from core.vision_matcher import match_template, prepare_image
st.set_page_config(page_title="Ndev Vision Lab",layout="wide")
st.title("Ndev Vision Lab")
st.caption("From-scratch coordinate and scale matching. No pretrained model or external vision API.")
with st.sidebar:
    st.header("Search settings")
    mode=st.selectbox("Comparison signal",["pixels","edges"],help="Edges are less sensitive to brightness changes.")
    min_scale,max_scale=st.slider("Reference scale range",.1,2.0,(.5,1.5),.1)
    steps=st.slider("Scale samples",1,31,11); stride=st.slider("Coordinate stride",1,16,4,help="Smaller is slower and more precise.")
    max_side=st.slider("Working image limit",200,1000,600,50)
a,b=st.columns(2)
with a: search_upload=st.file_uploader("1. Larger image to search",["png","jpg","jpeg","webp"])
with b: reference_upload=st.file_uploader("2. Image or object to find",["png","jpg","jpeg","webp"])
if search_upload and reference_upload:
    read=lambda item: Image.open(BytesIO(item.getvalue())).convert("RGB")
    search,reference=prepare_image(read(search_upload),max_side),prepare_image(read(reference_upload),max_side)
    a,b=st.columns(2); a.image(search,caption=f"Search: {search.width} x {search.height}",use_container_width=True); b.image(reference,caption=f"Reference: {reference.width} x {reference.height}",use_container_width=True)
    if st.button("Run coordinate and scale search",type="primary",use_container_width=True):
        bar,best=st.progress(0,text="Preparing comparisons"),st.empty()
        def update(done,total,scale,score):
            bar.progress(done/total,text=f"Scale {scale:.2f} - {done:,}/{total:,} comparisons"); best.metric("Best similarity so far",f"{score:.1%}")
        try: result=match_template(search,reference,min_scale=min_scale,max_scale=max_scale,scale_steps=steps,stride=stride,mode=mode,progress=update)
        except ValueError as error: st.error(str(error))
        else:
            bar.progress(1.0,text="Search complete")
            c,d,e,f=st.columns(4); c.metric("Similarity",f"{result.score:.1%}"); d.metric("Scale",f"{result.scale:.2f}x"); e.metric("Coordinate",f"({result.x}, {result.y})"); f.metric("Comparisons",f"{result.comparisons:,}")
            output=search.convert("RGB"); ImageDraw.Draw(output).rectangle((result.x,result.y,result.x+result.width,result.y+result.height),outline="red",width=max(2,search.width//200))
            st.image(output,caption="Red rectangle: best match",use_container_width=True)
            st.info("Template matching, not semantic recognition. It works best for similar pose, crop, and lighting.")
else: st.info("Upload both images. First test: crop part of the search image and use it as the reference.")
