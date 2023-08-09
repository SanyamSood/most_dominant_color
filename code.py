import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils

def main():
    st.title("Dominant Color Extraction")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(img, channels="BGR", caption="Uploaded Image")

        clusters = st.slider("Select the number of clusters:", 2, 20, 7)

        if st.button("Process"):
            processed_img = process_image(img, clusters)
            st.image(processed_img, channels="BGR", caption="Processed Image")

def process_image(img, clusters):
    org_img = img.copy()
    img = imutils.resize(img, height=200)
    flat_img = np.reshape(img, (-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = zip(percentages, dominant_colors)
    p_and_c = sorted(p_and_c, reverse=True)
    block = np.ones((50, 50, 3), dtype='uint')

    processed_img = org_img.copy()
    rows = 1000
    cols = int((org_img.shape[0] / org_img.shape[1]) * rows)
    img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)
    copy = img.copy()
    cv2.rectangle(copy, (rows // 2 - 350, cols // 2 - 90), (rows // 2 + 350, cols // 2 + 110), (255, 255, 255), -1)
    final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
    cv2.putText(final, 'Most Dominant Colors in the Image', (rows // 2 - 230, cols // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    start = rows // 2 - 305
    for i in range(clusters):
        end = start + 70
        final[cols // 2:cols // 2 + 70, start:end] = p_and_c[i][1]
        cv2.putText(final, str(i + 1), (start + 25, cols // 2 + 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        start = end + 20

    return final

if _name_ == "_main_":
    main()
