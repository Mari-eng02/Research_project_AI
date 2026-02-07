import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_input import preprocess_input
import joblib
from gnn_predictor import predict_dependencies, GNNRegressor
import torch
from ai_analysis import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = joblib.load("models/classifier.pkl")
scaler = joblib.load("models/scaler.pkl")
pca_emb = joblib.load("models/pca.pkl")

st.set_page_config(page_title="Tool of Requirements Prioritization and Analysis", layout="wide")
st.title("⭐ Tool of Requirements Prioritization")

########## Manual Input ##########

st.header("✚ Input")

predicted_dependencies = st.session_state.get("dependencies", "-")
predicted_cost = st.session_state.get("cost", "-")
predicted_time = st.session_state.get("time", "-")


with st.form("manual_input"):
    col1, col2, col3 = st.columns(3)
    with col1:
        type = st.selectbox("Type", ["Functional", "Non functional"])
        change = st.selectbox("Change", ["Addition", "Modification", "Deletion"])
    with col2:
        urgency = st.selectbox("Urgency", ["Soon", "Later"])
        origin = st.selectbox("Origin", ["Organization", "Market", "Customer", "Developer knowledge", "Project vision"])
    with col3:
        #cost = st.number_input("Cost", min_value=0)
        #time = st.number_input("Time", min_value=0)
        st.markdown(f"Cost (predicted): `{predicted_cost}`")
        st.markdown(f"Time (predicted): `{predicted_time}`")
        st.markdown(f"N.dependencies (predicted): `{predicted_dependencies}`")

    requirement = st.text_area("Description of requirement")

    submitted = st.form_submit_button("Predict Priority")


if submitted:
    if requirement == '':
        st.error('❗Error: Please insert a requirement description')
    else:
        row = {
            "Type": type,
            "Requirement": requirement,
            "Change": change,
            "Urgency": urgency,
            "Origin": origin,
            "N_dependencies": "",
            "Cost": "",
            "Time": ""
        }

        # dependencies prediction
        model_gnn = GNNRegressor(in_channels=388, hidden_channels=64).to(device)
        state_dict = torch.load("models/gnn_model.pt", map_location=device)
        model_gnn.load_state_dict(state_dict)
        df = pd.read_csv("dataset/full_requirements.csv")

        dependencies, new_embedding, new_cat = predict_dependencies(row, model_gnn, df, device=device)
        st.session_state.dependencies = dependencies
        row["N_dependencies"] = dependencies

        # scaling
        scaled_all = scaler.transform([[dependencies, 0.0, 0.0]])
        scaled_num = scaled_all[:, 0].reshape(1, -1)

        # embedding pca and normalization
        emb_norm = new_embedding.cpu().numpy() / np.linalg.norm(new_embedding.cpu().numpy(), axis=1, keepdims=True)
        emb_reduced = pca_emb.transform(emb_norm)

        # regressor input
        x_reg_input = np.hstack([emb_reduced * 0.5, new_cat.cpu().numpy().astype(float), scaled_num])

        # cost and time prediction
        regressor = joblib.load("models/regressor_cost_time.pkl")
        predicted_cost, predicted_time = regressor.predict([x_reg_input[0]])[0]
        predicted_cost = int(round(predicted_cost))
        predicted_time = int(round(predicted_time))

        row["Cost"] = predicted_cost
        row["Time"] = predicted_time

        st.session_state.cost = predicted_cost
        st.session_state.time = predicted_time

        x_processed = preprocess_input(row)

        proba = classifier.predict_proba([x_processed])[0]
        pred = ["low", "medium", "high"][np.argmax(proba)]

        st.session_state.prediction_result = pred

        # call to Groq API
        with st.spinner("Analysis of requirement with AI..."):
            full_text, plantuml_code = analyze_requirement(requirement, row["Type"])
            st.session_state.text_output = full_text
            st.session_state.plantuml_code = plantuml_code

        st.rerun()

if "prediction_result" in st.session_state:
    pred = st.session_state.prediction_result
    st.subheader("🎯 Result of prediction")
    emoji_map = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢"
    }

    st.markdown(
        f'<p style="padding-left: 40px; font-size:24px;"> ▶️ Priority: {emoji_map[pred]} <strong>{pred.upper()}</strong></p>',
        unsafe_allow_html=True
    )

if "text_output" in st.session_state:
    st.subheader("📄 AI Analysis")
    st.markdown(st.session_state.text_output)

if "plantuml_code" in st.session_state and st.session_state.plantuml_code:
    st.subheader("🏗️ Architectural Diagram")

    encoded = plantuml_encode(st.session_state.plantuml_code)
    plantuml_url = f"https://www.plantuml.com/plantuml/svg/{encoded}"
    st.image(plantuml_url)


st.markdown("<br><br>", unsafe_allow_html=True)


########## CSV Loading ##########

st.header("📁 Requirement upload from CSV")

uploaded_file = st.file_uploader("Upload file .csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ CSV uploaded:")
    st.dataframe(df.head())

    model_gnn = GNNRegressor(in_channels=388, hidden_channels=64).to(device)
    state_dict = torch.load("models/gnn_model.pt", map_location=device)
    model_gnn.load_state_dict(state_dict)
    df_existing = pd.read_csv("dataset/full_requirements.csv")

    regressor = joblib.load("models/regressor_cost_time.pkl")

    n_deps_list, costs_list, times_list = [], [], []
    ai_text_list, ai_plantuml_list = [], []

    for _, row in df.iterrows():
        dep, emb, cat = predict_dependencies(row.to_dict(), model_gnn, df_existing, device=device)
        n_deps_list.append(dep)

        # scaling
        scaled_all = scaler.transform([[dep, 0.0, 0.0]])
        scaled_num = scaled_all[:, 0].reshape(1, -1)

        # embedding pca and normalization
        emb_norm = emb.cpu().numpy() / np.linalg.norm(emb.cpu().numpy(), axis=1, keepdims=True)
        emb_reduced = pca_emb.transform(emb_norm)

        x_reg_input = np.hstack([emb_reduced * 0.5, cat.cpu().numpy().astype(float), scaled_num])

        # cost and time prediction
        predicted_cost, predicted_time = regressor.predict([x_reg_input[0]])[0]
        costs_list.append(int(round(predicted_cost)))
        times_list.append(int(round(predicted_time)))

        # AI analysis
        ai_text, plantuml_code = analyze_requirement(row["Requirement"], row["Type"])
        ai_text_list.append(ai_text)
        ai_plantuml_list.append(plantuml_code)

    df["N_dependencies"] = n_deps_list
    df["Cost"] = costs_list
    df["Time"] = times_list

    df["AI_Analysis"] = ai_text_list
    df["AI_PlantUML"] = ai_plantuml_list

    df_processed = df.apply(preprocess_input, axis=1, result_type="expand")

    df["Priority"] = df_processed.apply(
        lambda row: ["low", "medium", "high"][np.argmax(classifier.predict_proba([row])[0])], axis=1
    )

    st.subheader("🔍 Predictions on file requirements")
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV with Priority",
        data=csv,
        file_name="prioritized_requirements.csv",
        mime="text/csv"
    )
