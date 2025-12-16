import gradio as gr
import joblib
import numpy as np

# Charger le modèle et le scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Ordre des variables
feature_order = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI']

scaler_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI']


# Fonction de prédiction
def predict_hypertension(age, gender, height, weight, ap_hi, ap_lo,
                         cholesterol, gluc, smoke, alco, active):

    height_m = height / 100
    BMI = weight / (height_m ** 2)

    data = [age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active, BMI]

    X = np.array(data).reshape(1, -1)

    X_scaled = X.copy()
    idx = [feature_order.index(f) for f in scaler_features]
    X_scaled[:, idx] = scaler.transform(X[:, idx])

    proba = model.predict_proba(X_scaled)[0][1]
    pred = int(proba >= 0.65)

    diagnostic = (
        "🔴 **Risque ÉLEVÉ d'hypertension**"
        if pred == 1 else
        "🟢 **Profil sain**"
    )

    return (
        f"### 🩺 Diagnostic\n{diagnostic}",
        f"### 📈 Probabilité d'hypertension\n**{proba:.2%}**",
        f"### ⚖️ Indice de Masse Corporelle (BMI)\n**{BMI:.2f}**"
    )



# -----------------------------
#   Interface Pro & Moderne
# -----------------------------

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
        # 🔍 Prédiction du Risque d'Hypertension  
        ### Analyse intelligente des données cliniques  
        ---
        """
    )

    with gr.Row():

        # Colonne gauche : Inputs
        with gr.Column(scale=1):
            gr.Markdown("## 🧑‍⚕️ Informations Patient")
            age = gr.Number(label="Âge", value=40)
            gender = gr.Radio([0, 1], label="Genre (0 = Femme, 1 = Homme)", value=1)
            height = gr.Number(label="Taille (cm)", value=170)
            weight = gr.Number(label="Poids (kg)", value=70)

            gr.Markdown("---")
            ap_hi = gr.Number(label="Pression systolique (ap_hi)", value=120)
            ap_lo = gr.Number(label="Pression diastolique (ap_lo)", value=80)

            gr.Markdown("---")
            cholesterol = gr.Radio([1, 2, 3], label="Cholestérol")
            gluc = gr.Radio([1, 2, 3], label="Glycémie (gluc)")
            smoke = gr.Radio([0, 1], label="Fumeur ?")
            alco = gr.Radio([0, 1], label="Alcool")
            active = gr.Radio([0, 1], label="Actif physiquement ?")

            btn = gr.Button("🔎 Lancer l'analyse", variant="primary")

        # Colonne droite : Résultats
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Résultats de l'analyse")

            diagnosis = gr.Markdown("")
          
            proba_output = gr.Markdown("")
            bmi_output = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### ℹ️ Résumé Patient (auto-calculé)")

            summary = gr.Markdown("Aucun résumé pour le moment.")

        # Fonction du bouton
        def update_summary(age, gender, height, weight, ap_hi, ap_lo,
                           cholesterol, gluc, smoke, alco, active):
            text = f"""
            **Âge :** {age} ans  
            **Genre :** {"Homme" if gender==1 else "Femme"}  
            **Taille :** {height} cm  
            **Poids :** {weight} kg  
            **Pression :** {ap_hi}/{ap_lo} mmHg  
            **Cholestérol :** {cholesterol}  
            **Glycémie :** {gluc}  
            **Fumeur :** {smoke}  
            **Alcool :** {alco}  
            **Actif :** {active}  
            """
            return text

        btn.click(
            predict_hypertension,
            inputs=[age, gender, height, weight, ap_hi, ap_lo,
                    cholesterol, gluc, smoke, alco, active],
            outputs=[diagnosis, proba_output, bmi_output]
        )

        btn.click(
            update_summary,
            inputs=[age, gender, height, weight, ap_hi, ap_lo,
                    cholesterol, gluc, smoke, alco, active],
            outputs=summary
        )


app.launch(share=True)
