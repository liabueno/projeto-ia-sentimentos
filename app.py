import os

port = int(os.environ.get("PORT", 5000))

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Dados de treinamento simples
frases = ["eu amei esse filme", "foi uma ótima experiência", "eu odiei isso", "péssimo serviço", "gostei muito", "detestei completamente"]
sentimentos = ["positivo", "positivo", "negativo", "negativo", "positivo", "negativo"]

# Vetorização (transforma texto em números)
vetorizador = CountVectorizer()
X = vetorizador.fit_transform(frases)

# Modelo de aprendizado
modelo = MultinomialNB()
modelo.fit(X, sentimentos)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        frase_nova = request.form["frase"]
        X_novo = vetorizador.transform([frase_nova])
        resultado = modelo.predict(X_novo)[0]
    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)