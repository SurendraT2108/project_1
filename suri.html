from flask import Flask, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

app = Flask(__name__)

wrong_answers = [
    "Mean and median are same",
    "Median is average of all values",
    "Variance is square root of data",
    "Standard deviation and variance are same",
    "Mean is middle value",
    "Standard deviation is mean"
]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(wrong_answers)

model = KMeans(n_clusters=2, random_state=42)
labels = model.fit_predict(X)

clusters = {}
for i, label in enumerate(labels):
    clusters.setdefault(label, []).append(wrong_answers[i])

html = """
<!DOCTYPE html>
<html>
<head>
<title>Concept Confusion Mapper</title>
<style>
body {
    font-family: Arial;
    background-color: #f4f6f8;
    padding: 20px;
}
h1 {
    text-align: center;
    color: #333;
}
.cluster {
    background: white;
    padding: 15px;
    margin: 20px auto;
    width: 60%;
    border-left: 6px solid #007bff;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
.cluster h2 {
    color: #007bff;
}
li {
    margin: 8px 0;
}
</style>
</head>
<body>

<h1>Concept Confusion Mapper</h1>

{% for cluster, answers in clusters.items() %}
<div class="cluster">
    <h2>Confusion Cluster {{ loop.index }}</h2>
    <ul>
        {% for ans in answers %}
        <li>{{ ans }}</li>
        {% endfor %}
    </ul>
</div>
{% endfor %}

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(html, clusters=clusters)

if __name__ == "__main__":
    app.run(debug=True)
