from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
data = pd.read_csv("travel_data.csv")

# --------------------------------------------------
# Collaborative Filtering Setup
# --------------------------------------------------
ratings = pd.read_csv("ratings.csv")

user_item_matrix = ratings.pivot_table(
    index="User_ID",
    columns="Destination",
    values="Rating"
).fillna(0)

item_similarity = cosine_similarity(user_item_matrix.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

def get_top_collaborative_recommendations(destination, top_n=2):
    if destination in item_similarity_df.columns:
        similar_scores = item_similarity_df[destination].sort_values(ascending=False)
        similar_destinations = similar_scores.index.tolist()

        if destination in similar_destinations:
            similar_destinations.remove(destination)

        return similar_destinations[:top_n]

    return []

# --------------------------------------------------
# Map Links
# --------------------------------------------------
destination_links = {
    "Goa": "https://www.google.com/maps/place/Goa",
    "Pondicherry": "https://www.google.com/maps/place/Puducherry",
    "Gokarna": "https://www.google.com/maps/place/Gokarna",
    "Andaman and Nicobar Islands": "https://www.google.com/maps/place/Andaman+and+Nicobar+Islands",
    "Kovalam": "https://www.google.com/maps/place/Kovalam",
    "Manali": "https://www.google.com/maps/place/Manali",
    "Shimla": "https://www.google.com/maps/place/Shimla",
    "Mumbai": "https://www.google.com/maps/place/Mumbai",
    "Delhi": "https://www.google.com/maps/place/Delhi",
    "Jaipur": "https://www.google.com/maps/place/Jaipur",
    "Bangalore": "https://www.google.com/maps/place/Bangalore",
    "Rishikesh": "https://www.google.com/maps/place/Rishikesh",
    "Coorg": "https://www.google.com/maps/place/Coorg",
    "Leh": "https://www.google.com/maps/place/Leh",
    "Amritsar": "https://www.google.com/maps/place/Amritsar",
    "Varanasi": "https://www.google.com/maps/place/Varanasi",
    "Haridwar": "https://www.google.com/maps/place/Haridwar"
}

# --------------------------------------------------
# Home Route
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # ---------------- User Input ----------------
        budget = int(request.form["budget"])
        days = int(request.form["days"])
        group_size = int(request.form["group_size"])
        travel_type = request.form["type"]
        luxury = request.form["luxury"]
        season = request.form["season"]

        # ---------------- Rule-Based Filtering ----------------
        filtered = data[
            (data["Type"] == travel_type) &
            (data["Luxury"] == luxury) &
            (data["Season"] == season)
        ].copy()

        # If no exact match, relax filter
        if filtered.empty:
            filtered = data[data["Type"] == travel_type].copy()

        # Budget similarity ranking
        filtered["Budget_Diff"] = abs(filtered["Budget"] - budget)

        top_matches = filtered.sort_values("Budget_Diff").head(2)

        recommendations = top_matches["Destination"].tolist()

        # ---------------- Collaborative Filtering ----------------
        collab = []
        if recommendations:
            collab = get_top_collaborative_recommendations(recommendations[0], top_n=2)

        final_recommendations = list(dict.fromkeys(
            recommendations + collab
        ))[:2]

        recommendation_details = []
        for place in final_recommendations:
            recommendation_details.append({
                "name": place,
                "map": destination_links.get(place, "#")
            })

        return render_template(
            "index.html",
            recommendations=recommendation_details
        )

    return render_template("index.html")


# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
