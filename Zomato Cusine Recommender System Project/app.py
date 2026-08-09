import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the precomputed data
similarity = pickle.load(open("dishes_similarity.pkl", "rb"))
new_df = pickle.load(open("dishes.pkl", "rb"))

def get_recommendations(dish_name, top_n=10):
    try:
        dish_index = new_df[new_df["dish_liked"] == dish_name].index[0]
        distances = similarity[dish_index]
        dishes_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:top_n+1]

        recommendations = []
        for idx, score in dishes_list:
            rec_dish = new_df.iloc[idx].dish_liked
            if rec_dish != dish_name:
                recommendations.append({
                    "dish": rec_dish,
                    "cosine_similarity": round(score, 2)
                })
        return recommendations
    except IndexError:
        return None

@app.route('/recommend', methods=['GET'])
def recommend():
    dish = request.args.get('dish')
    if not dish:
        return jsonify({"error": "Missing 'dish' parameter"}), 400

    recs = get_recommendations(dish)
    if recs is None:
        return jsonify({"error": f"Dish '{dish}' not found"}), 404

    return jsonify({"recommendations": recs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)