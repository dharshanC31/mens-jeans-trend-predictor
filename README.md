 👖 Men's Jeans Trend Predictor — Myntra

> Predicting trending products using price, discount & rating signals — built for retail analytics internship applications.

## 🎯 Problem
Myntra lists 35,000+ jeans products. Identifying which products will trend helps brands stock smarter and discount strategically.

## 💡 Key Business Insights
- Trending jeans are **40% cheaper** — avg ₹1,083 vs ₹1,796
- Trending products have **higher discounts** — 56% vs 44%
- Rating alone doesn't predict trends — **price + discount matter more**

## 🤖 Model
- Algorithm: Random Forest Classifier
- Accuracy: **83.7%**
- Features: Price, MRP, Discount %, Rating
- Target: Trending vs Non-Trending (top 25% trend score)

## 📊 Dashboard Features
- Filter by price and rating
- Top 10 trending products table
- Price distribution chart
- Top brands by trending count
- Predict if any new product will trend

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Random Forest)
- Matplotlib, Seaborn
- Streamlit

## 📁 Project Structure
```
trend-predictor-project/
├── data/
│   └── myntra_products.csv
├── notebooks/
│   └── analysis.ipynb
├── app.py
└── README.md
```

## 🚀 How to Run
```
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author
Dharshan C | B.Com Business Analytics |

