
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales.csv"
MODEL_PATH = BASE_DIR / "models" / "sales_model.joblib"

st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Analytics & Prediction")
st.write("Explore sales performance and predict product demand.")

if not DATA_PATH.exists():
    st.error("Dataset not found. Add data/sales.csv first.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model not found. Run python train_model.py first.")
    st.stop()

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip().str.lower()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["quantity"] = pd.to_numeric(
        data["quantity"], errors="coerce"
    )
    data["unit_price"] = pd.to_numeric(
        data["unit_price"], errors="coerce"
    )

    data = data.dropna(
        subset=["date", "quantity", "unit_price",
                "product", "region", "salesperson"]
    ).copy()

    data["sales_amount"] = (
        data["quantity"] * data["unit_price"]
    )

    return data

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
bundle = load_model()
model = bundle["model"]

if df.empty:
    st.error("No valid sales records were found.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filter Sales")

regions = sorted(df["region"].unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Select region",
    regions,
    default=regions
)

products = sorted(df["product"].unique().tolist())
selected_products = st.sidebar.multiselect(
    "Select product",
    products,
    default=products
)

filtered = df[
    df["region"].isin(selected_regions)
    & df["product"].isin(selected_products)
].copy()

if filtered.empty:
    st.warning("No sales match these filters.")
    st.stop()

# Dashboard metrics
total_sales = filtered["sales_amount"].sum()
total_quantity = filtered["quantity"].sum()
total_orders = (
    filtered["order_id"].nunique()
    if "order_id" in filtered.columns
    else len(filtered)
)

a, b, c = st.columns(3)

a.metric("Total Sales", f"${total_sales:,.2f}")
b.metric("Units Sold", f"{total_quantity:,.0f}")
c.metric("Orders", f"{total_orders:,}")

st.divider()

# Sales over time
st.subheader("Sales Trend")

daily_sales = (
    filtered.groupby(filtered["date"].dt.date)["sales_amount"]
    .sum()
    .sort_index()
)

st.line_chart(daily_sales)

left, right = st.columns(2)

with left:
    st.subheader("Top Products by Sales")

    product_sales = (
        filtered.groupby("product")["sales_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(product_sales)

with right:
    st.subheader("Sales by Region")

    region_sales = (
        filtered.groupby("region")["sales_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(region_sales)

st.subheader("Sales Records")
st.dataframe(filtered, use_container_width=True)

# Prediction form
st.divider()
st.header("🤖 Predict Product Quantity")

st.write(
    "Estimate how many units may sell for a selected "
    "product, price, region and date."
)

with st.form("prediction_form"):
    product = st.selectbox(
        "Product",
        bundle["products"]
    )

    region = st.selectbox(
        "Region",
        bundle["regions"]
    )

    salesperson = st.selectbox(
        "Salesperson",
        bundle["salespeople"]
    )

    unit_price = st.number_input(
        "Unit price",
        min_value=0.0,
        value=max(0.0, bundle["price_min"]),
        step=1.0
    )

    sale_date = st.date_input("Expected sale date")

    submitted = st.form_submit_button("Predict Quantity")

if submitted:
    if not (
        bundle["price_min"]
        <= unit_price
        <= bundle["price_max"]
    ):
        st.warning(
            "This price is outside the training dataset's "
            "price range, so the prediction may be unreliable."
        )

    input_data = pd.DataFrame([{
        "product": product,
        "region": region,
        "salesperson": salesperson,
        "unit_price": unit_price,
        "month": sale_date.month,
        "weekday": sale_date.weekday()
    }])

    predicted_quantity = max(
        0, float(model.predict(input_data)[0])
    )

    st.success(
        f"Estimated quantity: {predicted_quantity:.1f} units"
    )

    st.metric(
        "Estimated sales value",
        f"${predicted_quantity * unit_price:,.2f}"
    )

    st.caption(
        "Estimates are based on historical sample data and "
        "may not accurately predict future sales."
    )
