from __future__ import annotations

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💰",
    layout="wide",
)


st.title("E-Commerce Dynamic Pricing Engine")
st.caption("Demand forecasting and price recommendation")


with st.sidebar:
    st.header("Product")

    product_id = st.number_input(
        "Product ID",
        min_value=1,
        value=101,
        step=1,
    )

    category = st.selectbox(
        "Category",
        ["electronics", "home", "beauty", "sports"],
    )

    date = st.date_input("Prediction Date")


tab1, tab2 = st.tabs(
    ["📈 Demand Prediction", "💰 Dynamic Pricing"]
)


with tab1:
    st.subheader("Predict Demand")

    col1, col2, col3 = st.columns(3)

    with col1:
        price = st.number_input(
            "Price",
            min_value=0.01,
            value=700.0,
        )

        competitor_price = st.number_input(
            "Competitor Price",
            min_value=0.01,
            value=720.0,
        )

    with col2:
        discount = st.number_input(
            "Discount (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
        )

        promotion = st.selectbox(
            "Promotion",
            [0, 1],
        )

    with col3:
        inventory = st.number_input(
            "Inventory",
            min_value=0,
            value=400,
            step=1,
        )

        holiday = st.selectbox(
            "Holiday",
            [0, 1],
        )

    recent_sales_text = st.text_area(
        "Last 30 days sales",
        value="65,68,70,67,72,71,69,73,75,70,"
        "68,71,74,76,72,70,69,73,75,77,"
        "74,72,71,76,78,75,73,77,79,80",
        help="Enter at least 30 daily sales values, oldest to newest.",
    )

    if st.button(
        "Predict Demand",
        type="primary",
        key="predict",
    ):
        try:
            recent_sales = [
                float(value.strip())
                for value in recent_sales_text.split(",")
                if value.strip()
            ]

            if len(recent_sales) < 30:
                st.error("Please provide at least 30 sales values.")
            else:
                payload = {
                    "product_id": product_id,
                    "category": category,
                    "price": price,
                    "competitor_price": competitor_price,
                    "discount": discount,
                    "promotion": promotion,
                    "inventory": inventory,
                    "holiday": holiday,
                    "recent_sales": recent_sales,
                    "date": str(date),
                }

                response = requests.post(
                    f"{API_URL}/predict-demand",
                    json=payload,
                    timeout=10,
                )

                if response.ok:
                    result = response.json()

                    st.success("Demand predicted successfully")

                    st.metric(
                        "Predicted Daily Demand",
                        f"{result['predicted_demand']:.2f} units",
                    )
                else:
                    st.error(response.text)

        except ValueError:
            st.error(
                "Recent sales must contain only numbers separated by commas."
            )
        except requests.RequestException:
            st.error(
                "Could not connect to FastAPI. "
                "Make sure the backend is running."
            )


with tab2:
    st.subheader("Dynamic Price Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        current_price = st.number_input(
            "Current Price",
            min_value=0.01,
            value=700.0,
            key="current_price",
        )

        unit_cost = st.number_input(
            "Unit Cost",
            min_value=0.0,
            value=450.0,
            key="unit_cost",
        )

    with col2:
        minimum_price = st.number_input(
            "Minimum Price",
            min_value=0.01,
            value=600.0,
            key="minimum_price",
        )

        maximum_price = st.number_input(
            "Maximum Price",
            min_value=0.01,
            value=800.0,
            key="maximum_price",
        )

    with col3:
        pricing_competitor_price = st.number_input(
            "Competitor Price",
            min_value=0.01,
            value=720.0,
            key="pricing_competitor_price",
        )

        pricing_discount = st.number_input(
            "Discount (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            key="pricing_discount",
        )

    pricing_promotion = st.selectbox(
        "Promotion",
        [0, 1],
        key="pricing_promotion",
    )

    pricing_inventory = st.number_input(
        "Inventory",
        min_value=0,
        value=400,
        step=1,
        key="pricing_inventory",
    )

    pricing_holiday = st.selectbox(
        "Holiday",
        [0, 1],
        key="pricing_holiday",
    )

    pricing_sales_text = st.text_area(
        "Last 30 days sales",
        value="65,68,70,67,72,71,69,73,75,70,"
        "68,71,74,76,72,70,69,73,75,77,"
        "74,72,71,76,78,75,73,77,79,80",
        key="pricing_sales",
        help="Enter at least 30 daily sales values, oldest to newest.",
    )

    if st.button(
        "Recommend Price",
        type="primary",
        key="recommend",
    ):
        try:
            recent_sales = [
                float(value.strip())
                for value in pricing_sales_text.split(",")
                if value.strip()
            ]

            if len(recent_sales) < 30:
                st.error("Please provide at least 30 sales values.")
            elif maximum_price < minimum_price:
                st.error(
                    "Maximum price must be greater than or equal to minimum price."
                )
            else:
                payload = {
                    "product_id": product_id,
                    "category": category,
                    "current_price": current_price,
                    "competitor_price": pricing_competitor_price,
                    "discount": pricing_discount,
                    "promotion": pricing_promotion,
                    "inventory": pricing_inventory,
                    "holiday": pricing_holiday,
                    "unit_cost": unit_cost,
                    "minimum_price": minimum_price,
                    "maximum_price": maximum_price,
                    "recent_sales": recent_sales,
                    "date": str(date),
                }

                response = requests.post(
                    f"{API_URL}/recommend-price",
                    json=payload,
                    timeout=10,
                )

                if response.ok:
                    result = response.json()

                    st.success("Price recommendation generated")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"₹{result['current_price']:.2f}",
                        )

                    with col2:
                        st.metric(
                            "Recommended Price",
                            f"₹{result['recommended_price']:.2f}",
                        )

                    with col3:
                        st.metric(
                            "Expected Revenue",
                            f"₹{result['expected_revenue']:.2f}",
                        )

                    with col4:
                        if result["expected_profit"] is not None:
                            st.metric(
                                "Expected Profit",
                                f"₹{result['expected_profit']:.2f}",
                            )

                    st.subheader("Candidate Price Evaluation")

                    table_data = [
                        {
                            "Price": item["price"],
                            "Predicted Demand": round(
                                item["predicted_demand"],
                                2,
                            ),
                            "Revenue": round(
                                item["revenue"],
                                2,
                            ),
                            "Profit": (
                                round(item["profit"], 2)
                                if item["profit"] is not None
                                else None
                            ),
                        }
                        for item in result["candidate_prices"]
                    ]

                    st.dataframe(
                        table_data,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.error(response.text)

        except ValueError:
            st.error(
                "Recent sales must contain only numbers separated by commas."
            )
        except requests.RequestException:
            st.error(
                "Could not connect to FastAPI. "
                "Make sure the backend is running."
            )