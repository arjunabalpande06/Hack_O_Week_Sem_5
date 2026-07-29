import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Customer:
    def __init__(self, customer_id, name, email, age, city):
        self.customer_id = customer_id
        self.name = name
        self.email = email
        self.age = age
        self.city = city

    def to_dict(self):
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "city": self.city,
        }


class Product:
    def __init__(self, product_id, name, category, price, stock):
        self.product_id = product_id
        self.name = name
        self.category = category
        self.price = price
        self.stock = stock

    def to_dict(self):
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "stock": self.stock,
        }


class Order:
    def __init__(self, order_id, customer_id, product_id, quantity, order_date, discount_pct):
        self.order_id = order_id
        self.customer_id = customer_id
        self.product_id = product_id
        self.quantity = quantity
        self.order_date = order_date
        self.discount_pct = discount_pct

    def to_dict(self):
        return {
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "order_date": self.order_date,
            "discount_pct": self.discount_pct,
        }


class ECommerceAnalytics:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.customers_df = self.read_customers()
        self.products_df = self.read_products()
        self.orders_df = self.read_orders()
        self.product_lookup = {row["product_id"]: row["name"] for _, row in self.products_df.iterrows()}
        self.customer_lookup = {row["customer_id"]: row["name"] for _, row in self.customers_df.iterrows()}
        self.merged_df = self.merge_datasets()
        self.recommendation_map = {
            "P001": ["P005", "P003"],
            "P002": ["P001", "P004"],
            "P003": ["P005", "P004"],
            "P004": ["P003", "P005"],
            "P005": ["P001", "P003"],
        }

    def read_customers(self):
        df = pd.read_csv(self.data_dir / "customers.csv")
        customers = [Customer(*row) for row in df.itertuples(index=False, name=None)]
        return pd.DataFrame([c.to_dict() for c in customers])

    def read_products(self):
        df = pd.read_csv(self.data_dir / "products.csv")
        products = [Product(*row) for row in df.itertuples(index=False, name=None)]
        return pd.DataFrame([p.to_dict() for p in products])

    def read_orders(self):
        df = pd.read_csv(self.data_dir / "orders.csv")
        orders = [Order(*row) for row in df.itertuples(index=False, name=None)]
        return pd.DataFrame([o.to_dict() for o in orders])

    def merge_datasets(self):
        merged = self.orders_df.merge(self.products_df, on="product_id", how="left")
        merged = merged.merge(self.customers_df, on="customer_id", how="left")
        merged["discount_factor"] = 1 - (merged["discount_pct"] / 100)
        merged["line_total"] = merged["quantity"] * merged["price"] * merged["discount_factor"]
        merged["product_name"] = merged["product_id"].map(self.product_lookup)
        merged["customer_name"] = merged["customer_id"].map(self.customer_lookup)
        return merged

    def calculate_revenue(self):
        prices = self.merged_df["price"].to_numpy()
        quantities = self.merged_df["quantity"].to_numpy()
        discounts = self.merged_df["discount_pct"].to_numpy() / 100
        revenue = np.sum(prices * quantities * (1 - discounts))
        return round(float(revenue), 2)

    def calculate_profit(self):
        profit_margin = 0.2
        return round(self.calculate_revenue() * profit_margin, 2)

    def get_total_orders(self):
        return int(self.orders_df["order_id"].nunique())

    def get_total_customers(self):
        return int(self.customers_df["customer_id"].nunique())

    def get_average_order_value(self):
        return round(self.calculate_revenue() / self.get_total_orders(), 2)

    def get_best_selling_product(self):
        product_sales = self.get_product_sales()
        top_row = product_sales.iloc[0]
        return top_row["product_name"], float(top_row["line_total"])

    def get_dashboard_metrics(self):
        monthly_sales = self.get_monthly_sales()
        latest_month_revenue = monthly_sales["revenue"].iloc[-1] if not monthly_sales.empty else 0
        best_product_name, _ = self.get_best_selling_product()
        return {
            "total_revenue": self.calculate_revenue(),
            "total_orders": self.get_total_orders(),
            "total_customers": self.get_total_customers(),
            "average_order_value": self.get_average_order_value(),
            "best_selling_product": best_product_name,
            "monthly_revenue": latest_month_revenue,
            "profit": self.calculate_profit(),
        }

    def get_product_sales(self):
        product_sales = (
            self.merged_df.groupby("product_id")[["quantity", "line_total"]]
            .sum()
            .sort_values("line_total", ascending=False)
            .reset_index()
        )
        product_sales["product_name"] = product_sales["product_id"].map(self.product_lookup)
        return product_sales

    def get_customer_spending(self):
        customer_spending = (
            self.merged_df.groupby("customer_id")[["line_total"]]
            .sum()
            .sort_values("line_total", ascending=False)
            .reset_index()
        )
        customer_spending["customer_name"] = customer_spending["customer_id"].map(self.customer_lookup)
        return customer_spending

    def get_monthly_sales(self):
        self.merged_df["order_date"] = pd.to_datetime(self.merged_df["order_date"])
        monthly = (
            self.merged_df.groupby(self.merged_df["order_date"].dt.to_period("M"))["line_total"]
            .sum()
            .reset_index()
        )
        monthly.columns = ["month", "revenue"]
        monthly["month"] = monthly["month"].astype(str)
        return monthly

    def segment_customers(self):
        spending = self.get_customer_spending()
        spending["category"] = np.select(
            [spending["line_total"] > 5000, spending["line_total"] >= 3000, spending["line_total"] >= 1000, spending["line_total"] < 1000],
            ["Platinum", "Gold", "Silver", "Bronze"],
            default="Bronze",
        )
        return spending[["customer_id", "customer_name", "line_total", "category"]]

    def get_popular_products(self, top_n=3):
        popular = self.get_product_sales().head(top_n)
        return [row["product_name"] for _, row in popular.iterrows()]

    def get_co_purchase_pairs(self):
        order_items = (
            self.merged_df.groupby("order_id")["product_id"]
            .apply(lambda values: list(dict.fromkeys(values)))
            .reset_index(name="products")
        )
        pairs = []
        for products in order_items["products"]:
            for left_index in range(len(products)):
                for right_index in range(left_index + 1, len(products)):
                    pairs.append((products[left_index], products[right_index]))
        pair_df = pd.DataFrame(pairs, columns=["product_a", "product_b"])
        if pair_df.empty:
            return pd.DataFrame(columns=["product_a", "product_b", "pair_count"])
        return (
            pair_df.groupby(["product_a", "product_b"])
            .size()
            .reset_index(name="pair_count")
            .sort_values("pair_count", ascending=False)
        )

    def recommend_products(self, customer_id, top_n=3):
        purchased_products = set(self.merged_df[self.merged_df["customer_id"] == customer_id]["product_id"])
        pair_df = self.get_co_purchase_pairs()

        related_candidates = []
        for product_id in purchased_products:
            product_recommendations = self.recommendation_map.get(product_id, [])
            related_candidates.extend(product_recommendations)

        for _, row in pair_df[pair_df["product_a"].isin(purchased_products)].iterrows():
            if row["product_b"] not in purchased_products:
                related_candidates.append(row["product_b"])

        popular_candidates = [product_id for product_id in self.get_popular_products(top_n * 5) if product_id not in purchased_products]
        recommendation_ids = list(dict.fromkeys(related_candidates + popular_candidates))
        recommended_names = [self.product_lookup.get(product_id, product_id) for product_id in recommendation_ids[:top_n]]
        return recommended_names

    def create_visualizations(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        plt.style.use("seaborn-v0_8")
        sns.set_theme(style="whitegrid")

        revenue_series = self.get_monthly_sales()
        plt.figure(figsize=(8, 4))
        plt.plot(revenue_series["month"], revenue_series["revenue"], marker="o")
        plt.title("Revenue Trend")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "revenue_trend.png"))
        plt.close()

        sales_by_product = self.get_product_sales()
        plt.figure(figsize=(8, 4))
        plt.bar(sales_by_product["product_id"], sales_by_product["line_total"])
        plt.title("Best Selling Products")
        plt.xlabel("Product ID")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "best_selling_products.png"))
        plt.close()

        spending = self.get_customer_spending()
        plt.figure(figsize=(8, 4))
        plt.bar(spending["customer_id"], spending["line_total"])
        plt.title("Customer Spending")
        plt.xlabel("Customer ID")
        plt.ylabel("Spending")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "customer_spending.png"))
        plt.close()

        corr_df = self.merged_df[["quantity", "price", "discount_pct", "line_total"]]
        corr = corr_df.corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

    def create_dashboard(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        metrics = self.get_dashboard_metrics()
        monthly_sales = self.get_monthly_sales()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("E-Commerce Dashboard", fontsize=14, fontweight="bold")

        text = (
            f"Total Revenue: ${metrics['total_revenue']:,.2f}\n"
            f"Total Orders: {metrics['total_orders']}\n"
            f"Total Customers: {metrics['total_customers']}\n"
            f"Average Order Value: ${metrics['average_order_value']:,.2f}\n"
            f"Best Selling Product: {metrics['best_selling_product']}\n"
            f"Monthly Revenue: ${metrics['monthly_revenue']:,.2f}\n"
            f"Estimated Profit: ${metrics['profit']:,.2f}"
        )

        axes[0].axis("off")
        axes[0].text(0.05, 0.5, text, fontsize=11, va="center", family="monospace")

        axes[1].plot(monthly_sales["month"], monthly_sales["revenue"], marker="o", color="#4C78A8")
        axes[1].set_title("Monthly Revenue")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Revenue")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(output_dir, "dashboard.png"))
        plt.close(fig)

    def display_summary(self):
        print("E-Commerce Sales Analysis Summary")
        print("-" * 35)
        print(f"Total Revenue: ${self.calculate_revenue():,.2f}")
        print(f"Total Orders: {self.get_total_orders()}")
        print(f"Total Customers: {self.get_total_customers()}")
        print(f"Average Order Value: ${self.get_average_order_value():,.2f}")
        best_product_name, _ = self.get_best_selling_product()
        print(f"Best Selling Product: {best_product_name}")
        print(f"Estimated Profit: ${self.calculate_profit():,.2f}")
        print("\nProduct-wise Sales:")
        print(self.get_product_sales().to_string(index=False))
        print("\nCustomer-wise Spending:")
        print(self.get_customer_spending().to_string(index=False))
        print("\nCustomer Segmentation:")
        print(self.segment_customers().to_string(index=False))
        print("\nMonthly Sales:")
        print(self.get_monthly_sales().to_string(index=False))
        print("\nExample Product Recommendations:")
        customer_id = self.customers_df.iloc[0]["customer_id"]
        print(f"Customer {customer_id}: {self.recommend_products(customer_id)}")


if __name__ == "__main__":
    analytics = ECommerceAnalytics("data")
    analytics.display_summary()
    analytics.create_visualizations("output/plots")
    analytics.create_dashboard("output/plots")
    print("\nCharts and dashboard saved to output/plots")
