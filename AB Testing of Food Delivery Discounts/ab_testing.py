# A/B Testing of Food Delivery Discounts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------------------------------
# 1. CREATE SIMULATED DATASET
# --------------------------------------------------

np.random.seed(10)

# Group A: 10% discount
group_A = pd.DataFrame({
    "Customer_ID": range(1, 51),
    "Group": "A",
    "Discount": "10%",
    "Ordered": [1] * 20 + [0] * 30
})

# Group B: 20% discount
group_B = pd.DataFrame({
    "Customer_ID": range(51, 101),
    "Group": "B",
    "Discount": "20%",
    "Ordered": [1] * 30 + [0] * 20
})

# Combine both groups
data = pd.concat([group_A, group_B], ignore_index=True)

# Shuffle customers
data = data.sample(frac=1, random_state=10).reset_index(drop=True)

print("FIRST 10 RECORDS:")
print(data.head(10))

print("\nTotal Customers:", len(data))


# --------------------------------------------------
# 2. CALCULATE RESPONSE / ORDER RATES
# --------------------------------------------------

group_A_orders = data[data["Group"] == "A"]["Ordered"].sum()
group_B_orders = data[data["Group"] == "B"]["Ordered"].sum()

group_A_total = len(data[data["Group"] == "A"])
group_B_total = len(data[data["Group"] == "B"])

rate_A = group_A_orders / group_A_total
rate_B = group_B_orders / group_B_total

print("\nORDER RESULTS")
print("----------------------")

print("10% Discount Orders:", group_A_orders)
print("10% Discount Order Rate:", rate_A * 100, "%")

print("\n20% Discount Orders:", group_B_orders)
print("20% Discount Order Rate:", rate_B * 100, "%")


# --------------------------------------------------
# 3. HYPOTHESES
# --------------------------------------------------

print("\nHYPOTHESES")
print("----------------------")

print("H0: There is no significant difference in order")
print("    rates between the two discount groups.")

print("\nH1: There is a significant difference in order")
print("    rates between the two discount groups.")


# --------------------------------------------------
# 4. TWO-PROPORTION Z-TEST
# --------------------------------------------------

# Number of successes
x1 = group_A_orders
x2 = group_B_orders

# Number of observations
n1 = group_A_total
n2 = group_B_total

# Combined proportion
pooled_rate = (x1 + x2) / (n1 + n2)

# Standard error
standard_error = np.sqrt(
    pooled_rate * (1 - pooled_rate) * (1/n1 + 1/n2)
)

# Z statistic
z_value = (rate_B - rate_A) / standard_error

# Two-tailed p-value
p_value = 2 * (1 - norm.cdf(abs(z_value)))

print("\nSTATISTICAL TEST")
print("----------------------")

print("Z-value:", round(z_value, 4))
print("P-value:", round(p_value, 4))


# --------------------------------------------------
# 5. DECISION
# --------------------------------------------------

alpha = 0.05

print("\nDECISION")
print("----------------------")

if p_value < alpha:
    print("P-value < 0.05")
    print("Reject H0.")
    print("There is a statistically significant difference")
    print("between the two discount groups.")
else:
    print("P-value >= 0.05")
    print("Fail to reject H0.")
    print("There is no statistically significant difference")
    print("between the two discount groups.")


# --------------------------------------------------
# 6. GRAPH 1 - ORDER RATE COMPARISON
# --------------------------------------------------

discounts = ["10% Discount", "20% Discount"]
order_rates = [rate_A * 100, rate_B * 100]

plt.figure(figsize=(7, 5))

plt.bar(discounts, order_rates)

plt.title("Order Rate Comparison")
plt.xlabel("Discount Offered")
plt.ylabel("Order Rate (%)")

plt.ylim(0, 100)

for i, value in enumerate(order_rates):
    plt.text(i, value + 2, f"{value:.1f}%", ha="center")

plt.show()


# --------------------------------------------------
# 7. GRAPH 2 - ORDERS VS NON-ORDERS
# --------------------------------------------------

orders = [group_A_orders, group_B_orders]

non_orders = [
    group_A_total - group_A_orders,
    group_B_total - group_B_orders
]

x = np.arange(2)
width = 0.35

plt.figure(figsize=(7, 5))

plt.bar(x - width/2, orders, width, label="Ordered")
plt.bar(x + width/2, non_orders, width, label="Not Ordered")

plt.xticks(x, discounts)
plt.xlabel("Discount Offered")
plt.ylabel("Number of Customers")

plt.title("Orders vs Non-Orders")
plt.legend()

plt.show()


# --------------------------------------------------
# 8. FINAL RESULT
# --------------------------------------------------

print("\nFINAL RESULT")
print("----------------------")

print("10% Discount Order Rate:", rate_A * 100, "%")
print("20% Discount Order Rate:", rate_B * 100, "%")
print("P-value:", round(p_value, 4))

if p_value < alpha:
    print("\nConclusion:")
    print("The difference in order rates is statistically significant.")
else:
    print("\nConclusion:")
    print("The difference in order rates is not statistically significant.")