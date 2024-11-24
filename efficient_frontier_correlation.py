import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# Define a function to calculate portfolio statistics
def calculate_portfolio(mu1, mu2, sigma1, sigma2, rho, weights):
    """Calculate portfolio mean and standard deviation."""
    w1, w2 = weights
    portfolio_return = w1 * mu1 + w2 * mu2
    portfolio_variance = (
        (w1 ** 2) * (sigma1 ** 2) +
        (w2 ** 2) * (sigma2 ** 2) +
        (2 * w1 * w2 * sigma1 * sigma2 * rho)
    )
    portfolio_std = np.sqrt(portfolio_variance)
    return portfolio_return, portfolio_std


# Define asset statistics
mu1 = 0.15  # Expected return of Asset 1
mu2 = 0.10  # Expected return of Asset 2
sigma1 = 0.20  # Standard deviation of Asset 1
sigma2 = 0.15  # Standard deviation of Asset 2

# Streamlit app
st.title("Efficient Frontier with Correlation Adjustment")

# Display the variance formula using LaTeX
latext = r'''
### Portfolio Variance Formula:
$$
\sigma_p^2 = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 \rho_{12} \sigma_1 \sigma_2
$$
- $w_1, w_2$: Weights of the two assets  
- $\sigma_1, \sigma_2$: Standard deviations of the two assets  
- $\rho_{12}$: Correlation between the two assets
'''
st.markdown(latext, unsafe_allow_html=True)

# User inputs
rho = st.slider(
    "Correlation between Asset 1 and Asset 2", -1.0, 1.0, 0.0, step=0.1)

# Generate portfolio weights
weights = np.linspace(0, 1, 100)
returns = []
std_devs = []

# Calculate portfolio stats for the user-selected correlation
for w1 in weights:
    w2 = 1 - w1
    ret, std = calculate_portfolio(mu1, mu2, sigma1, sigma2, rho, (w1, w2))
    returns.append(ret)
    std_devs.append(std)

# Calculate portfolio stats for ρ12 = -1 with optimized weights
w1_optimal = sigma2 / (sigma1 + sigma2)
w2_optimal = sigma1 / (sigma1 + sigma2)
ret_neg1_opt, std_neg1_opt = calculate_portfolio(
    mu1, mu2, sigma1, sigma2, -1, (w1_optimal, w2_optimal)
)

# Calculate portfolio stats for ρ12 = -1 and ρ12 = 1 for the full weight range
returns_neg1, std_devs_neg1 = [], []
returns_pos1, std_devs_pos1 = [], []
for w1 in weights:
    w2 = 1 - w1
    ret_neg1, std_neg1 = calculate_portfolio(
        mu1, mu2, sigma1, sigma2, -1, (w1, w2))
    ret_pos1, std_pos1 = calculate_portfolio(
        mu1, mu2, sigma1, sigma2, 1, (w1, w2))
    returns_neg1.append(ret_neg1)
    std_devs_neg1.append(std_neg1)
    returns_pos1.append(ret_pos1)
    std_devs_pos1.append(std_pos1)

### Plotting ###

# Plot efficient frontier
fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size for clarity
ax.plot(std_devs, returns,
        label=f"Efficient Frontier (ρ12 = {rho:.1f})", color="blue")
ax.plot(std_devs_neg1, returns_neg1, '--',
        label="Efficient Frontier (ρ12 = -1)", color="m")
ax.plot(std_devs_pos1, returns_pos1, '--',
        label="Efficient Frontier (ρ12 = 1)", color="maroon")
ax.scatter([sigma1, sigma2], [mu1, mu2],
           color="k", marker="o", s=60, label="Individual Assets")

# Highlight zero-risk portfolio for ρ12 = -1
ax.scatter([std_neg1_opt], [ret_neg1_opt], color="green",
           label="Zero Risk Portfolio (ρ12 = -1)", zorder=5)

# Add labels near the individual assets
ax.text(sigma1 + 0.005, mu1, "Asset 1", color="k", fontsize=11, ha="left")
ax.text(sigma2 + 0.005, mu2, "Asset 2", color="k", fontsize=11, ha="left")

# Add labels near the lines
# Move ρ12 = -1 label slightly downward by subtracting 0.01 from the y-coordinate
ax.text(std_devs_neg1[len(std_devs_neg1)//2], returns_neg1[len(returns_neg1)//2] - 0.01/4,
        "ρ12 = -1", color="m", fontsize=11, ha="left")

# Move ρ12 = 1 label slightly to the right by adding 0.01 to the x-coordinate
ax.text(std_devs_pos1[len(std_devs_pos1)//2] + 0.01/2, returns_pos1[len(returns_pos1)//2],
        "ρ12 = 1", color="maroon", fontsize=11, ha="left")

# Set plot limits
ax.set_xlim(0, max(std_devs) + 0.04)  # Extend X-axis limit
ax.set_ylim(0.095, max(returns) + 0.01)  # Extend Y-axis limit

# Change axis title font sizes
ax.set_title("Efficient Frontier", fontsize=16)  # Larger title font
ax.set_xlabel("Portfolio Risk (σp)", fontsize=14)  # Larger X-axis label font
# Larger Y-axis label font
ax.set_ylabel("Portfolio Return (E[rp])", fontsize=14)

# Add legend and grid
ax.legend(loc="upper left")  # Legend inside the plot
ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)


# Show portfolio statistics for extreme cases
min_risk_idx = np.argmin(std_devs)
st.write("### Portfolio Statistics")
st.write(f"Minimum Risk Portfolio: Risk = {
         std_devs[min_risk_idx]:.2%}, Return = {returns[min_risk_idx]:.2%}")
st.write(f"Maximum Return Portfolio: Risk = {
         std_devs[-1]:.2%}, Return = {returns[-1]:.2%}")
st.write(f"Zero Risk Portfolio (ρ12 = -1): Risk = {
         std_neg1_opt:.2%}, Return = {ret_neg1_opt:.2%}")
