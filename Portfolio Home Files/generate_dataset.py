import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
num_customers = 500

data = {
    "Customer_ID": [f"CUST_{i}" for i in range(1, num_customers + 1)],
    "Customer_Segment": np.random.choice(["Enterprise", "SMB", "Startup"], num_customers, p=[0.5, 0.3, 0.2]),
    "Monthly_Cloud_Spend": np.round(np.random.uniform(1000, 50000, num_customers), 2),
    "Service_Usage_EC2": np.round(np.random.uniform(10, 500, num_customers), 1),
    "Service_Usage_S3": np.round(np.random.uniform(5, 300, num_customers), 1),
    "Service_Usage_RDS": np.round(np.random.uniform(1, 100, num_customers), 1),
    "Contract_Term_Length": np.random.choice([12, 24, 36, 48], num_customers, p=[0.4, 0.3, 0.2, 0.1]),
    "Discount_Rate": np.round(np.random.uniform(5, 30, num_customers), 1),
    "Customer_Satisfaction_Score": np.random.randint(1, 6, num_customers),
    "Renewal_Likelihood": np.round(np.random.uniform(0.5, 1, num_customers), 2),
}

# Create DataFrame
synthetic_data = pd.DataFrame(data)

# Save to CSV
synthetic_data.to_csv("synthetic_cloud_pricing_dataset.csv", index=False)
print("Dataset saved as 'synthetic_cloud_pricing_dataset.csv'")
