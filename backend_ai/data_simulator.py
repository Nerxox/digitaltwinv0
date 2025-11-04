import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_simulated_data(output_path: str = '../datasets/simulated_data.csv'):
    """Generates synthetic time-series data for the digital twin PoC."""
    logger.info("Starting data simulation...")
    
    # --- Configuration ---
    start_date = '2024-01-01'
    end_date = '2025-01-01'
    freq = 'h' # Hourly frequency
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)[:-1] # Exclude the end date
    n_samples = len(date_range)
    
    # --- Data Generation ---
    
    # 1. Base Load (Diurnal Pattern)
    hour_of_day = date_range.hour
    base_load = 50 + 40 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
    
    # 2. Machine 1 (M001) - High power, runs during work hours (8am-6pm)
    m1_power = np.where((hour_of_day >= 8) & (hour_of_day <= 18), 
                        150 + 50 * np.random.rand(n_samples), 
                        5 + 5 * np.random.rand(n_samples))
    
    # 3. Machine 2 (M002) - Medium power, runs 24/7 with a dip at night
    m2_power = 80 + 30 * np.random.rand(n_samples) + 20 * np.sin(2 * np.pi * (hour_of_day - 12) / 24)
    m2_power = np.clip(m2_power, 10, 150)
    
    # 4. HVAC/Ambient Temperature (Seasonal/Diurnal)
    day_of_year = date_range.dayofyear
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    diurnal_temp = 3 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    ambient_temp = seasonal_temp + diurnal_temp + 2 * np.random.randn(n_samples)
    
    # 5. Total Power (kW)
    total_power_kw = base_load + m1_power + m2_power + (ambient_temp * 5) # Temp affects power
    
    # 6. Production Schedule (Binary/Categorical)
    # Assume production is active 5 days a week, 2 shifts (8am-10pm)
    day_of_week = date_range.dayofweek # Monday=0, Sunday=6
    is_production = np.where(
        (day_of_week < 5) & (hour_of_day >= 8) & (hour_of_day <= 22), 
        1, 
        0
    )
    
    # --- DataFrame Creation ---
    simulated_df = pd.DataFrame({
        'timestamp': date_range,
        'ambient_temp_c': ambient_temp,
        'm001_power_kw': m1_power,
        'm002_power_kw': m2_power,
        'total_power_kw': total_power_kw,
        'is_production': is_production
    })
    
    # --- Save Data ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    simulated_df.to_csv(output_path, index=False)
    logger.info(f"Data simulation complete. Saved {n_samples} samples to {output_path}")

if __name__ == "__main__":
    # Change directory to the project root to correctly resolve paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..") # Move up to backend_ai
    os.chdir("..") # Move up to digitaltwinv0/digitaltwinv0
    
    # The data needs to be in the datasets folder, not the backend_ai folder
    generate_simulated_data(output_path='datasets/simulated_data.csv')
