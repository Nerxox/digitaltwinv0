import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_simulated_data(start_date, end_date, freq='H'):
    """
    Generates synthetic time-series data for the digital twin PoC.

    The data simulates:
    1. Power consumption of two machines (M1, M2) and HVAC.
    2. Total power consumption (sum of all).
    3. Ambient temperature (influencing HVAC).
    4. Production schedule (binary, 0=off, 1=on).
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_samples = len(date_range)
    
    # 1. Ambient Temperature (Simulate daily and seasonal cycles)
    # Use a sine wave for daily cycle and a long-term trend for seasonal
    day_cycle = np.sin(np.linspace(0, 4 * np.pi * n_samples / (24 * (365/12)), n_samples)) * 5
    base_temp = 20 + np.sin(np.linspace(0, 2 * np.pi * n_samples / (24 * 365), n_samples)) * 3
    ambient_temp = (base_temp + day_cycle + np.random.normal(0, 0.5, n_samples)).round(2)

    # 2. Production Schedule (Simulate an 8-hour workday, 5 days a week)
    production_schedule = np.zeros(n_samples)
    for i, date in enumerate(date_range):
        # Workdays (Monday=0 to Friday=4)
        if 0 <= date.dayofweek <= 4:
            # Working hours (8am to 4pm)
            if 8 <= date.hour < 16:
                production_schedule[i] = 1

    # 3. Machine Power Consumption
    # Base power for machines (M1: 10kW, M2: 8kW)
    base_m1_power = 10
    base_m2_power = 8
    
    # M1 and M2 consume power only during production hours
    machine_1_power = (base_m1_power + np.random.normal(0, 0.5, n_samples)) * production_schedule
    machine_2_power = (base_m2_power + np.random.normal(0, 0.4, n_samples)) * production_schedule
    
    # 4. HVAC Power Consumption (Influenced by ambient temperature)
    # HVAC turns on if temp > 22 or temp < 18 (for heating/cooling)
    hvac_on = ((ambient_temp > 22) | (ambient_temp < 18)).astype(int)
    # HVAC base power (5kW) + a factor of how far the temp is from 20
    hvac_power = (5 + np.abs(ambient_temp - 20) * 0.5 + np.random.normal(0, 0.3, n_samples)) * hvac_on
    hvac_power[hvac_power < 0] = 0 # Ensure power is not negative

    # 5. Total Power Consumption
    total_power = machine_1_power + machine_2_power + hvac_power + 2 # Add 2kW for base lighting/IT

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': date_range,
        'ambient_temp': ambient_temp,
        'production_schedule': production_schedule,
        'machine_1_power': machine_1_power.round(2),
        'machine_2_power': machine_2_power.round(2),
        'hvac_power': hvac_power.round(2),
        'total_power': total_power.round(2)
    })
    
    data = data.set_index('timestamp')
    return data

if __name__ == '__main__':
    # Generate 1 year of hourly data for the PoC
    start = datetime(2024, 1, 1)
    end = datetime(2025, 1, 1)
    
    print(f"Generating data from {start.date()} to {end.date()}...")
    simulated_df = generate_simulated_data(start, end)
    
    import os
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, '..', 'data', 'simulated_data.csv')
    simulated_df.to_csv(output_path)
    
    print(f"Data generation complete. Saved {len(simulated_df)} samples to {output_path}")
    print("\nFirst 5 rows of generated data:")
    print(simulated_df.head())
    print("\nData types:")
    print(simulated_df.dtypes)
