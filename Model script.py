import pandas as pd
import numpy as np
import sys
seasonal_df = pd.read_excel('YE_1990_2000.xlsx',index_col=0) # Input file name

seasonal_df['Inflow'] = seasonal_df['Diversion'] + seasonal_df['Catchment_inflow']  # Total inflow during the season
seasonal_df['Inflow_check'] = seasonal_df['Diversion_check'] + seasonal_df['Catchment_inflow_check'] # Farmers check inflow during the observation period

# Calculate crop water requirement
initial_days = 20
development_days = 30
middle_days = 30
late_days = 25

initial_kc = 1.0
development_kc = 1.15
middle_kc = 1.2
late_kc = 0.9

initial_cwr = seasonal_df['Ref_evapotranspiration_avg'] * initial_kc * initial_days / 30
development_cwr = seasonal_df['Ref_evapotranspiration_avg'] * development_kc * development_days / 30
middle_cwr = seasonal_df['Ref_evapotranspiration_avg'] * middle_kc * middle_days / 30
late_cwr = seasonal_df['Ref_evapotranspiration_avg'] * late_kc * late_days / 30

total_cwr = initial_cwr + development_cwr + middle_cwr + late_cwr
seasonal_df['Total_water_recquiremnt'] = total_cwr  # mm

#Reservoir water balance model
def reservoir_simulation(df, application_efficiency_wet, application_efficiency_dry, initial_storage, storage_capacity, hedging_slope_wet, hedging_slope_dry,loss_factor_wet, loss_factor_dry,
                         threshold_storage_wet, threshold_storage_dry, land_preparation_wet,land_preparation_dry, eflow_river_factor, eflow_canal_factor_wet, eflow_canal_factor_dry,
                         threshold_precipitation_wet, threshold_precipitation_dry, curve_steepness_storage_wet, curve_steepness_storage_dry,
                         curve_steepness_precipitation_wet, curve_steepness_precipitation_dry, pond_surface_area, pond_storage_factor_wet, pond_storage_factor_dry,
                         max_pond_storage, normalised_distance):

    inflow = df['Inflow'].to_numpy()
    inflow_check = df['Inflow_check'].to_numpy()
    allocation = df['Allocation'].to_numpy()
    precipitation_check = df['Precipitation_check'].to_numpy() # Mar-July and Sep-Jan cumulative rainfall is observed. Data is arranged in the input file
    precipitation_seasonal = df['Precipitation_seasonal'].to_numpy()
    total_cwr = df['Total_water_recquiremnt'].to_numpy()
    season = df['Season'].to_numpy()  # Check for the season (wet or dry)
    spillage = df['Spillage'].to_numpy()

    N = len(inflow)

    storage = np.zeros(N)
    eflow_river = np.zeros(N)
    eflow_canal = np.zeros(N)
    withdrawal_irrigation = np.zeros(N)
    withdrawal_env = np.zeros(N)
    withdrawal = np.zeros(N)
    spill = np.zeros(N)
    deficit = np.zeros(N)
    available_water = np.zeros(N)
    effective_precipitation = np.zeros(N)
    total_wr = np.zeros(N)
    salience_storage = np.zeros(N)
    salience_precipitation = np.zeros(N)
    salience = np.zeros(N)
    cultivation_area = np.zeros(N)
    flexibility = np.zeros(N)
    irrigation_water_requirement = np.zeros(N)
    irrigation_water_demand = np.zeros(N)
    virtual_storage = np.zeros(N)
    combined_effect = np.zeros(N)
    pond_storage = np.zeros(N)
    available_pond_water = np.zeros(N)

    storage[0] = initial_storage

    for i in range(N):

        # Calculate cumulative reservoir water availability during the observation window
        virtual_storage[i] = storage[i] + inflow_check[i]

        # Assign parameters based on season
        if season[i] == 'Wet':
            land_preparation = land_preparation_wet
            loss_factor = loss_factor_wet
            eflow_canal_factor = eflow_canal_factor_wet
            threshold_storage = threshold_storage_wet #perceived threshold reservoir storage
            threshold_precipitation = threshold_precipitation_wet #perceived threshold rainfall
            curve_steepness_storage = curve_steepness_storage_wet #responsiveness to reservoir storage
            curve_steepness_precipitation = curve_steepness_precipitation_wet #responsiveness to rainfall
            application_efficiency = application_efficiency_wet
            hedging_slope = hedging_slope_wet
            pond_storage_factor = pond_storage_factor_wet
        else:
            land_preparation = land_preparation_dry
            loss_factor = loss_factor_dry
            eflow_canal_factor = eflow_canal_factor_dry
            threshold_storage = threshold_storage_dry
            threshold_precipitation = threshold_precipitation_dry
            curve_steepness_storage = curve_steepness_storage_dry
            curve_steepness_precipitation = curve_steepness_precipitation_dry
            application_efficiency = application_efficiency_dry
            hedging_slope = hedging_slope_dry
            pond_storage_factor = pond_storage_factor_dry


        # salience based on seasonal storage
        salience_storage[i] = 1 / (1 + np.exp(-curve_steepness_storage * (virtual_storage[i] - threshold_storage)))

        # salience based on rainfall
        salience_precipitation[i] = 1 / (1 + np.exp(-curve_steepness_precipitation * (precipitation_check[i] - threshold_precipitation)))

        # Combined salience
        salience[i] = salience_storage[i] * salience_precipitation[i]

        # Calculate pond storage
        pond_storage[i] = precipitation_seasonal[i] * pond_surface_area * pond_storage_factor * 0.001 * 1e-6 # mcm

        #Calculate flexibility generated by ponds
        flexibility[i] = np.clip(pond_storage[i] / max_pond_storage, 0, 1) * (1-normalised_distance)

        #Combined salience + flexibility effect
        combined_effect[i] = salience[i] + (1- salience[i]) * flexibility[i]

        # Calculate cultivation area
        command_area = 4721 # ha
        cultivation_area[i] = combined_effect[i] * command_area

        # Estimate water from ponds to satisfy CWR
        available_pond_water[i] = pond_storage[i] # mcm

        total_wr[i] = cultivation_area[i] * 10000 * 1e-9 *(total_cwr[i] + land_preparation) / application_efficiency
        effective_precipitation[i] = max(0, (precipitation_seasonal[i] - 150) * 0.67)

        irrigation_water_requirement[i] = max(0, total_wr[i] - (effective_precipitation[i] * cultivation_area[i] * 10000 * 0.001 * 1e-6 + available_pond_water[i]))

        irrigation_water_demand[i] = irrigation_water_requirement[i] * loss_factor # Account for conveynace loss

        eflow_river[i] = inflow[i] * eflow_river_factor #This volume is now accounted in the net spillage

        available_water[i] = storage[i] + inflow[i] - eflow_river[i] - allocation[i] - spillage[i]

        if available_water[i] < irrigation_water_demand[i] * hedging_slope:
            withdrawal_irrigation[i] = available_water[i] / hedging_slope
            withdrawal_env[i] = 0
        else:
            withdrawal_irrigation[i] = irrigation_water_demand[i]
            remaining_water = available_water[i] - irrigation_water_demand[i]
            eflow_canal[i] = remaining_water * eflow_canal_factor
            withdrawal_env[i] = eflow_canal[i]

        withdrawal[i] = withdrawal_irrigation[i] + withdrawal_env[i]
        trial_storage = available_water[i] - withdrawal[i]

        if trial_storage > storage_capacity:
            spill[i] = trial_storage - storage_capacity
            if i < N-1:
                storage[i+1] = storage_capacity
        else:
            spill[i] = 0
            if i < N-1:
                storage[i+1] = trial_storage

        deficit[i] = irrigation_water_demand[i] - withdrawal_irrigation[i]

    df['Storage'] = storage
    df['Available_water'] = available_water
    df['Cultivation_area'] = cultivation_area
    df['Withdrawal'] = withdrawal
    df['Spill'] = spill
    df['Deficit'] = deficit
    df['Withdrawal_irrigation'] = withdrawal_irrigation
    df['Irrigation_demand'] = irrigation_water_demand
    df['Withdrawal_env'] = withdrawal_env
    df['Flexibility'] = flexibility
    df['Salience'] = salience
    df['Combined_effect'] = combined_effect
    df['Virtual_storage'] = virtual_storage
    df['Pond_storage'] = pond_storage
    df['Available_pond_water'] = available_pond_water
    df['total_wr'] = total_wr
    df['effective_precipitation'] = effective_precipitation
    df['irrigation_water_requirement'] = irrigation_water_requirement
    df['irrigation_water_demand'] = irrigation_water_demand

    return df

from google.colab import drive
drive.mount('/content/drive')

#Model paramterization
import sys
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt

# Define the fixed parameters
fixed_params = {
    'initial_storage': 115.9, #Change according to simulation period P1-P3
    'storage_capacity': 123.7,
    'eflow_river_factor': 0,
    #'hedging_slope_wet': 3,
    #'hedging_slope_dry': 3,
    'land_preparation_wet': 168,
    'land_preparation_dry': 275,
    'application_efficiency_wet': 0.55,
    'application_efficiency_dry': 0.55,
    #'eflow_canal_factor_wet': 0.06,
    #'eflow_canal_factor_dry': 0.0,
    'loss_factor_wet': 1.3,
    'loss_factor_dry': 1.3,
    'pond_surface_area': 10.64e6,  # Fixed pond surface area in m²
    'max_pond_storage': 6.0,  # Fixed max pond storage
    'normalised_distance': 0.2,  # Fixed value for normalised_distance
    #'pond_storage_factor_wet': 0.38,
    #'pond_storage_factor_dry': 0.38,
}

# Define the parameter names and bounds for the parameters to be optimized
optimized_params_info = [
    ('threshold_storage_wet', (100, 500)),
    ('threshold_storage_dry', (100, 500)),
    ('threshold_precipitation_wet', (100, 500)),
    ('threshold_precipitation_dry', (100, 500)),
    ('curve_steepness_storage_wet', (0.01, 1)),
    ('curve_steepness_storage_dry', (0.01, 1)),
    ('curve_steepness_precipitation_wet', (0.01, 1)),
    ('curve_steepness_precipitation_dry', (0.01, 1)),
    ('hedging_slope_wet', (1, 4)),
    ('hedging_slope_dry', (1, 4)),
    ('eflow_canal_factor_wet', (0, 1)),
    ('eflow_canal_factor_dry', (0, 1)),
    ('pond_storage_factor_wet', (0, 1)),
    ('pond_storage_factor_dry', (0, 1))
]

# Lists to store best parameter sets and scores for each seed
best_parameter_sets = []  # List to store best parameter sets
best_scores = []  # List to store corresponding scores
all_seeds = []  # List to store the seed used for each run
nse_cultivation_area_list = []  # List to store NSE for cultivation area
nse_withdrawal_list = []  # List to store NSE for withdrawal
nse_storage_list = []  # List to store NSE for storage

# Define the objective function for Simulated Annealing
def objective_function_sa(opt_params):
    # Combine fixed parameters with optimized parameters
    params = {
        'land_preparation_wet': fixed_params['land_preparation_wet'],
        'land_preparation_dry': fixed_params['land_preparation_dry'],
        'initial_storage': fixed_params['initial_storage'],  # Fixed parameter
        'storage_capacity': fixed_params['storage_capacity'],  # Fixed parameter
        'eflow_river_factor': fixed_params['eflow_river_factor'],  # Fixed parameter
        'application_efficiency_wet': fixed_params['application_efficiency_wet'],
        'application_efficiency_dry': fixed_params['application_efficiency_dry'],
        'loss_factor_wet': fixed_params['loss_factor_wet'],
        'loss_factor_dry': fixed_params['loss_factor_dry'],
        'pond_surface_area': fixed_params['pond_surface_area'],  # Fixed parameter
        'max_pond_storage': fixed_params['max_pond_storage'],  # Fixed parameter
        'normalised_distance': fixed_params['normalised_distance'],
        #'pond_storage_factor_wet': fixed_params['pond_storage_factor_wet'],
        #'pond_storage_factor_dry': fixed_params['pond_storage_factor_dry'],
        'threshold_storage_wet': opt_params[0],
        'threshold_storage_dry': opt_params[1],
        'threshold_precipitation_wet': opt_params[2],
        'threshold_precipitation_dry': opt_params[3],
        'curve_steepness_storage_wet': opt_params[4],
        'curve_steepness_storage_dry': opt_params[5],
        'curve_steepness_precipitation_wet': opt_params[6],
        'curve_steepness_precipitation_dry': opt_params[7],
        'hedging_slope_wet': opt_params[8],
        'hedging_slope_dry': opt_params[9],
        'eflow_canal_factor_wet': opt_params[10],
        'eflow_canal_factor_dry': opt_params[11],
        'pond_storage_factor_wet': opt_params[12],
        'pond_storage_factor_dry': opt_params[13]
    }

    simulated_df = reservoir_simulation(seasonal_df.copy(), **params)

    observed_cultivation_area = seasonal_df['Observed_cultivation_area'].to_numpy()
    simulated_cultivation_area = simulated_df['Cultivation_area'].to_numpy()

    observed_withdrawal = seasonal_df['Observed_demand'].to_numpy()
    simulated_withdrawal = simulated_df['Withdrawal'].to_numpy()

    observed_storage = seasonal_df['Observed_storage'].to_numpy()
    simulated_storage = simulated_df['Storage'].to_numpy()

    # Calculate the NSEs
    nse_cultivation_area = 1 - (np.sum((observed_cultivation_area - simulated_cultivation_area)**2) / np.sum((observed_cultivation_area - np.mean(observed_cultivation_area))**2))
    nse_withdrawal = 1 - (np.sum((observed_withdrawal - simulated_withdrawal)**2) / np.sum((observed_withdrawal - np.mean(observed_withdrawal))**2))
    nse_storage = 1 - (np.sum((observed_storage - simulated_storage)**2) / np.sum((observed_storage - np.mean(observed_storage))**2))

    # Combine the NSEs
    combined_nse = -(nse_cultivation_area + nse_withdrawal + nse_storage) / 3

    # Return the negative combined NSE to maximize it
    return combined_nse

# Extract the bounds for optimization
bounds = [bound for _, bound in optimized_params_info]

# Run optimization with multiple seeds
num_runs = 50  # Number of runs
for i in range(num_runs):
    seed = i * 5  # Different seed for each run
    np.random.seed(seed)
    result = dual_annealing(objective_function_sa, bounds, maxiter=1000)

    # Extract optimized parameters and the best score
    optimized_params = result.x
    # Combine fixed parameters with optimized parameters for the final simulation
    final_params = {
        'land_preparation_wet': fixed_params['land_preparation_wet'],
        'land_preparation_dry': fixed_params['land_preparation_dry'],
        'initial_storage': fixed_params['initial_storage'],  # Fixed parameter
        'storage_capacity': fixed_params['storage_capacity'],  # Fixed parameter
        'eflow_river_factor': fixed_params['eflow_river_factor'],  # Fixed parameter
        'application_efficiency_wet': fixed_params['application_efficiency_wet'],
        'application_efficiency_dry': fixed_params['application_efficiency_dry'],
        'loss_factor_wet': fixed_params['loss_factor_wet'],
        'loss_factor_dry': fixed_params['loss_factor_dry'],
        'pond_surface_area': fixed_params['pond_surface_area'],  # Fixed parameter
        'max_pond_storage': fixed_params['max_pond_storage'],  # Fixed parameter
        'normalised_distance': fixed_params['normalised_distance'],
        #'pond_storage_factor_wet': fixed_params['pond_storage_factor_wet'],
        #'pond_storage_factor_dry': fixed_params['pond_storage_factor_dry'],
        'threshold_storage_wet': optimized_params[0],
        'threshold_storage_dry': optimized_params[1],
        'threshold_precipitation_wet': optimized_params[2],
        'threshold_precipitation_dry': optimized_params[3],
        'curve_steepness_storage_wet': optimized_params[4],
        'curve_steepness_storage_dry': optimized_params[5],
        'curve_steepness_precipitation_wet': optimized_params[6],
        'curve_steepness_precipitation_dry': optimized_params[7],
        'hedging_slope_wet': optimized_params[8],
        'hedging_slope_dry': optimized_params[9],
        'eflow_canal_factor_wet': optimized_params[10],
        'eflow_canal_factor_dry': optimized_params[11],
        'pond_storage_factor_wet': optimized_params[12],
        'pond_storage_factor_dry': optimized_params[13]
    }

    # Run simulation with optimized parameters
    optimized_df = reservoir_simulation(seasonal_df.copy(), **final_params)

    # Calculate NSE for cultivation area
    observed_cultivation_area = seasonal_df['Observed_cultivation_area'].to_numpy()
    optimized_cultivation_area = optimized_df['Cultivation_area'].to_numpy()
    nse_cultivation_area = 1 - (np.sum((observed_cultivation_area - optimized_cultivation_area)**2) / np.sum((observed_cultivation_area - np.mean(observed_cultivation_area))**2))

    # Calculate NSE for observed demand vs. withdrawal
    observed_withdrawal = seasonal_df['Observed_demand'].to_numpy()
    optimized_withdrawal = optimized_df['Withdrawal'].to_numpy()
    nse_withdrawal = 1 - (np.sum((observed_withdrawal - optimized_withdrawal)**2) / np.sum((observed_withdrawal - np.mean(observed_withdrawal))**2))

    # Calculate NSE for observed and simulated storage
    observed_storage = seasonal_df['Observed_storage'].to_numpy()
    optimized_storage = optimized_df['Storage'].to_numpy()
    nse_storage = 1 - (np.sum((observed_storage - optimized_storage)**2) / np.sum((observed_storage - np.mean(observed_storage))**2))

    # Store results for this seed
    best_parameter_sets.append(result.x)  # Store best parameter set
    best_scores.append(-result.fun)  # Store the best score
    all_seeds.append(seed)  # Store the seed

    # Store NSE values
    nse_cultivation_area_list.append(nse_cultivation_area)
    nse_withdrawal_list.append(nse_withdrawal)
    nse_storage_list.append(nse_storage)

# Create DataFrame for best parameter sets
param_names = [name for name, _ in optimized_params_info]  # Get parameter names
results_df = pd.DataFrame(best_parameter_sets, columns=param_names)
results_df['Score'] = best_scores  # Add the scores column
results_df['Seed'] = all_seeds  # Add the seed column
results_df['NSE_Cultivation_Area'] = nse_cultivation_area_list  # Add NSE for cultivation area
results_df['NSE_Withdrawal'] = nse_withdrawal_list  # Add NSE for withdrawal
results_df['NSE_Storage'] = nse_storage_list  # Add NSE for storage

# Output the results DataFrame
print("Best Parameter Set and Score for Each Seed:")
print(results_df)

# Save to Excel (optional)
results_df.to_excel('25June_YE_1990_2000.xlsx', index=False)

results_df.to_excel('/content/drive/MyDrive/Colab_Notebooks/25June_YE_1990_2000.xlsx', index=False)

import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for better box plot aesthetics

#Box plot for all parameters
for param in param_names:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=results_df[param])
    plt.title(f'Box Plot of {param}')
    plt.ylabel(param)
    plt.show()

representative_params = {}
for param in param_names:
    # Choose the median as the representative value
    representative_params[param] = results_df[param].median()

# Print representative parameters
print("Representative Parameters:")
for param, value in representative_params.items():
    print(f"{param}: {value}")

# Combine fixed parameters with representative parameters
final_params = {**fixed_params, **representative_params}

# Run the simulation with representative parameters
optimized_df = reservoir_simulation(seasonal_df.copy(), **final_params)

# Calculate NSE for cultivation area
observed_cultivation_area = seasonal_df['Observed_cultivation_area'].to_numpy()
optimized_cultivation_area = optimized_df['Cultivation_area'].to_numpy()
nse_cultivation_area = 1 - (np.sum((observed_cultivation_area - optimized_cultivation_area)**2) / np.sum((observed_cultivation_area - np.mean(observed_cultivation_area))**2))
print(f'Optimized NSE (Cultivation Area): {nse_cultivation_area}')

# Plot the observed and calculated cultivation areas
plt.figure(figsize=(10, 6))
plt.plot(seasonal_df.index, observed_cultivation_area, label='Observed Cultivation Area', color='blue')
plt.plot(seasonal_df.index, optimized_cultivation_area, label='Calculated Cultivation Area', color='red')
plt.xlabel('Month')
plt.ylabel('Cultivation area (ha)')
plt.legend()
plt.title(f'Observed vs. Calculated Cultivation Area\nNSE: {nse_cultivation_area:.3f}')
plt.show()

# Calculate NSE for observed demand vs. withdrawal
observed_withdrawal = seasonal_df['Observed_demand'].to_numpy()
optimized_withdrawal = optimized_df['Withdrawal'].to_numpy()
nse_withdrawal = 1 - (np.sum((observed_withdrawal - optimized_withdrawal)**2) / np.sum((observed_withdrawal - np.mean(observed_withdrawal))**2))
print(f'Optimized NSE (Withdrawal): {nse_withdrawal}')

# Plot the observed demand vs. withdrawal
plt.figure(figsize=(10, 6))
plt.plot(seasonal_df.index, observed_withdrawal, label='Observed Demand', color='blue')
plt.plot(seasonal_df.index, optimized_withdrawal, label='Withdrawal', color='red')
plt.xlabel('Month')
plt.ylabel('Water Volume')
plt.legend()
plt.title(f'Observed Demand vs. Withdrawal\nNSE: {nse_withdrawal:.3f}')
plt.show()

# Calculate NSE for observed and simulated storage
observed_storage = seasonal_df['Observed_storage'].to_numpy()
optimized_storage = optimized_df['Storage'].to_numpy()
nse_storage = 1 - (np.sum((observed_storage - optimized_storage)**2) / np.sum((observed_storage - np.mean(observed_storage))**2))
print(f'Optimized NSE (Storage): {nse_storage}')

# Plot the observed and calculated storage
plt.figure(figsize=(10, 6))
plt.plot(seasonal_df.index, observed_storage, label='Observed Storage', color='blue')
plt.plot(seasonal_df.index, optimized_storage, label='Calculated Storage', color='red')
plt.xlabel('Month')
plt.ylabel('Storage (mcm)')
plt.legend()
plt.title(f'Observed vs. Calculated Storage\nNSE: {nse_storage:.3f}')
plt.show()

optimized_df[['Storage', 'Cultivation_area', 'Withdrawal']].to_excel('/content/drive/MyDrive/Colab_Notebooks/25June_Results_YE_P1', index=True)
