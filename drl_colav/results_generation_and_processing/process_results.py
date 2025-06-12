import os
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import statistics
from pathlib import Path

all_scores_list = ["r_cpa", "S_safety", "S_safety_theta", "S_safety_r", "avg_eval_result", "eval_reward",
                    "S_13", "P_13_ahead", "S_14", "P_14_sts", "P_14_nsb", "S_15", "P_15_ahead",
                    "S_16", "P_16_na_man", "P_16_na_delta_chi", "P_16_na_delta_v", "P_delay", "S_17"
                    ]

# Scores that are relevant regardless of situation type
relevant_scores_common = ["r_cpa", "S_safety", "S_safety_theta", "S_safety_r", "avg_eval_result", "eval_reward"]

# Relevant scores based on the situation type
# Note: OTSO not present in imazu cases
relevant_scores_by_situation = {
    "HO"  : ["S_14", "P_14_sts", "P_14_nsb", "P_16_na_delta_chi",
             "P_delay", "S_16", "P_16_na_man",
             "P_16_na_delta_v", "P_16_na_delta_chi"],
    "CRSO": ["S_15", "S_17"],
    "CRGW": ["S_15", "P_15_ahead", "S_16", "P_delay", "P_16_na_man",
             "P_16_na_delta_v", "P_16_na_delta_chi"],
    "OTGW": ["S_13", "P_13_ahead", "S_16", "P_delay", "P_16_na_man",
             "P_16_na_delta_v", "P_16_na_delta_chi"],
    "OTSO": ["S_13", "S_17"]
}

PARAM_COLORS = {
    "K_COLL_": "blue",
    "K_CHI_": "green",
    "Q_": "red",
    "P_": "purple",
    "KAPPA_": "orange",
    "K_DCHI_SB_": "cyan",
    "K_DCHI_P_": "magenta",
}

def add_prefix_to_files(folder_path : Path, prefix : str):
    for file_name in os.listdir(folder_path):
        if not file_name.startswith(prefix):
            new_name = prefix + file_name
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
            print(f"Renamed {file_name} to {new_name}")

def extract_scenario(file_name : str):
    if "imazu" in file_name:
        match = re.search(r"imazu\d{2}", file_name)
        return match.group(0) if match else None
    else:
        return file_name.split(".yaml")[0]

def compare_models_by_scenario(results: dict, output_folder: Path = None):
    if output_folder is None:
        output_folder = Path("model_comparisons_by_scenario")
    os.makedirs(output_folder, exist_ok=True)
    
    for scenario, data in results.items():
        df = pd.DataFrame(data)
        # Group by model and compute averages
        comparison = df.groupby("model").mean(numeric_only=True)
        output_file = os.path.join(output_folder, f"{scenario}_comparison.xlsx")
        comparison.to_excel(output_file)
        print(f"Saved comparison for scenario {scenario} to {output_file}")

# One model in one scenario
def process_result_file(file_path: Path):
    df = pd.read_excel(file_path, header=None)
    ship_columns = df.iloc[0, 1:]

    data = []
    for col_index, ship_name in enumerate(ship_columns, start=1):  # Start at column 1 (skip first column)
        situation = df.iloc[2, col_index]  # Extract the situation for this ship
        relevant_fields = relevant_scores_by_situation.get(situation, [])
        relevant_fields.extend(relevant_scores_common)

        ship_data = {"situation": situation}
        for field in relevant_fields:
            if field in df[0].values:
                row = df[df[0] == field].iloc[0]
                ship_data[field] = row[col_index]
        
        data.append(ship_data)

    return data

def process_all_models(results_folder: Path, models=None):
    model_results = {}
    scenario_results = {}
    
    for model_folder in os.listdir(results_folder):
        if models is not None and model_folder not in models:
            continue
        model_path = os.path.join(results_folder, model_folder)
        if not os.path.isdir(model_path):
            continue

        results = []
        for file_name in os.listdir(model_path):
            file_path = os.path.join(model_path, file_name)
            scenario = extract_scenario(file_name)
            ship_data = process_result_file(file_path)
            
            for data in ship_data:
                data["model"] = model_folder
                data["scenario"] = scenario
                
            results.extend(ship_data)
            
            if scenario not in scenario_results:
                scenario_results[scenario] = []
            scenario_results[scenario].extend(ship_data)

        model_results[model_folder] = pd.DataFrame(results)

    return model_results, scenario_results

def make_model_summaries(results: dict, output_folder: Path = None):
    # Compute averages and include the "All" row
    summary_results = compute_averages(results)

    # Save the summaries to Excel files
    if output_folder is None:
        output_folder = Path("model_summaries")
    os.makedirs(output_folder, exist_ok=True)

    for model, summary in summary_results.items():
        output_file = os.path.join(output_folder, f"{model}_summary.xlsx")
        summary.to_excel(output_file)
        print(f"Saved summary for {model} to {output_file}")

def compute_averages(results : dict):
    average_results = {}
    
    for model, df in results.items():
        numeric_df = df.select_dtypes(include="number")
         
        averages = numeric_df.groupby(df["situation"]).mean()
        counts = df.groupby(["situation"]).size().rename("count")
        
        average_results[model] = pd.concat([counts, averages], axis=1) # Include the counts for each type of situation
        
    return average_results

# Load already computed scenario comparisons
def load_scenario_comparisons(folder_path):
    scenario_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_comparison.xlsx"):
            #scenario = file_name.split("_comparison")[0]  # Extract the scenario name from the file name
            scenario = extract_scenario(file_name)
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path, index_col=0)  # Read the file with "model" as index
            scenario_data[scenario] = df
    return scenario_data

# Load already computed model summaries
def load_model_summaries(folder_path):
    model_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_summary.xlsx"):
            model = file_name.split("_summary")[0]  # Extract the model name from the file name
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path, index_col=0)  # Load the summary file
            model_data[model] = df
    return model_data

# Define a consistent color palette for the models
def get_model_colors(model_data):
    color_palette = sns.color_palette("tab10", n_colors=len(model_data))
    return {model: color for model, color in zip(model_data.keys(), color_palette)}

def plot_model_summaries_comparison(model_data, models=None, scores=all_scores_list, colors=None, output_folder=None, file_prefix="model_comparison"):
    plt.figure(figsize=(12, 8))

    # Plot each model's scores across categories
    for model, data in model_data.items():
        
        if models is not None and model not in models:
            continue
        
        data_mean = data.mean()
        # Reindex so categories are in the same order for all models
        data_mean = data_mean.reindex(scores, fill_value=None)
        data_mean = data_mean.dropna()  # Filter out NaN values that may arise from reindexing
        
        # Covert penalties so plotting with rewards looks consistent
        for score in data_mean.index:
            if score.startswith("P"):
                data_mean[score] = 1 - data_mean[score]
        
        x_values = data_mean.index
        y_values = data_mean.values
        
        filtered_data = [(x, y) for x, y in zip(x_values, y_values) if x in scores]
        filtered_x = [x for x, _ in filtered_data]
        filtered_y = [y for _, y in filtered_data]

        for i in range(len(filtered_x)):
            filtered_x[i] = f"1 - {filtered_x[i]}" if filtered_x[i].startswith("P") else filtered_x[i]
        
        if model == "standard" or model == "K_COLL 0.5":
            linestyle = "--"
            colors[model] = "black"
        else:
            linestyle = "-"

        plt.plot(filtered_x, filtered_y, marker='o', markersize = 10, linewidth=3.0, linestyle=linestyle, label=model, color=colors[model])

        print(f"Model {model} total mean score: {statistics.mean(filtered_y)}")
    
    #plt.title("Model Comparison Across Score Categories", fontsize=16)
    plt.xlabel("Score Category", fontsize=22)
    plt.ylabel("Average Score", fontsize=22)
    #plt.minorticks_on()
    plt.xticks(rotation=65, fontsize=18)
    plt.yticks(fontsize=20)
    legend = plt.legend(title="Model", fontsize=19, title_fontsize=22, loc=0)
    legend.get_frame().set_alpha(0.7)
    plt.grid(
        which="both",
        color="gray",
        linestyle="-",
        alpha=0.8
    )
    plt.tight_layout()
    if output_folder is None:
        output_folder = Path("figs")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(output_folder/f"{file_prefix}.pdf")

def plot_comparison_across_scenarios(scenario_data, score, models=None, colors=None, output_folder=Path("figs")):
    plt.figure(figsize=(12, 8))

    sorted_scenarios = sorted(
        scenario_data.keys(), key=lambda x: int(extract_scenario(x).replace("imazu", ""))
    )

    all_y_values = []

    all_models = scenario_data[list(scenario_data.keys())[0]].index

    for model in all_models:

        if models is not None and model not in models:
            continue

        x_values = []
        y_values = []

        for scenario in sorted_scenarios:
            data = scenario_data[scenario]
            if score in data.columns:
                # Convert penalties to rewards temporarily for plotting
                value = data.loc[model, score]
                if score.startswith("P"):
                    converted_value = 1 - value
                    value = converted_value
                x_values.append(scenario)
                y_values.append(value)

        all_y_values.extend(y_values)

        linestyle = "--" if model == "standard" else "-"
        model_color = colors.get(model, "black")

        plt.plot(
            x_values, y_values, marker='o', markersize=10, linewidth=3.0,
            linestyle=linestyle, label=model, color=model_color
        )

    # Adjust y-axis scaling
    min_y = min(all_y_values)
    max_y = max(all_y_values)
    plt.ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))

    plt.xlabel("Scenario", fontsize=22)
    plt.ylabel(f"1 - {score}" if score.startswith("P") else score, fontsize=22)
    legend = plt.legend(title="Model", fontsize=20, title_fontsize=22, loc=0)
    legend.get_frame().set_alpha(0.7)
    #plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(
        which="both",
        color="gray",
        linestyle="-",
        alpha=0.8
    )
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(output_folder / f"{score}_comparison.pdf")

def plot_params(file_path: Path, output_folder=Path("figs/params")):
    df = pd.read_csv(file_path)

    plt.figure(figsize=(13, 6))
    for col in df.columns:
        if col.lower() != "time":
            color = PARAM_COLORS.get(col, None)
            plt.plot(df["time"], df[col], label=col, color=color, linewidth=2.0)

    plt.xlabel("Time [s]", fontsize=28)
    plt.ylabel("Parameter Value", fontsize=28)
    legend = plt.legend(fontsize=22)
    legend.get_frame().set_alpha(0.7)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(output_folder / f"{extract_scenario(file_path.stem)}.pdf")
    plt.close()

if __name__ == "__main__":

    root = Path(__file__).parents[1]

    folder_name = "temp"

    results_folder = root /"results"/"by_model"/folder_name
    scenario_comparisons_folder = root /"processed_results"/folder_name
    model_summaries_folder = root /"processed_results"/"by_model"/folder_name/"model_summaries"

    model_data, scenario_data = process_all_models(results_folder)
    compare_models_by_scenario(scenario_data, output_folder=scenario_comparisons_folder)
    make_model_summaries(model_data, output_folder=model_summaries_folder)
    
    scores2 = ["S_safety", "S_safety_theta", "S_safety_r",
                    "S_13", "P_13_ahead", "S_14", "P_14_sts", "P_14_nsb", "S_15", "P_15_ahead",
                    "S_16", "P_16_na_man", "P_16_na_delta_chi", "P_16_na_delta_v", "P_delay", "S_17"
                    ]
    
    
    scores = []
    #scores = ["P_14_sts", "S_17"]
    
    #models = ["standard", "gen5_scratch_ACR_noTrajCost", "gen5_scratch_ACR_noTrajCost_300k_lowerTC_lowerKAPPAlimit"]
    models = None

    model_data = load_model_summaries(model_summaries_folder)
    scenario_data = load_scenario_comparisons(scenario_comparisons_folder)
    colors = get_model_colors(model_data)
    
    plot_model_summaries_comparison(model_data, models=models, scores=scores2, colors=colors, output_folder=root /"figs"/"by_model"/folder_name, file_prefix="model_comparison")
    for s in scores:
        plot_comparison_across_scenarios(scenario_data, s, models=models, colors=colors, output_folder=root /"figs"/"by_model"/folder_name)
    
    #"""