import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

max_epochs = 100
def set_academic_style():
    #Configure matplotlib for a professional, academic presentation.
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "serif" 
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12


def extract_accuracies_energies(file_path):
    #Extract a list of accuracies from a log file, energies are not used.

    accuracies = []
    energies = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Final accuracies:" in line:
                items_str = line.split(':')[-1].strip().strip('[]')
                if items_str:
                    accuracies = list(map(float, items_str.split(',')))
                break
    # Convert to % if in [0,1]
    if accuracies and max(accuracies) <= 1.0:
        accuracies = [a * 100 for a in accuracies]

    if len(accuracies) != max_epochs and len(accuracies) > 0:
         print(f"Info: Trial {os.path.basename(file_path)} has fewer than {max_epochs} accuracy entries ({len(accuracies)} found). Training might have been interrupted or finished early.")
    elif not accuracies:
         print(f"Warning: No accuracy data found or parsed in {os.path.basename(file_path)}.")

    epochs = list(range(1, len(accuracies) + 1))
    return epochs, energies, accuracies

def calculate_convergence(file_path, threshold, patience):

    #Define 'convergence epoch' as the first epoch i where
    #abs(acc[i] - acc[i-1]) <= threshold for 'patience' consecutive epochs.

    epochs, energies, accuracies = extract_accuracies_energies(file_path)
    consecutive_count = 0
    for i in range(1, len(accuracies)):
        if i > max_epochs:
            break
        if abs(accuracies[i] - accuracies[i - 1]) <= threshold:
            consecutive_count += 1
            if consecutive_count >= patience:
                return epochs[i], accuracies, energies
        else:
            consecutive_count = 0
    # If never converged, return final epoch
    return max_epochs, accuracies, energies

def process_directory(directory, threshold, patience):

    #Processes all .log files in 'directory' to get convergence epoch and final accuracy.
    #Sorting is done alphabetically.

    results = []
    final_accs = []
    for fname in os.listdir(directory):
        if fname.endswith(".log"):
            path = os.path.join(directory, fname)
            conv_epoch, accuracies, _ = calculate_convergence(path, threshold, patience)
            results.append((fname, conv_epoch))
            if conv_epoch is not None and conv_epoch <= len(accuracies) and len(accuracies) > 0:
                final_accs.append((fname, accuracies[conv_epoch - 1]))
            else:
                final_accs.append((fname, None))
    
    # Sort alphabetically
    results.sort(key=lambda x: x[0])
    final_accs.sort(key=lambda x: x[0])
    
    convs = np.array([r[1] for r in results], dtype=np.float32)
    accs  = np.array([fa[1] for fa in final_accs], dtype=np.float32)
    return convs, accs

def gather_bptt_data(bptt_dir, hidden_sizes, threshold, patience):
    #For each folder 'lam=..., eta=...' in bptt_dir:
    #  - For each logs_<hidden_size> subfolder,
    #  - For each batch_{X} inside that logs_* folder,
    #  Process .log files, and accumulate data into dictionaries.

    data_acc = {}
    data_conv = {}
    
    for cond_name in sorted(os.listdir(bptt_dir)):
        cond_path = os.path.join(bptt_dir, cond_name)
        if not os.path.isdir(cond_path):
            continue

        for hs in hidden_sizes:
            logs_folder = f"logs_{hs}"
            logs_path = os.path.join(cond_path, logs_folder)
            if not os.path.isdir(logs_path):
                continue

            for item in sorted(os.listdir(logs_path)):
                if item.startswith("batch_"):
                    batch_dir = os.path.join(logs_path, item)
                    if not os.path.isdir(batch_dir):
                        continue
                    
                    group_name = f"H{hs}_{item}"
                    if group_name not in data_acc:
                        data_acc[group_name] = {}
                        data_conv[group_name] = {}
                    data_acc[group_name].setdefault(cond_name, [])
                    data_conv[group_name].setdefault(cond_name, [])

                    convs, accs = process_directory(batch_dir, threshold, patience)
                    data_acc[group_name][cond_name].extend(accs)
                    data_conv[group_name][cond_name].extend(convs)
    
    for grp in data_acc:
        for cnd in data_acc[grp]:
            data_acc[grp][cnd]  = np.array(data_acc[grp][cnd], dtype=np.float32)
            data_conv[grp][cnd] = np.array(data_conv[grp][cnd], dtype=np.float32)
    
    return data_acc, data_conv


# Helper functions for formatting labels in plot
def format_group_label(group):
    if group.startswith("STDP_"):
        m = re.search(r'batch_(\d+)', group)
        if m:
            return f"STDP (Batch = {m.group(1)})"
        else:
            return "STDP"
    else:
        m = re.search(r'batch_(\d+)', group)
        if m:
            return f"BPTT (Batch = {m.group(1)})"
        else:
            return group

def format_condition_label(cond):
    mapping = {
        "lam=0.0-eta=0.0": "Non-coupled",
    }
    return mapping.get(cond, cond)

def limit_and_jitter_points(data, max_per_value=20, jitter_scale=0.01):
    #Cap repeated occurrences of the same data value (max_per_value).
    #Add small vertical jitter to avoid exact overlap.

    data = np.array(data)
    if len(data) == 0:
        return data, data, np.arange(0)
    
    idx_sorted = np.argsort(data)
    data_sorted = data[idx_sorted]
    
    unique_vals, counts = np.unique(data_sorted, return_counts=True)
    kept_indices = []
    start_idx = 0
    for val, cnt in zip(unique_vals, counts):
        slice_indices = np.arange(start_idx, start_idx + cnt)
        slice_cnt = min(cnt, max_per_value)
        kept_indices.extend(slice_indices[:slice_cnt])
        start_idx += cnt
    
    kept_indices = np.array(kept_indices)
    data_limited = data_sorted[kept_indices]
    data_min, data_max = np.min(data_limited), np.max(data_limited)
    data_range = max(data_max - data_min, 1e-6)
    vertical_jitter = (np.random.rand(len(data_limited)) - 0.5) * 2.0 * (jitter_scale * data_range)
    data_jittered = data_limited + vertical_jitter
    original_indices = idx_sorted[kept_indices]
    
    return data_limited, data_jittered, original_indices

# helper function to match experiments with same random seed together
def sort_key_func(group_name):
        match = re.search(r'(?:STDP_)?H(\d+)_batch_(\d+)', group_name)
        if match:
            hidden_size_str, batch_size_str = match.groups()
            try:
                # Convert both captured parts to integers for numerical sorting
                return (int(hidden_size_str), int(batch_size_str))
            except ValueError:
                # Handle cases where conversion might fail unexpectedly
                print(f"Warning: Could not convert parts of '{group_name}' to integers.")
                return (999999, 999999) # Sort problematic ones last
        else:
            # Fallback for keys that don't match the expected pattern
            print(f"Warning: Group name '{group_name}' did not match expected pattern for sorting.")
            return (999999, 999999) # Sort unrecognized ones last

def plot_violin_comparison_all_models(
    models_data,
    y_label="",
    title="",
    save_dir="",
    group_spacing=2,
    offset_step=0.3,
    max_points_per_value=3,
    vertical_jitter_scale=0.01
):
    #Plots violin plots for each group in models_data.

    set_academic_style()
    
    group_names = list(models_data.keys())
    # Sort group names by hidden size + batch
    group_names.sort(key=sort_key_func)

    
    formatted_group_labels = [format_group_label(g) for g in group_names]
    
    # Collect all conditions
    all_conditions = set()
    for grp in group_names:
        for cond in models_data[grp]:
            all_conditions.add(cond)
    conditions = sorted(all_conditions)
    
    # Assign colors to conditions
    palette = plt.get_cmap("tab10")
    condition_colors = {cond: palette(i % palette.N) for i, cond in enumerate(conditions)}
    
    all_values = []
    for grp in group_names:
        for cond in models_data[grp]:
            all_values.extend(models_data[grp][cond])
    all_values = np.array(all_values)
    if len(all_values) == 0:
        print("No data found to plot.")
        return
    
    global_min = np.nanmin(all_values)
    global_max = np.nanmax(all_values)
    global_range = max(global_max - global_min, 1e-6)
    
    fig, ax = plt.subplots(figsize=(max(10, len(group_names)*1.5), 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    xticks = []
    # For each group (Hxx_batch_yy), plot a cluster of violins (one per condition).
    for i, grp in enumerate(group_names):
        group_center_x = i * group_spacing
        n_conditions = len(conditions)
        half = (n_conditions - 1) / 2.0
        offsets = [(j - half) * offset_step for j in range(n_conditions)]
        
        for j, cond in enumerate(conditions):
            data_cnd = models_data[grp].get(cond, np.array([]))
            x_pos = group_center_x + offsets[j]
            if len(data_cnd) > 0:
                data_capped, data_jittered, _ = limit_and_jitter_points(
                    data_cnd,
                    max_per_value=max_points_per_value,
                    jitter_scale=vertical_jitter_scale
                )
                # Plot violin for this group + condition
                vp = ax.violinplot([data_cnd], positions=[x_pos], widths=0.5, showmedians=True)
                
                # Customize the violin's appearance
                for part in ('cbars', 'cmins', 'cmaxes'):
                    vp[part].set_edgecolor('black')
                    vp[part].set_linewidth(1.0)
                vp['cmedians'].set_edgecolor('black')
                vp['cmedians'].set_linewidth(1.0)
                vp['cmedians'].set_zorder(5)
                
                # Body color
                for pc in vp['bodies']:
                    color = condition_colors[cond]
                    pc.set_facecolor(color)
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.8)
                
                # Add jittered points
                ax.scatter(
                    np.full(len(data_capped), x_pos) + np.random.normal(0, 0.05, size=len(data_capped)),
                    data_jittered,
                    color='black',
                    alpha=0.15,
                    s=8,
                    zorder=3
                )
        xticks.append(group_center_x)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(formatted_group_labels, rotation=45, ha='right')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    # Dynamic range for metrics OTHER than Convergence Epoch
    dynamic_lower = global_min - 1
    dynamic_upper = global_max + 0.05 * global_range
    
    if y_label == "Convergence Epoch":
        # Force the axis from 5 to 101 (enough margin above 100 and below 6)
        ax.set_ylim(5, 101)
        
        # Put dashed lines exactly at y=6 and y=100
        ax.axhline(y=11, color='gray', linestyle='--', linewidth=1.0, zorder=2.5)
        ax.axhline(y=max_epochs, color='gray', linestyle='--', linewidth=1.0, zorder=2.5)
        
        # Set custom ticks so 6 and 100 align perfectly with the grid
        ax.set_yticks([11, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    else:
        # For final accuracy or other metrics, use dynamic range
        ax.set_ylim(dynamic_lower, dynamic_upper)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=condition_colors[cond], label=format_condition_label(cond))
        for cond in conditions
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False
    )
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname_base = title.replace(" ", "_").replace("=", "")
        fig.savefig(os.path.join(save_dir, f"violin_{fname_base}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(save_dir, f"violin_{fname_base}.eps"), dpi=300, bbox_inches='tight')

    plt.show()


import re

def plot_hidden_sizes_for_batch(models_data, batch_str, y_label="", title="", save_dir=""):
    # Creates violin plots comparing different hidden layer sizes for a specific batch of data
    
    # Extract the numeric part of the target batch string (e.g., "1" from "batch_1")
    target_batch_num_match = re.search(r'batch_(\d+)', batch_str)
    if not target_batch_num_match:
        print(f"Error: Could not extract numeric batch number from '{batch_str}'")
        return
    target_batch_num = target_batch_num_match.group(1)

    filtered_data = {}
    
    for k, v in models_data.items():
        if re.search(rf'_batch_{target_batch_num}$', k):
             filtered_data[k] = v

    if not filtered_data:
        print(f"Warning: No data found matching the exact batch identifier '{batch_str}'")
        return

    # Create a new dictionary using "Hidden = {hs}" as keys.
    new_data = {}
    for key, cond_dict in filtered_data.items():
        # Now extract hidden size reliably from the correctly filtered keys
        m = re.search(r'H(\d+)_batch_', key) # This regex is fine here
        if not m:
            continue
        hs = int(m.group(1))
        label = f"Hidden = {hs}"
        if label not in new_data:
            new_data[label] = {}
        # Merge conditions (rest of the logic is likely okay, but ensure robust concatenation)
        for cond, arr in cond_dict.items():
            # Ensure arr is a numpy array before concatenating
            valid_arr = arr if isinstance(arr, np.ndarray) else np.array([])

            if cond in new_data[label]:
                 # Ensure existing data is numpy array
                existing_arr = new_data[label][cond] if isinstance(new_data[label][cond], np.ndarray) else np.array([])
                # Concatenate valid arrays
                if existing_arr.size > 0 or valid_arr.size > 0:
                    new_data[label][cond] = np.concatenate((existing_arr, valid_arr))
                else: # both empty
                    new_data[label][cond] = np.array([]) # Keep it as empty array
            else:
                new_data[label][cond] = valid_arr # Assign the valid (potentially empty) array


    # Sort new_data keys numerically (by hidden size)
    if not new_data:
        print(f"Warning: No data aggregated for plotting for batch '{batch_str}'")
        return

    sorted_keys = sorted(new_data.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    sorted_new_data = {k: new_data[k] for k in sorted_keys}
    print(f"--- Plotting for {batch_str} ---") 
    # Check the specific data for H100 just before plotting
    if 'Hidden = 100' in sorted_new_data:
        pass
    else:
         print("'Hidden = 100' not found in final data for plotting.")


    # Plot using the existing violin plotting function.
    plot_violin_comparison_all_models(
        sorted_new_data,
        y_label=y_label,
        title=title,
        save_dir=save_dir
    )


def calculate_speedup_data(combined_conv_data, baseline_condition="lam=0.0-eta=0.0"):
    #Calculates the convergence speedup relative to a baseline condition based on median epochs.

    speedup_data = {}
    found_baseline_overall = False

    print(f"\n--- Calculating Median Speedup relative to Baseline: '{baseline_condition}' ---")

    for group, conditions_data in combined_conv_data.items():
        baseline_epochs_raw = conditions_data.get(baseline_condition)

        # Skip group if baseline missing or empty
        if baseline_epochs_raw is None or not isinstance(baseline_epochs_raw, np.ndarray) or baseline_epochs_raw.size == 0:
            continue 

        baseline_epochs = baseline_epochs_raw[~np.isnan(baseline_epochs_raw)]
        # Skip group if baseline has only NaNs
        if baseline_epochs.size == 0:
            continue

        found_baseline_overall = True
        median_baseline = np.median(baseline_epochs) 

        speedup_data[group] = {} 

        for comp_cond, comp_epochs_raw in conditions_data.items():
            # Skip comparing baseline to itself
            if comp_cond == baseline_condition:
                continue 

            # Skip comparison if comparison data missing/empty
            if comp_epochs_raw is None or not isinstance(comp_epochs_raw, np.ndarray) or comp_epochs_raw.size == 0:
                continue 

            # Remove NaNs from comparison epochs
            comp_epochs = comp_epochs_raw[~np.isnan(comp_epochs_raw)]
            if comp_epochs.size == 0:
                 continue

            median_comparison = np.median(comp_epochs)
            # Calculate speedup as difference in medians
            speedup = median_baseline - median_comparison 
            speedup_data[group][comp_cond] = speedup

        # Clean up group entry if no valid comparisons were made
        if not speedup_data[group]:
            del speedup_data[group]

    # If no valid speedup data was found, print an error message
    if not found_baseline_overall:
         print(f"ERROR: Baseline condition '{baseline_condition}' not found in any group. Speedup calculation failed.")
         return {}
    if not speedup_data:
         print(f"WARNING: No valid speedup data could be calculated (check comparison conditions).")
         return {}

    print("--- Finished Speedup Calculation ---")
    return speedup_data

def plot_speedup_comparison(
    speedup_data,
    title="Convergence Speedup vs. Non-Coupled",
    y_label="Median Epochs Faster than Non-Coupled",
    save_dir="",
    group_spacing_factor=0.8 # Controls space between groups of bars
):

    #Plots the calculated convergence speedup using a bar chart.
    set_academic_style() 

    # Check if speedup_data is empty or None
    if not speedup_data:
        print(f"Error in plot_speedup_comparison: No speedup data provided for title '{title}'. Skipping plot.")
        return

    group_names = list(speedup_data.keys())
    # Sort group names by hidden size + batch
    try:
        group_names.sort(key=sort_key_func) 
    except NameError:
        print("Warning: sort_key_func not defined globally. Sorting groups alphabetically.")
        group_names.sort()

    all_comparison_conditions = set()
    for group in group_names:
        # Collect all unique comparison conditions across groups
        all_comparison_conditions.update(speedup_data[group].keys())

    if not all_comparison_conditions:
        # If no comparison conditions found, skip plotting
        print(f"Error in plot_speedup_comparison: No comparison conditions found in speedup data for '{title}'. Skipping plot.")
        return

    comparison_conditions = sorted(list(all_comparison_conditions))
    num_comp_conditions = len(comparison_conditions)

    # Assign colors
    palette = plt.get_cmap("tab10") 
    try:
      # Use format_condition_label if available
      condition_colors = {cond: palette(i % palette.N) for i, cond in enumerate(comparison_conditions)}
      condition_labels = {cond: format_condition_label(cond) for cond in comparison_conditions}
    except NameError:
       print("Warning: format_condition_label not defined globally. Using raw condition keys.")
       condition_colors = {cond: palette(i % palette.N) for i, cond in enumerate(comparison_conditions)}
       condition_labels = {cond: cond for cond in comparison_conditions}


    # Bar Plot Setup
    num_groups = len(group_names)
    # Adjust bar width and spacing based on number of conditions to compare

    # Max portion of group space bars can occupy
    total_bar_width_per_group = 0.8 
    bar_width = total_bar_width_per_group / num_comp_conditions
    # Calculate group centers, leaving space between groups
    group_centers = np.arange(num_groups) * (total_bar_width_per_group * group_spacing_factor + bar_width * 2) # Add more spacing


    fig, ax = plt.subplots(figsize=(max(10, num_groups * num_comp_conditions * bar_width * 1.2), 6)) # Adjust figsize
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plot_successful = False
    all_speedup_values = []
    legend_handles = {}

    # Plot Bars
    for i, group in enumerate(group_names):
        group_center_x = group_centers[i]
        # Calculate offsets from group center
        offsets = (np.arange(num_comp_conditions) - (num_comp_conditions - 1) / 2) * bar_width

        for j, cond in enumerate(comparison_conditions):
            speedup_value = speedup_data[group].get(cond, 0) 
            bar_pos = group_center_x + offsets[j]
            color = condition_colors.get(cond, 'gray')
            label = condition_labels.get(cond, cond)

            if speedup_value is not None and not np.isnan(speedup_value):
                 bar = ax.bar(bar_pos, speedup_value, bar_width, color=color, edgecolor='black', linewidth=0.5)
                 if label not in legend_handles:
                     legend_handles[label] = bar
                 plot_successful = True
                 all_speedup_values.append(speedup_value)

    if not plot_successful:
        print(f"Warning in plot_speedup_comparison: No valid speedup data points were plotted for title '{title}'.")
        ax.text(0.5, 0.5, 'No valid speedup data to display', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='red')

    # Formatting
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(group_centers)

    try:
        formatted_group_labels = [format_group_label(g) for g in group_names]
        ax.set_xticklabels(formatted_group_labels, rotation=45, ha='right')
    except NameError:
        print("Warning: format_group_label not defined globally. Using raw group names.")
        ax.set_xticklabels(group_names, rotation=45, ha='right')

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

    # Dynamic Y-limits
    if all_speedup_values:
        min_val, max_val = min(all_speedup_values), max(all_speedup_values)
        y_range = max(abs(min_val), abs(max_val))
        y_buffer = y_range * 0.1
        ax.set_ylim(-(y_range + y_buffer + 1), y_range + y_buffer + 1) 
    else:
        ax.set_ylim(-5, 5)

    # Legend
    if legend_handles:
        ax.legend(
            handles=legend_handles.values(), 
            labels=legend_handles.keys(),    
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            title="Compared Condition"
        )

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save and Show
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title) # Sanitize filename
            fname_base = f"speedup_{safe_title.replace(' ', '_').replace('=', '').replace(':', '').replace('(', '').replace(')', '')}"
            plot_path_png = os.path.join(save_dir, f"{fname_base}.png")
            plot_path_eps = os.path.join(save_dir, f"{fname_base}.eps")

            print(f"Saving speedup plot to {plot_path_png}")
            fig.savefig(plot_path_png, dpi=300, bbox_inches='tight')
            print(f"Saving speedup plot to {plot_path_eps}")
            fig.savefig(plot_path_eps, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving speedup plot '{title}' to {save_dir}: {e}")

    plt.show()
    plt.close(fig)


def compute_speedup_by_hidden_size(data_conv,
                                   baseline_cond="lam=0.0-eta=0.0",
                                   test_cond="lam=0.0-eta=1.0",
                                   use_mean=False,
                                   with_error=False):

    #Compute speedup (%) between two conditions.

    speedup_dict = {}
    # Choose aggregation function (mean or median)
    aggregator = np.mean if use_mean else np.median

    for group_name, cond_dict in data_conv.items():
        # Skip if either baseline or test condition is missing
        if baseline_cond not in cond_dict or test_cond not in cond_dict:
            continue

        # baseline convergence epochs
        bdata = cond_dict[baseline_cond]
        # test convergence epochs
        tdata = cond_dict[test_cond]     
        # Extract hidden size and batch size from group name using regex
        # Pattern matches: "H100_batch_10" or "STDP_H100_batch_10"
        m = re.search(r'(?:STDP_)?H(\d+)_batch_(\d+)', group_name)
        if not m:
            continue
        hs    = int(m.group(1))    
        batch = m.group(2)         

        # Create method identifier (stdp or batch_X)
        method_str = "stdp" if group_name.startswith("STDP_") else f"batch_{batch}"

        if with_error:
            # Calculate speedup for each paired run and compute error bars
            paired = []
            for b, t in zip(bdata, tdata):
                if b > 0:  
                    # Speedup formula: (baseline - test) / baseline * 100%
                    paired.append(((b - t) / b) * 100.0)
            if not paired:
                continue

            central = aggregator(paired)  # mean or median speedup
            if use_mean:
                # Standard error for mean
                sd  = np.std(paired, ddof=1)
                sem = sd / np.sqrt(len(paired))
                err_low, err_high = sem, sem
            else:
                # Interquartile range for median
                lower, upper = np.percentile(paired, [25, 75])
                err_low  = central - lower
                err_high = upper - central

            speedup_dict[(hs, method_str)] = (central, err_low, err_high)

        else:
            # Simple speedup calculation using aggregated values
            baseline_stat = aggregator(bdata)  # aggregate baseline epochs
            test_stat     = aggregator(tdata)  # aggregate test epochs
            if baseline_stat <= 0:
                continue
            # Calculate percentage speedup
            speedup = ((baseline_stat - test_stat) / baseline_stat) * 100.0
            speedup_dict[(hs, method_str)] = speedup

    return speedup_dict

def plot_speedup_lines(speedup_dict, 
                        title="Convergence Speedup Comparison Across Hidden Layer Sizes", 
                        save_dir="", 
                        methods_order=("batch_1", "batch_10", "batch_50", "batch_100", "stdp"),
                        with_error=False):

        #Plot convergence speedup curves as a function of hidden layer size with optional error bars.
        set_academic_style()

        # Extract and sort unique hidden sizes from the keys
        hidden_sizes = sorted(set(hs for (hs, _) in speedup_dict.keys()))
        print(hidden_sizes)
        print(speedup_dict)
        if not hidden_sizes:
            print("No speedup data available for plotting.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Remove top and right spines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Define a mapping for more descriptive legend labels
        legend_mapping = {
            "batch_1": "BPTT (Batch Size = 1)",
            "batch_10": "BPTT (Batch Size = 10)",
            "batch_50": "BPTT (Batch Size = 50)",
            "batch_100": "BPTT (Batch Size = 100)",
            "stdp": "STDP"
        }
        
        if with_error:
            # Draw vertical dashed lines at the center for each hidden size
            for hs in hidden_sizes:
                ax.axvline(x=hs, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            n_methods = len(methods_order)
            # Set a total width for each group (the full width in x-direction around each hidden size)
            group_width = 3
            # Compute evenly spaced offsets based on the number of methods
            offsets = np.linspace(-group_width/2, group_width/2, n_methods)
            
            # Plot error bars for each method
            for i, method in enumerate(methods_order):
                # Compute x positions automatically using the computed offset
                x_positions = [hs + offsets[i] for hs in hidden_sizes]
                central_vals = []
                err_low_vals = []
                err_high_vals = []
                for hs in hidden_sizes:
                    key = (hs, method)
                    if key in speedup_dict:
                        central, err_low, err_high = speedup_dict[key]
                        central_vals.append(central)
                        err_low_vals.append(err_low)
                        err_high_vals.append(err_high)
                    else:
                        central_vals.append(np.nan)
                        err_low_vals.append(0)
                        err_high_vals.append(0)
                if not all(np.isnan(val) for val in central_vals):
                    ax.errorbar(x_positions, central_vals, 
                                yerr=[err_low_vals, err_high_vals], 
                                marker='o', 
                                label=legend_mapping.get(method, method))
        else:
            # Plot each method's speedup curve in the specified order (simple lines)
            for method in methods_order:
                y_vals = []
                for hs in hidden_sizes:
                    val = speedup_dict.get((hs, method), np.nan)
                    y_vals.append(val)
                if not np.all(np.isnan(y_vals)):
                    ax.plot(hidden_sizes, y_vals, marker='o', label=legend_mapping.get(method, method))
        
        # Label axes with professional formatting
        ax.set_xlabel("Hidden Layer Size", fontweight='bold')
        ax.set_ylabel("Convergence Speedup (%)", fontweight='bold')
        ax.set_title(title, fontweight='bold')
        
        # Place the legend outside the plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname_base = title.replace(" ", "_").replace("=", "")
            fig.savefig(os.path.join(save_dir, f"line_{fname_base}.png"), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_dir, f"line_{fname_base}.eps"), dpi=300, bbox_inches='tight')

        plt.show()
