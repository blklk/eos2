import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import warnings
from scipy.optimize import OptimizeWarning

plt.rcParams['font.family'] = 'Arial'

# Define your x and y axis limits outside of the file_config:
defined_x_lim = (-3, 103)  # Example limits, adjust as needed
defined_y_lim = (120, 170)  # Example limits, adjust as needed

# Define your list of excel filenames
excel_files = ["diamond2023.xlsx", "desy2023-HT.xlsx", "ballaran2012.xlsx", "catalli2011.xlsx","wolf2015.xlsx", "tange2012.xlsx"]

file_config = {
    "diamond2023.xlsx": {
        "mgsio3": [
            {
                "x_column": "pressure",
                "y_columns": ["vol_brg"],
                "styles": [{"color": "b", "edgecolor": 'black', "marker": "o", "size": 115}],
                "labels": ["This Study - MgSiO$_3$"]  # Corresponding labels for "y_columns"
            },
        ],
        "fealo3": [
            {
                "x_column": "pressure",
                "y_columns": ["vol_brg"],
                "styles": [{"color": "r", "edgecolor": 'black', "marker": "o", "size": 115}],
                "labels": ["This Study - (Mg$_{0.68}$,Fe$_{0.32}$)(Si$_{0.70}$,Al$_{0.3}$)O$_3$"]
            },
        ]
    },
    #"desy2023-HT.xlsx": {
    #    "HT-comp": [
    #        {
    #            "x_column": "pressure",
    #            "y_columns": ["vol"],
    #            "styles": [{"color": "orange", "edgecolor": 'k', "marker": "o", "size": 55}],
    #            "labels": ["This study - HT: (Mg,Fe)(Si,Al)O$_3$ Brg w/ Garnet"]
    #        },
    #    ],
    #},
    "ballaran2012.xlsx": {
        "H3004": [
            {
               "x_column": "pressure",
                "y_columns": ["vol"],
                "styles": [{"color": "white", "edgecolor": 'blue', "marker": "<", "size": 105}],
                "labels": ["Boffa Ballaran et al. 2012 - MgSiO$_3$ (H3004)"]
            },
        ],
        "H3014": [
            {
               "x_column": "pressure",
                "y_columns": ["vol"],
                "styles": [{"color": "white", "edgecolor": 'green', "marker": "<", "size": 105}],
                "labels": ["Boffa Ballaran et al. 2012 - (Mg$_{0.96}$,Fe$_{0.04}$)SiO$_3$ (H3104)"]
            },
        ],
        "S4253": [
            {
               "x_column": "pressure",
                "y_columns": ["vol"],
                "styles": [{"color": "white", "edgecolor": 'red', "marker": "<", "size": 105}],
                "labels": ["Boffa Ballaran et al. 2012 - Mg$_{0.60}$Fe$_{0.03}^{2+}$Fe$_{0.38}^{3+}$Si$_{0.62}$Al$_{"
                           "0.36}$O$_3$ (S4253)"]
            },
        ],
    },
    "catalli2011.xlsx": {
        "mgfesialo3": [
            {
                "x_column": "pressure",
                "y_columns": ["vol"],
                "styles": [{"color": "white", "edgecolor": 'red', "marker": "^", "size": 105}],
                "labels": ["Catalli et al. 2011 - (Mg$_{0.88}$,Fe$_{0.13}$)(Si$_{0.88}$,Al$_{0.11}$)O$_3$"]
            },
        ],
    },
    "wolf2015.xlsx": {
        "ID1-300K": [
            {
                "x_column": "pressure",
                "y_columns": ["vol"],
                "styles": [{"color": "white", "edgecolor": 'green', "marker": "s", "size": 45}],
                "labels": ["Wolf et al. 2015 - RT: (Mg$_{0.88}$,Fe$_{0.13}$)SiO$_3$"]
            },
        ],
        #"ID2-LH": [
        #    {
        #        "x_column": "pressure",
        #        "y_columns": ["vol"],
        #        "styles": [{"color": "white", "edgecolor": 'green', "marker": "s", "size": 25}],
        #        "labels": ["Wolf et al. 2015 - LH: (Mg$_{0.88}$,Fe$_{0.13}$)SiO$_3$"]
        #    },
        #],
    },
    #"tange2012.xlsx": {
    #    "HT-tange": [
    #        {
    #            "x_column": "pressure",
    #            "y_columns": ["vol"],
    #            "styles": [{"color": "white", "edgecolor": 'blue', "marker": ">", "size": 25}],
    #            "labels": ["Tange et al. 2012 - LH: MgSiO$_3$"]
    #        },
    #    ],
    #},
}


# MASTER PLOT FUNCTION
def plot_sheet(df, configs, x_lim=None, y_lim=None):
    """
        Plot data from a given dataframe based on specified configurations.
        """
    for config in configs:
        x_data = df[config["x_column"]]
        for y_col, style, label in zip(config["y_columns"], config["styles"], config["labels"]):
            scatter_kwargs = {
                'color': style.get("color", None),
                'marker': style.get("marker", 'o'),
                's': style.get("size", None),
                'label': label,
                'zorder': 3,
                'clip_on': False
            }
            if 'edgecolor' in style:
                scatter_kwargs['edgecolor'] = style["edgecolor"]

            plt.scatter(x_data, df[y_col], **scatter_kwargs)

        if x_lim:
            plt.xlim(x_lim)

        if y_lim:
            plt.ylim(y_lim)

        # plt.title("Bridgmanite")
        plt.xlabel("Pressure (GPa)", size=14)
        plt.ylabel("Volume ($\AA^3$)", size=14)

        ax = plt.gca()  # Get current axis
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))  # Major ticks every 10 units
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))  # Minor ticks every 1 units
        # ax.xaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels

        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))  # Major ticks every 5 units
        # ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=1))  # Minor ticks every 1 units
        # ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels

        ax.tick_params(which='major', direction='out', length=5, labelsize=14)  # For major ticks
        # ax.tick_params(which='minor', direction='in', length=5)  # For minor ticks
        # plt.tick_params(bottom=True, top=True, left=True, right=True)


plt.figure(figsize=(8, 6))


# The loop to plot data from the files and sheets based on file_config
for excel_file, sheets in file_config.items():
    for sheet, configs in sheets.items():
        data = pd.read_excel(excel_file, sheet_name=sheet)
        plot_sheet(data, configs, defined_x_lim, defined_y_lim)


plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
############### plt.savefig("alldata-w-lit_plot.jpeg", bbox_inches='tight', dpi=900)
# Display the plot after plotting data from all sheets and files
plt.tight_layout()
############## plt.show(dpi=900)


# BEGIN EOS FITTING

# Birch-Murnaghan equation of state


def birch_murnaghan(V, V0, K0, K0_prime):
    """
    Birch-Murnaghan 3rd order equation of state.
    """
    # K0_prime = 4
    return (3 / 2) * K0 * ((V0 / V) ** (7 / 3) - (V0 / V) ** (5 / 3)) * (
                1 + 3 / 4 * (K0_prime - 4) * ((V0 / V) ** (2 / 3) - 1))


# Vinet equation of state (kept for future reference or usage)

def vinet(V, V0, K0, K0_prime):
    x = (V / V0) ** (1 / 3)
    eta = 3 / 2 * (K0_prime - 1)
    return 3 * K0 * (1 - x) * np.exp(eta * (1 - x))


def calculate_and_plot_eos(df, pressure_col, volume_col, color, linestyle, K0_prime=None, extrapolate=0.2, num_points=100):
    P = np.array(df[pressure_col])
    V = np.array(df[volume_col])
    initial_guess = [V[0], 165, K0_prime if K0_prime is not None else 4]

    # Define bounds for parameters; if K0_prime is provided, its bounds will be a single value
    bounds = (0, [np.inf, np.inf, K0_prime if K0_prime is not None else np.inf])

    # Fit the Birch-Murnaghan equation to the data
    try:
        popt_bm, pcov_bm = curve_fit(birch_murnaghan, V, P, p0=initial_guess, bounds=bounds, maxfev=500000)
        # Calculate the uncertainties as the square root of the diagonal of the covariance matrix
        V0_uncertainty = np.sqrt(pcov_bm[0, 0])
        K0_uncertainty = np.sqrt(pcov_bm[1, 1])
        K0_prime_uncertainty = np.sqrt(pcov_bm[2, 2])

        # Print the optimized parameters with their uncertainties
        print(f"Birch-Murnaghan EOS parameters and uncertainties:")
        print(f"V0 = {popt_bm[0]:.2f} ± {V0_uncertainty:.2f}")
        print(f"K0 = {int(popt_bm[1])} ± {int(K0_uncertainty)}")
        print(f"K0' = {popt_bm[2]:.1f} ± {K0_prime_uncertainty:.1f}")

        # Generate a range of V values for plotting that extends beyond the range of the data
        V_min_extrap = min(V) * (1 - extrapolate)
        V_max_extrap = max(V) * (1 + extrapolate)
        V_extrap = np.linspace(V_min_extrap, V_max_extrap, num_points)

        P_extrap = birch_murnaghan(V_extrap, *popt_bm)
        plt.plot(P_extrap, V_extrap, linestyle=linestyle, color=color) #,
                 # label=f'Birch-Murnaghan Fit: V0={popt_bm[0]:.3f}, K0={popt_bm[1]:.3f}, K0_prime={popt_bm[2]:.3f}')
    except Exception as e:
        print(f"Error in Birch-Murnaghan curve fitting for pressure column {pressure_col} and volume column {volume_col}: {e}")


all_handles, all_labels = plt.gca().get_legend_handles_labels()

# Here we define the datasets for which we wish to compute and plot EOS
eos_datasets = [
    {"filename": "diamond2023.xlsx",
     "sheet": "mgsio3",
     "pressure_col": "pressure",
     "volume_col": "vol_brg",
     "color": "mediumblue",
     "linestyle": "-"
     # "fixed_K0_prime": 4,
     # "label": "This Study - MgSiO$_3$"
     },
    {"filename": "ballaran2012.xlsx",
     "sheet": "H3004",
     "pressure_col": "pressure",
     "volume_col": "vol",
     "color": "blue",
     "linestyle": ":",
     # "label": "Boffa Ballaran et al. 2012 - H3004"
     },
    # {"filename": "desy2023-HT.xlsx",
     # "sheet": "HT-comp",
     # "pressure_col": "pressure",
     # "volume_col": "vol",
     # "color": "orange",
     # "linestyle": ":",
     # "label": "This Study - HT (Mg,Fe)(Si,Al)O$_3$ Brg w/ Garnet"
     # },
    {"filename": "diamond2023.xlsx",
     "sheet": "fealo3",
     "pressure_col": "pressure",
     "volume_col": "vol_brg",
     "color": "red",
     "linestyle": "-",
     # "label": "This Study - (Mg$_{0.68}$,Fe$_{0.32}$)(Si$_{0.70}$,Al$_{0.3}$)O$_3$"
     },
    {"filename": "catalli2011.xlsx",
     "sheet": "mgfesialo3",
     "pressure_col": "pressure",
     "volume_col": "vol",
     "color": "red",
     "linestyle": ":",
     # "label": "Catalli et al. 2011 - (Mg$_{0.88}$,Fe$_{0.13}$)(Si$_{0.88}$,Al$_{0.11}$)O$_3$"
     }
    # Add more datasets as needed...
]

plt.figure(figsize=(8, 6))


for dataset in eos_datasets:
    data = pd.read_excel(dataset["filename"], sheet_name=dataset["sheet"])
    if "fixed_K0_prime" in dataset:
        calculate_and_plot_eos(data, dataset["pressure_col"], dataset["volume_col"], dataset["color"],
                               dataset["linestyle"], K0_prime=dataset["fixed_K0_prime"])
    else:
        calculate_and_plot_eos(data, dataset["pressure_col"], dataset["volume_col"], dataset["color"],
                               dataset["linestyle"])
    data_numeric = data.select_dtypes(include=[np.number])
    data[data_numeric.columns] = data_numeric.interpolate()
    plot_sheet(data, file_config[dataset["filename"]][dataset["sheet"]], defined_x_lim, defined_y_lim)

    # calculate_and_plot_eos(data, dataset["pressure_col"], dataset["volume_col"], dataset["color"], dataset["linestyle"], extrapolate=0.2, num_points=100)
    # plt.legend(all_handles, all_labels, loc="upper left", bbox_to_anchor=(1, 1))

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_zorder(100)
ax.tick_params(axis='x', which='both', top=True, direction='in', length=5, labelsize=13)
ax.tick_params(axis='y', which='both', right=True, direction='in', length=5, labelsize=13)
# plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
plt.plot(27.1, 151.572, 'o', color='purple', markersize=11, markeredgecolor='black', label='DESY 2023')
plt.plot(38.6, 147.110, 'o', color='purple', markersize=11, markeredgecolor='black')
plt.legend(frameon=False, fontsize=14, loc="lower left")
plt.tight_layout()
plt.savefig("output_plot_with_eos.jpeg", bbox_inches='tight', dpi=900)
plt.show(dpi=900)
