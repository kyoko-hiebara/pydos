# pydos

# Description
Based on pydos developed by QijingZheng.
<br>This program is edited to export DOS data in csv format (pydos_csv.py) and plot it (pydos_plot.py).
<br>[pydos original version](https://github.com/QijingZheng/pyband)

# Changes to the pydos Code

Add the following code after the section "def generateDos(opts):" in pydos.

```
import pandas as pd

def save_dos_to_csv(xen, tdos, pdos, filename="dos_data.csv"):
    # xen (energy values) will be the first column
    data = {"Energy": xen}

    # Adding total DOS data to the dataframe
    for i in range(tdos.shape[1]):
        data[f"Total DOS (Spin {i+1})"] = tdos[:, i]

    # Adding PDOS data to the dataframe
    for i, pdos_data in enumerate(pdos):
        for j in range(pdos_data.shape[1]):
            data[f"PDOS {i+1} (Spin {j+1})"] = pdos_data[:, j]

    # Creating a DataFrame and saving it to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename
```
Add this code to the last line of the entire pydos.

```
xen, tdos, pdos = generateDos(opts)
filename = save_dos_to_csv(xen, tdos, pdos, "dos_data.csv")
```

# Plot DOS
