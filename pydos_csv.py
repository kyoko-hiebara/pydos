import pandas as pd

#add to under "def generateDos(opts):" in pydos#
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

#Add to the last line in pydos.#
xen, tdos, pdos = generateDos(opts)
filename = save_dos_to_csv(xen, tdos, pdos, "dos_data.csv") 
